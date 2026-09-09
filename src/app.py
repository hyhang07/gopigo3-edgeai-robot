import time
import threading
import io
import os
import sys
import logging
import numpy as np
import picamera
from flask import Flask, render_template_string, Response, jsonify
from easygopigo3 import EasyGoPiGo3
from tflite_runtime.interpreter import Interpreter

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ============================================================
# Core Parameters
# ============================================================
INSERT_ANGLE      = 75
CARRY_ANGLE       = 130
SAFE_STOP_DIST_MM = 150
AUTOPILOT_STOP_MM = 120
MAX_RPM           = 800
MODEL_PATH        = "best_float16.tflite"
INPUT_SIZE        = 640
CAM_W, CAM_H      = 480, 360
CAM_FPS           = 24
CONF_THRESHOLD    = 0.20
CENTER_TOLERANCE  = 36
CLASS_NAMES       = {0: "front_ring", 1: "diagonal_ring", 2: "side_ring"}
SPEED_GEARS       = {'1': 100, '2': 300, '3': 600, '4': 800}

# ============================================================
# Global State
# ============================================================
stop_event = threading.Event()
hw_lock    = threading.Lock()
state_lock = threading.Lock()

global_dist        = 9999
auto_brake_enabled = True
auto_pilot         = False
current_speed      = 300
current_gear       = '2'
s1_angle           = 90
s2_angle           = 90

# ── AI Inference Results ─────────────────────────────────────
ai_lock = threading.Lock()
ai_result = {
    'class_id'  : -1,
    'class_name': 'None',
    'conf'      : 0.0,
    'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
    'cx': CAM_W // 2, 'cy': CAM_H // 2,
    'fps'       : 0.0,
    'valid'     : False,
}

# ── Shared Raw Frame Buffer ──────────────────────────────────
frame_lock   = threading.Lock()
shared_frame = None

# ── AI Tracking State for autopilot_thread ───────────────────
ai_class_id    = -1
ai_cx          = CAM_W // 2
ai_frame_count = 0

# ── Web Teleoperation Control State ──────────────────────────
ctrl_lock = threading.Lock()
control_state = {
    "command": None,
    "active" : False,
}

# ============================================================
# 1. Hardware Initialization
# ============================================================
try:
    gpg         = EasyGoPiGo3()
    dist_sensor = gpg.init_distance_sensor()
    servo1      = gpg.init_servo("SERVO1")
    servo2      = gpg.init_servo("SERVO2")
    servo1.rotate_servo(s1_angle)
    servo2.rotate_servo(s2_angle)
    print("✅ Hardware initialized successfully!")
except Exception as e:
    print(f"❌ Hardware connection failed: {e}")
    sys.exit(1)

# ============================================================
# 2. Victory Celebration Sequence (Reusable function)
# ============================================================
def run_celebration():
    """
    Executes celebration sequence in an independent daemon thread.
    All hardware calls acquire hw_lock to ensure thread safety.
    Autopilot and web control commands are suspended during celebration.
    """
    global auto_pilot, s2_angle

    # ── Takeover: Stop all movements ─────────────────────────
    with ctrl_lock:
        control_state["command"] = None
        control_state["active"]  = False
    with state_lock:
        auto_pilot = False
    with hw_lock:
        try:
            gpg.stop()
        except Exception:
            pass

    time.sleep(0.1)

    # ── Synchronized AV Loop x 3 ─────────────────────────────
    for _ in range(3):
        # First half-beat: Green eyes + arm raised + spin left
        with hw_lock:
            try:
                gpg.set_eye_color((0, 255, 0))
                gpg.open_eyes()
                servo2.rotate_servo(150)
                gpg.set_speed(300)
                gpg.left()
            except Exception:
                pass
        time.sleep(0.3)

        # Second half-beat: Blue eyes + arm lowered + spin right
        with hw_lock:
            try:
                gpg.set_eye_color((0, 0, 255))
                gpg.open_eyes()
                servo2.rotate_servo(45)
                gpg.right()
            except Exception:
                pass
        time.sleep(0.3)

    # ── Finale ───────────────────────────────────────────────
    with hw_lock:
        try:
            gpg.stop()
            gpg.set_eye_color((255, 255, 255))
            gpg.open_eyes()
            servo2.rotate_servo(90)
        except Exception:
            pass

    s2_angle = 90
    print("\r🎉 Celebration completed! Restored to standby mode.     ", end="")


# ============================================================
# 3. Flask Web Template
# ============================================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>GUNDAM 5.1 AI</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
        :root {
            --green:  #00ff88; --red: #ff3b3b; --amber: #ffb800;
            --cyan:   #00e5ff; --bg: #050a0e;  --panel: #0a1520;
            --border: #1a3040; --dim: #334455; --magenta: #e020ff;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg); color: var(--green);
            font-family: 'Share Tech Mono', monospace;
            min-height: 100vh; display: flex; flex-direction: column;
            align-items: center; padding: 12px;
            background-image: repeating-linear-gradient(
                0deg, transparent, transparent 2px,
                rgba(0,255,136,0.015) 2px, rgba(0,255,136,0.015) 4px);
        }
        h1 {
            font-family: 'Orbitron', sans-serif; font-weight: 900;
            font-size: clamp(14px, 3vw, 22px); letter-spacing: 6px;
            color: var(--green); text-shadow: 0 0 20px rgba(0,255,136,0.5);
            margin-bottom: 10px;
        }
        #video-wrap { position: relative; border: 1px solid var(--green);
            box-shadow: 0 0 24px rgba(0,255,136,0.2), inset 0 0 40px rgba(0,0,0,0.8); }
        #video { display: block; width: 100%; max-width: 480px; }
        #crosshair {
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 24px; height: 24px;
            border: 1px solid rgba(0,255,136,0.4); border-radius: 50%;
            pointer-events: none;
        }
        #crosshair::before, #crosshair::after {
            content: ''; position: absolute; background: rgba(0,255,136,0.4); }
        #crosshair::before { width:1px; height:10px; left:50%; top:-6px; transform:translateX(-50%); }
        #crosshair::after  { width:10px; height:1px; top:50%; left:-6px; transform:translateY(-50%); }
        .dashboard {
            display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;
            margin: 8px 0; width: 100%; max-width: 480px;
        }
        .stat { background: var(--panel); border: 1px solid var(--border);
            border-radius: 4px; padding: 6px 12px; font-size: 12px;
            text-align: center; flex: 1; min-width: 80px; }
        .stat .label { color: var(--dim); font-size: 10px; display: block; margin-bottom: 2px; }
        .stat .val   { font-family: 'Orbitron', sans-serif; font-size: 14px; color: #fff; }
        .stat .val.warn   { color: var(--red); }
        .stat .val.ok     { color: var(--green); }
        .stat .val.muted  { color: var(--dim); }
        .stat .val.active { color: var(--cyan); }
        .gear-bar { display: flex; gap: 6px; margin: 4px 0; width: 100%; max-width: 480px; }
        .gear-btn {
            flex: 1; background: var(--panel); border: 1px solid var(--border);
            border-radius: 4px; color: var(--dim); font-family: 'Share Tech Mono', monospace;
            font-size: 12px; padding: 6px 4px; cursor: pointer; transition: all 0.15s; text-align: center;
        }
        .gear-btn:hover  { border-color: var(--green); color: var(--green); }
        .gear-btn.active { background: var(--green); color: #000; border-color: var(--green); font-weight: bold; }
        .dpad-wrap {
            display: grid; grid-template-columns: repeat(3, 72px);
            grid-template-rows: repeat(3, 72px); gap: 4px; margin: 10px auto;
        }
        .dpad-btn {
            background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
            color: var(--green); font-size: 24px; display: flex; align-items: center;
            justify-content: center; cursor: pointer; user-select: none;
            -webkit-user-select: none; touch-action: none; transition: background 0.08s, transform 0.08s;
        }
        .dpad-btn:active, .dpad-btn.pressed {
            background: rgba(0,255,136,0.18); border-color: var(--green);
            transform: scale(0.93); box-shadow: 0 0 10px rgba(0,255,136,0.3);
        }
        .dpad-center {
            background: rgba(255,59,59,0.08); border-color: var(--red); color: var(--red);
            font-size: 12px; font-family: 'Orbitron', sans-serif; letter-spacing: 1px;
        }
        .dpad-center:active {
            background: rgba(255,59,59,0.25) !important; border-color: var(--red) !important;
            box-shadow: 0 0 10px rgba(255,59,59,0.4);
        }
        .action-row { display: flex; gap: 8px; margin: 6px 0; width: 100%; max-width: 480px; }
        .action-btn {
            flex: 1; padding: 10px 6px; border-radius: 5px; border: 1px solid var(--border);
            background: var(--panel); color: var(--green); font-family: 'Share Tech Mono', monospace;
            font-size: 12px; cursor: pointer; transition: all 0.15s; text-align: center;
        }
        .action-btn:hover  { box-shadow: 0 0 8px rgba(255,255,255,0.15); }
        .action-btn.danger { border-color: var(--red); color: var(--red); }
        .action-btn.danger:hover { background: rgba(255,59,59,0.15); }
        .action-btn.cyan   { border-color: var(--cyan); color: var(--cyan); }
        .action-btn.cyan:hover { background: rgba(0,229,255,0.1); }
        .action-btn.amber  { border-color: var(--amber); color: var(--amber); }
        .action-btn.amber.on { background: rgba(255,184,0,0.15); box-shadow: 0 0 8px rgba(255,184,0,0.3); }

        /* ── Victory Button Styles ─────────────────────── */
        .action-btn.victory {
            border-color: var(--magenta);
            color: var(--magenta);
            position: relative;
            overflow: hidden;
        }
        .action-btn.victory:hover {
            background: rgba(224,32,255,0.12);
            box-shadow: 0 0 12px rgba(224,32,255,0.4);
        }
        .action-btn.victory.firing {
            animation: victory-flash 0.3s ease-out;
        }
        @keyframes victory-flash {
            0%   { background: rgba(224,32,255,0.5); box-shadow: 0 0 24px rgba(224,32,255,0.8); }
            100% { background: transparent; box-shadow: none; }
        }

        .arm-row { display: flex; align-items: center; gap: 10px; width: 100%; max-width: 480px; margin: 4px 0; }
        .arm-label { font-size: 11px; color: var(--dim); white-space: nowrap; }
        input[type=range] { flex: 1; accent-color: var(--green); height: 4px; cursor: pointer; }
        #arm-val { font-family: 'Orbitron', sans-serif; font-size: 13px; min-width: 40px; text-align: right; }
        #alert-banner {
            display: none; width: 100%; max-width: 480px;
            background: rgba(255,59,59,0.15); border: 1px solid var(--red);
            color: var(--red); text-align: center; padding: 6px;
            font-family: 'Orbitron', sans-serif; font-size: 12px;
            letter-spacing: 2px; margin: 4px 0; animation: blink 0.8s infinite;
        }
        @keyframes blink { 50% { opacity: 0.4; } }
        .hint { font-size: 10px; color: var(--dim); margin-top: 8px; letter-spacing: 1px; text-align: center; }
    </style>
</head>
<body>
<h1>◈ GUNDAM 5.1 AI ◈</h1>
<div id="video-wrap">
    <img id="video" src="/video_feed" alt="feed">
    <div id="crosshair"></div>
</div>
<div id="alert-banner">⚠ PROXIMITY ALERT — MOTORS LOCKED</div>
<div class="dashboard">
    <div class="stat"><span class="label">DIST mm</span><span id="d" class="val ok">--</span></div>
    <div class="stat"><span class="label">SPEED RPM</span><span id="s" class="val">300</span></div>
    <div class="stat"><span class="label">AI TARGET</span><span id="ai_target" class="val muted">None</span></div>
    <div class="stat"><span class="label">AI FPS</span><span id="ai_fps" class="val">--</span></div>
    <div class="stat"><span class="label">AUTOPILOT</span><span id="pilot" class="val muted">OFF</span></div>
    <div class="stat"><span class="label">BRAKE</span><span id="brake" class="val ok">ON</span></div>
</div>
<div class="gear-bar">
    <button class="gear-btn"        id="g1" onclick="setGear('1')">G1 · 100</button>
    <button class="gear-btn active" id="g2" onclick="setGear('2')">G2 · 300</button>
    <button class="gear-btn"        id="g3" onclick="setGear('3')">G3 · 600</button>
    <button class="gear-btn"        id="g4" onclick="setGear('4')">G4 · 800</button>
</div>
<div class="dpad-wrap">
    <div></div>
    <div class="dpad-btn" id="btn-forward"  data-cmd="forward">▲</div>
    <div></div>
    <div class="dpad-btn" id="btn-left"     data-cmd="left">◀</div>
    <div class="dpad-btn dpad-center"       id="btn-stop" onclick="sendStop()">STOP</div>
    <div class="dpad-btn" id="btn-right"    data-cmd="right">▶</div>
    <div></div>
    <div class="dpad-btn" id="btn-backward" data-cmd="backward">▼</div>
    <div></div>
</div>

<!-- Action Buttons Row 1 -->
<div class="action-row">
    <button class="action-btn danger" onclick="fetch('/emergency_stop')">🛑 E-STOP</button>
    <button class="action-btn amber"  id="pilot-btn" onclick="toggleAutopilot()">🤖 AUTOPILOT</button>
    <button class="action-btn cyan"   onclick="fetch('/grab')">🎯 GRAB</button>
    <button class="action-btn"        id="brake-btn" onclick="toggleBrake()">🛡 BRAKE ON</button>
</div>

<!-- Action Buttons Row 2: Victory Celebration -->
<div class="action-row">
    <button class="action-btn victory" id="victory-btn" onclick="triggerVictory()">
        🎉 VICTORY
    </button>
</div>

<div class="arm-row">
    <span class="arm-label">ARM°</span>
    <input type="range" min="0" max="180" value="90" id="arm-slider" oninput="setArm(this.value)">
    <span id="arm-val">90°</span>
</div>
<p class="hint">HOLD W/A/S/D · SPACE=STOP · P=PILOT · G=GRAB · T=BRAKE · M=VICTORY · 1-4=GEAR</p>

<script>
let autoPilotOn = false;
let brakeOn     = true;
let activeCmd   = null;

function startCmd(cmd) {
    if (autoPilotOn) return;
    if (activeCmd === cmd) return;
    activeCmd = cmd;
    document.querySelectorAll('.dpad-btn[data-cmd]').forEach(b =>
        b.classList.toggle('pressed', b.dataset.cmd === cmd));
    fetch('/control/start/' + cmd);
}
function sendStop() {
    if (activeCmd === null) return;
    activeCmd = null;
    document.querySelectorAll('.dpad-btn[data-cmd]').forEach(b => b.classList.remove('pressed'));
    setTimeout(() => { fetch('/control/stop'); }, 100);
}
function setGear(g) {
    fetch('/gear/' + g).then(r => r.json()).then(d => {
        document.querySelectorAll('.gear-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('g' + g).classList.add('active');
        document.getElementById('s').textContent = d.speed;
    });
}
function setArm(v) {
    document.getElementById('arm-val').textContent = v + '°';
    fetch('/servo/arm/' + v);
}
function toggleAutopilot() {
    const url = autoPilotOn ? '/autopilot/off' : '/autopilot/on';
    fetch(url).then(r => r.json()).then(d => { autoPilotOn = d.auto_pilot; updatePilotUI(); });
}
function toggleBrake() {
    fetch('/brake/toggle').then(r => r.json()).then(d => { brakeOn = d.brake_on; updateBrakeUI(); });
}

// ── Trigger Victory Celebration (with visual feedback) ──────
function triggerVictory() {
    const btn = document.getElementById('victory-btn');
    btn.classList.add('firing');
    btn.textContent = '🎉 CELEBRATING...';
    setTimeout(() => { btn.classList.remove('firing'); }, 300);
    fetch('/celebrate').then(() => {
        // Restore button text after 2.2s (3 cycles * 0.6s + margin)
        setTimeout(() => { btn.textContent = '🎉 VICTORY'; }, 2200);
    });
}

function updatePilotUI() {
    const el = document.getElementById('pilot');
    const btn = document.getElementById('pilot-btn');
    el.textContent = autoPilotOn ? 'ON' : 'OFF';
    el.className   = 'val ' + (autoPilotOn ? 'active' : 'muted');
    btn.className  = 'action-btn amber' + (autoPilotOn ? ' on' : '');
    btn.textContent = autoPilotOn ? '🤖 PILOT ON' : '🤖 AUTOPILOT';
}
function updateBrakeUI() {
    const el = document.getElementById('brake');
    const btn = document.getElementById('brake-btn');
    el.textContent = brakeOn ? 'ON' : 'OFF';
    el.className   = 'val ' + (brakeOn ? 'ok' : 'warn');
    btn.textContent = '🛡 BRAKE ' + (brakeOn ? 'ON' : 'OFF');
}

const KEY_MAP = {
    'w':'forward','s':'backward','a':'left','d':'right',
    'ArrowUp':'forward','ArrowDown':'backward','ArrowLeft':'left','ArrowRight':'right',
};
document.addEventListener('keydown', function(e) {
    if (e.repeat) return;
    const cmd = KEY_MAP[e.key];
    if (cmd) { e.preventDefault(); startCmd(cmd); return; }
    switch (e.key.toLowerCase()) {
        case ' ':        e.preventDefault(); sendStop(); break;
        case 'p':        toggleAutopilot(); break;
        case 'g':        fetch('/grab'); break;
        case 't':        toggleBrake(); break;
        case 'm':        triggerVictory(); break;          // ← Victory hotkey
        case '1': case '2': case '3': case '4': setGear(e.key); break;
    }
});
document.addEventListener('keyup', function(e) {
    const cmd = KEY_MAP[e.key];
    if (cmd && activeCmd === cmd) sendStop();
});

document.querySelectorAll('.dpad-btn[data-cmd]').forEach(btn => {
    const cmd = btn.dataset.cmd;
    btn.addEventListener('mousedown',   () => startCmd(cmd));
    btn.addEventListener('mouseup',     sendStop);
    btn.addEventListener('mouseleave',  () => { if (activeCmd === cmd) sendStop(); });
    btn.addEventListener('touchstart',  (e) => { e.preventDefault(); startCmd(cmd); }, { passive: false });
    btn.addEventListener('touchend',    (e) => { e.preventDefault(); sendStop(); },    { passive: false });
    btn.addEventListener('touchcancel', (e) => { e.preventDefault(); sendStop(); },    { passive: false });
});

setInterval(function() {
    fetch('/status').then(r => r.json()).then(data => {
        const dEl = document.getElementById('d');
        dEl.textContent = data.dist;
        dEl.className   = 'val ' + (data.alert ? 'warn' : 'ok');
        document.getElementById('alert-banner').style.display = data.alert ? 'block' : 'none';
        document.getElementById('ai_fps').textContent = data.ai_fps;
        const tEl = document.getElementById('ai_target');
        tEl.textContent = data.ai_target;
        tEl.className   = 'val ' + (data.ai_target === 'None' ? 'muted' : 'active');
        if (autoPilotOn !== data.auto_pilot) { autoPilotOn = data.auto_pilot; updatePilotUI(); }
        if (brakeOn !== data.brake_on)       { brakeOn = data.brake_on;       updateBrakeUI(); }
    });
}, 600);
</script>
</body>
</html>
"""

# ============================================================
# 4. Flask Routes
# ============================================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    with ai_lock:
        target = ai_result['class_name']
        ai_fps = round(ai_result['fps'], 1)
    with state_lock:
        d     = global_dist
        spd   = current_speed
        gear  = current_gear
        brake = auto_brake_enabled
        pilot = auto_pilot
        arm   = s2_angle
    return jsonify(
        dist=d, speed=spd, angle=arm, gear=gear,
        brake_on=brake, auto_pilot=pilot,
        ai_target=target, ai_fps=ai_fps,
        alert=(brake and isinstance(d, (int, float)) and 0 < d < SAFE_STOP_DIST_MM),
    )

@app.route('/control/start/<cmd>')
def control_start(cmd):
    if cmd not in {"forward", "backward", "left", "right"}:
        return jsonify(ok=False, error="invalid command"), 400
    with ctrl_lock:
        control_state["command"] = cmd
        control_state["active"]  = True
    return jsonify(ok=True, command=cmd)

@app.route('/control/stop')
def control_stop():
    with ctrl_lock:
        control_state["command"] = None
        control_state["active"]  = False
    with hw_lock:
        try: gpg.stop()
        except Exception: pass
    return jsonify(ok=True)

@app.route('/emergency_stop')
def emergency_stop():
    global auto_pilot
    with ctrl_lock:
        control_state["command"] = None
        control_state["active"]  = False
    with state_lock:
        auto_pilot = False
    with hw_lock:
        try: gpg.stop()
        except Exception: pass
    return jsonify(ok=True, message="Emergency stop executed")

@app.route('/autopilot/on')
def autopilot_on():
    global auto_pilot
    with ctrl_lock:
        control_state["command"] = None
        control_state["active"]  = False
    with hw_lock:
        try: gpg.stop()
        except Exception: pass
    with state_lock:
        auto_pilot = True
    return jsonify(ok=True, auto_pilot=True)

@app.route('/autopilot/off')
def autopilot_off():
    global auto_pilot
    with state_lock:
        auto_pilot = False
    with hw_lock:
        try: gpg.stop()
        except Exception: pass
    return jsonify(ok=True, auto_pilot=False)

@app.route('/gear/<g>')
def set_gear(g):
    global current_speed, current_gear
    if g not in SPEED_GEARS:
        return jsonify(ok=False, error="invalid gear"), 400
    with state_lock:
        current_gear  = g
        current_speed = SPEED_GEARS[g]
    return jsonify(ok=True, gear=g, speed=current_speed)

@app.route('/servo/arm/<int:angle>')
def set_arm(angle):
    global s2_angle
    angle    = max(0, min(180, angle))
    s2_angle = angle
    with hw_lock:
        try: servo2.rotate_servo(angle)
        except Exception: pass
    return jsonify(ok=True, angle=angle)

@app.route('/brake/toggle')
def brake_toggle():
    global auto_brake_enabled
    with state_lock:
        auto_brake_enabled = not auto_brake_enabled
        s = auto_brake_enabled
    return jsonify(ok=True, brake_on=s)

@app.route('/grab')
def grab():
    def _grab():
        with hw_lock:
            try:
                gpg.stop()
                gpg.set_speed(150)
                gpg.drive_cm(6)
                time.sleep(0.5)
                servo2.rotate_servo(CARRY_ANGLE)
            except Exception: pass
    threading.Thread(target=_grab, daemon=True).start()
    return jsonify(ok=True, action="grab")

# ── Victory Celebration Route ────────────────────────────────
@app.route('/celebrate')
def celebrate():
    """
    Trigger victory celebration macro.
    Returns 200 immediately; celebration runs asynchronously in a daemon thread.
    """
    threading.Thread(target=run_celebration, daemon=True, name="Celebrate").start()
    return jsonify(ok=True, action="celebrate")

# ============================================================
# 5. Camera Thread
# ============================================================
def camera_thread():
    global shared_frame
    try:
        with picamera.PiCamera() as cam:
            cam.resolution = (CAM_W, CAM_H)
            cam.framerate  = CAM_FPS
            stream = io.BytesIO()
            for _ in cam.capture_continuous(stream, 'jpeg', use_video_port=True):
                if stop_event.is_set(): break
                stream.seek(0)
                data = stream.read()
                with frame_lock:
                    shared_frame = data
                stream.seek(0)
                stream.truncate()
    except Exception as e:
        print(f"❌ Camera error: {e}")

# ============================================================
# 6. Video Streaming — YOLO Style Persistent Overlays
# ============================================================
@app.route('/video_feed')
def video_feed():
    import cv2

    def generate_frames():
        while not stop_event.is_set():
            with frame_lock:
                raw = shared_frame
            if raw is None:
                time.sleep(0.04)
                continue
            try:
                nparr = np.frombuffer(raw, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    time.sleep(0.04)
                    continue
            except Exception:
                time.sleep(0.04)
                continue

            with ai_lock:
                r = ai_result.copy()

            if r['valid']:
                x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']
                cx, cy          = r['cx'], r['cy']
                raw_name        = r['class_name'].lower().replace(' ', '_')
                label           = f"{raw_name} {r['conf']:.2f}"
                box_color       = (255, 0, 0)
            else:
                cx, cy    = CAM_W // 2, CAM_H // 2
                half      = 60
                x1, y1    = cx - half, cy - half
                x2, y2    = cx + half, cy + half
                label     = "no_target 0.00"
                box_color = (120, 120, 120)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            thickness  = 1
            (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            lx1, ly1 = x1,      y1 - lh - baseline - 4
            lx2, ly2 = x1 + lw, y1
            if ly1 < 0:
                ly1 = y2
                ly2 = y2 + lh + baseline + 4
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), box_color, -1)
            cv2.putText(frame, label, (lx1, ly2 - baseline - 1),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.line(frame, (CAM_W // 2, 0), (CAM_W // 2, CAM_H), (80, 80, 80), 1)

            try:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
            except Exception:
                pass

            time.sleep(0.04)

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================
# 7. AI Inference Thread
# ============================================================
def ai_thread():
    global ai_class_id, ai_cx, ai_frame_count
    import cv2

    try:
        interp  = Interpreter(model_path=MODEL_PATH, num_threads=4)
        interp.allocate_tensors()
        inp_det = interp.get_input_details()
        out_det = interp.get_output_details()
        print(f"✅ TFLite loaded successfully | Input: {inp_det[0]['shape']}")
    except Exception as e:
        print(f"❌ Failed to load TFLite model: {e}")
        return

    scale_x = CAM_W / INPUT_SIZE
    scale_y = CAM_H / INPUT_SIZE
    t_last  = time.time()

    while not stop_event.is_set():
        with frame_lock:
            raw = shared_frame
        if raw is None:
            time.sleep(0.1)
            continue
        try:
            nparr      = np.frombuffer(raw, np.uint8)
            img_bgr    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                time.sleep(0.1)
                continue
            img_640    = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
            img_rgb    = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
            inp_tensor = img_rgb.astype(np.float32)[np.newaxis] / 255.0
        except Exception:
            time.sleep(0.1)
            continue
        try:
            interp.set_tensor(inp_det[0]['index'], inp_tensor)
            interp.invoke()
            raw_out = interp.get_tensor(out_det[0]['index'])
            preds   = raw_out[0].T
        except Exception as e:
            print(f"Inference error: {e}")
            time.sleep(0.1)
            continue

        cls_confs = preds[:, 4:7]
        row_maxes = cls_confs.max(axis=1)
        top_idx   = int(row_maxes.argmax())
        top_conf  = float(row_maxes[top_idx])

        now    = time.time()
        fps    = 1.0 / max(now - t_last, 1e-6)
        t_last = now

        if top_conf > CONF_THRESHOLD:
            best_cls = int(cls_confs[top_idx].argmax())
            box      = preds[top_idx, :4].copy()
            xc_cam = box[0] * scale_x
            yc_cam = box[1] * scale_y
            bw_cam = box[2] * scale_x
            bh_cam = box[3] * scale_y
            x1 = max(0,         int(xc_cam - bw_cam / 2))
            y1 = max(0,         int(yc_cam - bh_cam / 2))
            x2 = min(CAM_W - 1, int(xc_cam + bw_cam / 2))
            y2 = min(CAM_H - 1, int(yc_cam + bh_cam / 2))
            with ai_lock:
                ai_result.update({
                    'class_id': best_cls,
                    'class_name': CLASS_NAMES.get(best_cls, '?'),
                    'conf': top_conf,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': int(xc_cam), 'cy': int(yc_cam),
                    'fps': fps, 'valid': True,
                })
            ai_class_id    = best_cls
            ai_cx          = int(xc_cam)
            ai_frame_count += 1
        else:
            with ai_lock:
                ai_result.update({
                    'class_id': -1, 'class_name': 'None',
                    'conf': 0.0, 'fps': fps, 'valid': False,
                })
            ai_class_id    = -1
            ai_cx          = CAM_W // 2
            ai_frame_count += 1

# ============================================================
# 8. Ultrasonic Sensor Thread
# ============================================================
def sensor_thread():
    global global_dist
    while not stop_event.is_set():
        try:
            with hw_lock:
                dist = dist_sensor.read_mm()
            if dist <= 0:
                time.sleep(0.12)
                continue
            with state_lock:
                global_dist = dist
                pilot_on    = auto_pilot
                brake_on    = auto_brake_enabled
            if brake_on and not pilot_on and dist < SAFE_STOP_DIST_MM:
                with hw_lock:
                    gpg.stop()
                with ctrl_lock:
                    control_state["command"] = None
                    control_state["active"]  = False
        except Exception:
            with state_lock:
                global_dist = 9999
        time.sleep(0.12)

# ============================================================
# 9. Web Control Loop Thread
# ============================================================
def control_loop():
    last_cmd = None
    while not stop_event.is_set():
        with ctrl_lock:
            cmd    = control_state["command"]
            active = control_state["active"]
        with state_lock:
            dist     = global_dist
            brake_on = auto_brake_enabled
            pilot_on = auto_pilot

        if pilot_on:
            last_cmd = None
            time.sleep(0.02)
            continue

        if not active or cmd is None:
            if last_cmd is not None:
                with hw_lock:
                    try: gpg.stop()
                    except Exception: pass
                last_cmd = None
            time.sleep(0.02)
            continue

        if cmd == "forward" and brake_on and 0 < dist < SAFE_STOP_DIST_MM:
            with hw_lock:
                try: gpg.stop()
                except Exception: pass
            last_cmd = None
            time.sleep(0.02)
            continue

        if cmd != last_cmd:
            with hw_lock:
                try:
                    gpg.set_speed(current_speed)
                    if   cmd == "forward":  gpg.forward()
                    elif cmd == "backward": gpg.backward()
                    elif cmd == "left":     gpg.left()
                    elif cmd == "right":    gpg.right()
                except Exception: pass
            last_cmd = cmd

        time.sleep(0.02)

# ============================================================
# 10. Autopilot Thread
# ============================================================
def autopilot_thread():
    global auto_pilot

    NUDGE_TIME   = 0.08
    STEP_TIME    = 0.20
    ORBIT_TIME   = 0.25
    TURN_SPEED   = 90
    FWD_SPEED    = 200
    ORBIT_SPEED  = 150
    WAIT_TIMEOUT = 3.0

    def fuse_check():
        global auto_pilot
        with state_lock:
            d = global_dist
        if 0 < d < AUTOPILOT_STOP_MM:
            with hw_lock:
                try: gpg.stop()
                except Exception: pass
            with state_lock:
                auto_pilot = False
            print("\n🛑 Reached grab range (12cm). Autopilot disengaged, please take manual control!")
            return True
        return False

    def wait_for_new_frame(last_count):
        deadline = time.time() + WAIT_TIMEOUT
        while time.time() < deadline:
            if stop_event.is_set(): break
            if ai_frame_count != last_count:
                return ai_frame_count
            time.sleep(0.02)
        return ai_frame_count

    while not stop_event.is_set():
        with state_lock:
            pilot_on = auto_pilot
        if not pilot_on:
            time.sleep(0.1)
            continue

        if fuse_check(): continue

        current_count = ai_frame_count
        current_count = wait_for_new_frame(current_count)
        if fuse_check(): continue

        cls = ai_class_id
        cx  = ai_cx

        if cls == -1:
            with hw_lock:
                try: gpg.set_speed(TURN_SPEED); gpg.right()
                except Exception: pass
            time.sleep(NUDGE_TIME * 1.5)
            with hw_lock:
                try: gpg.stop()
                except Exception: pass

        elif cls in (1, 2):
            with hw_lock:
                try:
                    gpg.set_speed(ORBIT_SPEED)
                    gpg.steer(40, 100) if cx < (CAM_W // 2) else gpg.steer(100, 40)
                except Exception: pass
            time.sleep(ORBIT_TIME)
            with hw_lock:
                try: gpg.stop()
                except Exception: pass

        elif cls == 0:
            error = cx - (CAM_W // 2)
            if abs(error) > CENTER_TOLERANCE:
                with hw_lock:
                    try:
                        gpg.set_speed(TURN_SPEED)
                        gpg.left() if error < 0 else gpg.right()
                    except Exception: pass
                time.sleep(NUDGE_TIME)
                with hw_lock:
                    try: gpg.stop()
                    except Exception: pass
            else:
                with hw_lock:
                    try: gpg.set_speed(FWD_SPEED); gpg.forward()
                    except Exception: pass
                time.sleep(STEP_TIME)
                with hw_lock:
                    try: gpg.stop()
                    except Exception: pass

        time.sleep(0.1)

# ============================================================
# 11. Main Entry Point
# ============================================================
if __name__ == '__main__':
    # os.system("kill -9 $(pgrep -u jupyter python) 2>/dev/null")
    time.sleep(0.5)

    threads = [
        threading.Thread(target=camera_thread,    daemon=True, name="Camera"),
        threading.Thread(target=sensor_thread,    daemon=True, name="Sensor"),
        threading.Thread(target=ai_thread,        daemon=True, name="AI"),
        threading.Thread(target=autopilot_thread, daemon=True, name="AutoPilot"),
        threading.Thread(target=control_loop,     daemon=True, name="ControlLoop"),
        threading.Thread(
            target=lambda: app.run(
                host='0.0.0.0', port=5003,
                debug=False, use_reloader=False, threaded=True
            ),
            daemon=True, name="Flask"
        ),
    ]
    for t in threads:
        t.start()

    print("🚀 GUNDAM 5.1 AI Edition Started!")
    print("👉 Access web UI via: http://10.10.10.10:5003")
    print("⚡ Full Web Control Mode — No terminal keyboard dependency")
    print("🎉 'M' Key / Web Button = Victory Celebration Macro")
    print("🛑 Press Ctrl+C to safely terminate")

    try:
        stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            gpg.stop()
            gpg.reset_all()
        except Exception:
            pass
        print("\n✅ System shutdown safely!")
        os._exit(0)
