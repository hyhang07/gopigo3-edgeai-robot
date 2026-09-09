# 🤖 GUNDAM 5.1: Edge AI Autonomous & Teleoperated Robotics Platform

[![Platform: Raspberry Pi 4](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red.svg?logo=raspberry-pi)](https://www.raspberrypi.com/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask-000000.svg?logo=flask)](https://flask.palletsprojects.com/)
[![Vision: YOLOv8 TFLite](https://img.shields.io/badge/Vision-YOLOv8%20FP16%20TFLite-blue.svg?logo=tensorflow)](https://tensorflow.org/)
[![Computer Vision: OpenCV](https://img.shields.io/badge/CV-OpenCV-green.svg?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GUNDAM 5.1** is a high-responsiveness, multithreaded edge robotics platform built on a GoPiGo3 chassis powered by a Raspberry Pi 4. It integrates an on-device quantized YOLOv8 object detection pipeline, zero-latency MJPEG video streaming with a cyberpunk Head-Up Display (HUD), asynchronous reactive safety mechanisms, and a multi-agent AI collaborative architecture.

---

## 📌 Key Highlights

- **Custom-Trained Edge YOLOv8 (TFLite FP16)**: Trained on 1,500+ annotated samples across multiple target orientations (`front_ring`, `diagonal_ring`, `side_ring`), running locally on Raspberry Pi 4 without cloud dependency.
- **6-Thread Decoupled Concurrency**: Strict hardware mutex isolation (`hw_lock`) preventing I2C communication bus starvation and race conditions between chassis motors, distance sensors, and dual servos.
- **Cyberpunk Military Web HUD**: Zero external JS framework dependency. Features real-time telemetry (distance, speed RPM, target class, AI FPS), live crosshair, and keyboard shortcuts.
- **Proactive Virtual Bumper (Autobrake)**: High-frequency ultrasonic distance daemon automatically enforces emergency motor lock if obstacles approach closer than 150mm.
- **Automated Macro Execution**: Single-key asynchronous grab routine (150 RPM approach + Z-axis intake flick) and a synchronized audio-visual victory celebration sequence.
- **Human-AI Collaborative Engineering**: Architected and tuned through an iterative multi-agent relay (Gemini $\to$ Claude $\to$ Hardware Validation).

---

## 🏛️ System Architecture

The robot decouples UI serving, sensor acquisition, inference, motion control, and high-level autonomous state machines across **6 concurrent daemon threads**:

```text
                  +----------------------------------------------+
                  |           Flask Web Server (Main)            |
                  |     (Web UI / Status JSON / REST APIs)       |
                  +----------------------+-----------------------+
                                         |
     +-------------------+---------------+-------------------+
     |                   |                                   |
[camera_thread]    [sensor_thread]                      [control_loop]
  Captures raw       Polls ultrasonic (120ms)             Executes 50Hz non-blocking
  JPEG to buffer     Virtual Bumper (<150mm)              I2C motor commands
     |                   |                                   |
     v                   v                                   v
+------------+     +------------+                      +------------+
| frame_lock |     | state_lock |                      | ctrl_lock  |
+------------+     +------------+                      +------------+
     |                   |                                   |
     +---------+---------+                                   |
               |                                             |
               v                                             v
         [ai_thread]                                [autopilot_thread]
         Runs YOLOv8 TFLite FP16                     FSM State Machine
         Writes detection & coordinates              (Orbit, Center, Forward)
               |                                             |
               +----------------------+----------------------+
                                      |
                                      v
                             +-----------------+
                             |     hw_lock     |  <-- Strict Mutex Isolation
                             +--------+--------+
                                      |
                       +--------------+--------------+
                       |                             |
                 GoPiGo3 Base                Dual Servos & LEDs
                 (I2C Motors)                (Robotic Arm & Eyes)
```

### Concurrency & Thread-Safety Matrix

| Thread | Rate / Timeout | Managed Lock | Responsibility |
|---|---|---|---|
| `camera_thread` | ~24 FPS continuous | `frame_lock` | Captures JPEG frames directly from PiCamera into shared buffer. |
| `sensor_thread` | 120 ms interval | `state_lock`, `hw_lock` | Continuous ultrasonic distance polling & reactive brake trigger. |
| `ai_thread` | Continuous | `ai_lock`, `frame_lock` | Preprocessing, 640x640 letterboxing, and TFLite tensor inference. |
| `control_loop` | 50 Hz (20 ms) | `ctrl_lock`, `hw_lock` | State-driven motion execution. Dispatches motor writes only on state delta. |
| `autopilot_thread`| Event-driven | `state_lock`, `hw_lock` | FSM alignment: Target search $\to$ Orbit $\to$ Centering $\to$ Forward drive. |
| `Flask thread` | Asynchronous | *Multiple* | Serves telemetry endpoints, MJPEG video generator, and Web UI. |

---

## 🎮 Web Teleoperation Interface (HUD)

The browser client is crafted with an embedded HUD aesthetic using Google Fonts (*Orbitron* and *Share Tech Mono*) and dynamic scanline styling:

```text
+-----------------------------------------------------------+
|                    ◈ GUNDAM 5.1 AI ◈                      |
| +-------------------------------------------------------+ |
| |                  [ Live Camera Feed ]                 | |
| |                         +--+                          | |
| |                         |  | front_ring 0.91          | |
| |                        --+--                          | |
| +-------------------------------------------------------+ |
|  [DIST: 525mm] [SPEED: 300] [TARGET: front_ring] [FPS: 0.8]  |
|  [G1: 100]       [G2: 300]       [G3: 600]      [G4: 800]  |
|                    [ ▲ ]                                  |
|            [ ◀ ]  [STOP]  [ ▶ ]                            |
|                    [ ▼ ]                                  |
|  [🛑 E-STOP]   [🤖 AUTOPILOT]   [🎯 GRAB]   [🛡 BRAKE ON]  |
|  [                     🎉 VICTORY                       ]  |
|  ARM Z-AXIS SLIDER: [=======O===========] 90°             |
+-----------------------------------------------------------+
```

### Teleoperation Keybindings

| Key | Action | Functionality |
|---|---|---|
| `W` / `▲` | Forward | Continuous drive forward |
| `S` / `▼` | Backward | Continuous reverse |
| `A` / `◀` | Spin Left | Counter-clockwise pivot |
| `D` / `▶` | Spin Right | Clockwise pivot |
| `Space` | E-STOP | Immediate hard stop and control state wipe |
| `1` - `4` | Gear Shift | Switch wheel velocity: 100, 300, 600, 800 RPM |
| `G` | Grab Macro | Approach 6cm + lift arm to 130° intake angle |
| `T` | Toggle Brake | Enable / disable 150mm ultrasonic virtual bumper |
| `P` | Autopilot | Toggle autonomous target tracking state machine |
| `M` | Victory | Multi-stage LED blink + chassis pivot celebration |
| `Slider` | Arm Height | Direct Servo 2 angular elevation control (0°–180°) |

---

## 🦾 Mechanical Design & Hardware Integration

- **Chassis**: GoPiGo3 Mobile Robotic Platform.
- **Compute**: Raspberry Pi 4 Model B.
- **Vision**: Raspberry Pi Camera Module v2.
- **Proximity**: Forward-facing ultrasonic distance ranger.
- **Custom Z-Axis Arm**: Fabricated with an internal tension-twist core encased in an aligned sleeve, secured with elastic counter-tension dampeners to eliminate mechanical backlash during high-speed macro lifts.

---

## 🧠 Edge Computer Vision Pipeline

The system utilizes a custom-trained **YOLOv8** model quantized to **FP16 TFLite** to classify target retrieval items by pose:
- `front_ring`: Normal attack vector; triggers centered forward advance.
- `diagonal_ring`: Angled vector; triggers coordinated differential arc steer.
- `side_ring`: Orthogonal vector; triggers orbital repositioning.

```bash
# Model Export Routine
yolo export model=runs/detect/train/weights/best.pt format=tflite imgsz=640 half=True
```

---

## 🔄 Human-AI Collaborative Workflow

```text
[ PHASE 1: Conceptualization ]
   (Human) Analyze operational criteria & mechanical constraints
      |
      v
   (Gemini) System feasibility analysis & architecture planning
      |
[ PHASE 2: Hardware & Sensor Profiling ]
   (Human) Hardware testing on Pi 4 -> camera blur & I2C latency
      |
      v
   (Gemini) V4L2 optimization & focal plane calibration
      |
[ PHASE 3: Custom Vision Model Pipeline ]
   (Human) GoPiGo camera collection & LabelImg annotation (1500+ labels)
      |
      v
   (Gemini) Data split & class balancing routines
      |
      v
   (Human) Local GPU training (RTX 4060) -> FP16 TFLite conversion
      |
[ PHASE 4: Concurrency Refactoring (Multi-Agent Relay) ]
   (Human) Monolithic script facing I2C bus locks & video drops
      |
      v
   (Gemini) Concurrency model definition & strict lock hierarchy prompt
      |
      v
   (Claude) Generates 6-thread decoupled architecture framework
      |
[ PHASE 5: Deployment & Tuning ]
   (Human) Physical deployment on robot & latency tuning
      |
      v
   (Gemini) Edge tensor dimension fixes & hardware mutex scope tuning
      |
      v
   (Human) Production run & automated mission completion
```

---

## 🚀 Getting Started

### 1. Hardware Requirements
- Raspberry Pi 4 (2GB+ RAM recommended)
- GoPiGo3 Robot Base Kit with battery pack
- PiCamera Module
- Dexter Ultrasonic Sensor & Micro Servo

### 2. Software Setup
Clone the repository and install dependencies on your Raspberry Pi:

```bash
git clone https://github.com/hyhang07/gundam-edgeai-teleop.git
cd gundam-edgeai-teleop

# Install Python dependencies
pip3 install -r requirements.txt
```

### 3. Execution
Ensure your model file (`best_float16.tflite`) is placed in the project root:

```bash
python3 src/app.py
```

Once running, access the HUD from any device on the same local network:
```text
http://<YOUR_PI_IP_ADDRESS>:5003
```

---

## 👥 Contributors

- **Hong Yee Hang** ([@hyhang07](https://github.com/hyhang07)) - Concurrency Architecture, Control Systems & Deployment
- **Tan Zong Ting** - Edge AI Training (YOLOv8), Quantization & Vision Pipeline
- **Louis Cha Hao Le** - Mechanical Fabrication, Hardware Integration & Web HUD

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
