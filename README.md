<div align="center">

# 😴 Sleep Alarm Detector

### 🚗 AI-Powered Real-Time Drowsiness Detection System

Detect fatigue before it becomes dangerous using **Computer Vision**, **MediaPipe Face Mesh**, and **OpenCV**.

<p>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-FF6F00?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Working-success?style=for-the-badge)

</p>

---

### 🎥 Demo

▶️ **[Watch the Demo Video](assets/output.mp4)**

---

</div>

# 🌟 Overview

Sleep Alarm Detector is an AI-based computer vision application that continuously monitors the user's eyes through a webcam. By analyzing facial landmarks in real time, it calculates the **Eye Aspect Ratio (EAR)** to determine whether the eyes are open or closed.

If the eyes remain closed for more than **2.5 seconds**, the system instantly triggers an alarm, helping prevent microsleep and drowsiness.

This project demonstrates the practical application of **Artificial Intelligence**, **Computer Vision**, and **Human Safety Systems**.

---

# ✨ Features

✅ Real-time Face Detection

✅ Eye Aspect Ratio (EAR) Calculation

✅ Yawn Detection (Mouth Aspect Ratio)

✅ Head-Droop / Head-Pose Detection

✅ Blink-Rate Tracking

✅ Composite Drowsiness Score (0–100)

✅ Personalised Startup Calibration

✅ Voice Alerts ("Wake up!") via Text-to-Speech

✅ Escalating Alarm Volume

✅ Snooze by Key or Head-Shake Gesture

✅ Live Status Overlay & Face Bounding Box

✅ Lightweight & Fast

✅ Cross Platform

---

# 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| 🐍 Python | Core Programming |
| 👁️ OpenCV | Webcam, Image Processing & Head-Pose (solvePnP) |
| 🤖 MediaPipe Face Mesh | Facial Landmark Detection |
| 🔢 NumPy | Mathematical Calculations |
| 🔊 Pygame | Alarm System |
| 🗣️ pyttsx3 | Offline Voice Alerts |

---

# ⚙️ Working Flow

```text
 Webcam
    │
    ▼
 Capture Video Frames
    │
    ▼
 Detect Face using MediaPipe
    │
    ▼
 Extract Eye Landmarks
    │
    ▼
 Calculate Eye Aspect Ratio (EAR)
    │
    ▼
 Eyes Closed?
   │        │
   │ No     │ Yes
   ▼        ▼
 Continue   Start Timer
                 │
                 ▼
      Closed > 2.5 Seconds?
             │
             ▼
        🚨 Trigger Alarm
```

---

# 📂 Project Structure

```
Sleep-Alarm-Detector
│
├── sleep_alarm.py
├── requirements.txt
├── dragon-studio-censor-beep-3-372460.mp3
└── README.md
```

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Sleep-Alarm-Detector.git
```

```bash
cd Sleep-Alarm-Detector
```

### Create Virtual Environment

```bash
python3 -m venv venv
```

### Activate

macOS / Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python sleep_alarm.py
```

---

# 🧠 How It Works

The system uses **MediaPipe Face Mesh**, which detects **468 facial landmarks** in every frame captured by the webcam.

Using six landmark points around each eye, the application computes the **Eye Aspect Ratio (EAR)**.

- 👁️ High EAR → Eyes Open
- 😴 Low EAR → Eyes Closed

Beyond the eyes, the app also tracks the **Mouth Aspect Ratio (MAR)** for yawns,
estimates **head pose** with `cv2.solvePnP` to catch a drooping/nodding head, and
monitors **blink rate**. These signals are blended into a single **drowsiness
score (0–100)**.

The alarm triggers if the eyes stay closed past the threshold, the head droops
for too long, **or** the overall score crosses **85** — then it escalates in
volume and speaks aloud until you respond or snooze.

---

# 📊 Detection Parameters

| Parameter | Value |
|-----------|------:|
| EAR Threshold | **0.22** (auto-calibrated at startup) |
| Eye Closure Duration | **2.5 Seconds** |
| MAR (Yawn) Threshold | **0.60** |
| Yawn Minimum Duration | **0.6 Seconds** |
| Head-Droop Angle | **18°** |
| Head-Droop Duration | **2.0 Seconds** |
| Score Alarm Threshold | **85 / 100** |
| Snooze Duration | **20 Seconds** |
| Maximum Faces | **1** |

The alarm fires on **any** of: eyes closed too long, head drooped too long, or a
composite drowsiness score at/above the threshold.

---

# 🎮 Controls

| Key / Gesture | Action |
|---------------|--------|
| **Q** | Quit |
| **S** | Snooze the alarm |
| 🙆 **Shake your head** | Snooze the alarm hands-free |

At launch the app runs a short calibration — **keep your eyes open** for a few
seconds so it can personalise the closed-eye threshold (press **S** to skip).

---

# 💡 Applications

🚗 Driver Drowsiness Detection

💻 Long Working Hours Monitoring

📚 Student Study Assistant

🏭 Industrial Safety

👨‍💻 Programmer Fatigue Monitoring

---

# 🔮 Future Scope

- 📱 Mobile Notifications
- ☁️ Cloud Dashboard
- 📈 Sleep Analytics & Session Reports
- 🧠 Deep Learning Based Drowsiness Detection
- 🌙 Night Vision Support
- 👥 Multi-Person Detection

---

# 📦 Dependencies

```
mediapipe
opencv-python
numpy
pygame
pyttsx3
```

> `pyttsx3` is optional — without it the app still runs, just without voice alerts.

---

# 👩‍💻 Author

**Shravani Rane**

Computer Science Engineering (AI & ML)

🌐 GitHub: https://github.com/shravaniranee

---

<div align="center">

### ⭐ If you found this project interesting, consider giving it a Star!

**Made with ❤️ using Python, OpenCV & MediaPipe**

</div>
