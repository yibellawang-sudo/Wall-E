# Wall-E Backend + Trash Intelligence Dashboard

This repository contains the cloud AI backend and real-time visualization dashboard for **Wall-E**, an autonomous waste-detection platform designed to help cities move from *reactive cleanup* to *proactive waste management*.
**This version of the system does not connect to physical hardware — instead, it includes a simulation script that mimics robot motion and trash detection, generating fictional data for testing and demonstration.**

---

## Core Features

### AI Backend (Flask, Python)

* Receives (simulated) image + metadata streams from a virtual robot
* Calls **Gemini Vision API** for detailed trash classification
* Extracts materials, item type, contamination level & disposal category
* Stores results with GPS + timestamps for temporal and spatial mapping
* Batch analysis every 5 minutes to produce AI-generated insights

### Live Dashboard

* Interactive map built with **Leaflet.js**
* Heatmap + marker view toggle for trash hotspots
* Live detection feed populated using simulated data
* Click-to-navigate to cleanup position
* Auto-refresh for real-time visualization
* AI recommendations panel summarizing patterns + actions

---

## Tech Stack

| Component         | Tech                                             |
| ----------------- | ------------------------------------------------ |
| AI Classification | Gemini Vision                                    |
| API Backend       | Python • Flask • Flask-CORS                      |
| CV Integration    | Simulated YOLOv8/OpenCV pipeline via JSON stream |
| Storage           | Local JSON store + image directory structure     |
| Frontend          | HTML • CSS • JavaScript                          |
| Mapping           | Leaflet.js • Leaflet.heat                        |

---

## Data Flow (Simulation-Based)

1. Simulated robot script generates mock detections + movement
2. Frames + metadata are POSTed to this backend
3. Backend forwards images to **Gemini Vision** for analysis
4. Classified objects are saved with GPS + timestamp
5. Dashboard polls backend + displays live results
6. AI insights generated periodically from synthetic dataset

---

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start backend

```bash
python aiBackend.py
```
```bash
python roboSim.py
```

### Launch dashboard

Open `webapp.html` in a browser
(Data will stream in automatically from the simulator.)

---

## Next Steps

* Replace simulated robot with real hardware integration
* Transition JSON storage → scalable cloud database
* Add volunteer cleanup features and hotspot forecasting

---

## License

Open source for experimentation and environmental research. Attribution appreciated.


