## SIYI Camera YOLO Inference with Controls and Tracker 📸
Python implementation of YOLO Inference and Basic Camera Control with ZR10 and A8 Mini cameras. 

* Camera webpage: http://en.siyi.biz/en/Gimbal%20Camera/ZR10/overview/
* Documentation: http://en.siyi.biz/en/Gimbal%20Camera/ZR10/download/

Repo based on this repo: https://github.com/mzahana/siyi_sdk.git

And this repo: https://github.com/Innopolis-UAV-Team/siyi-python-sdk

## Setup
* Git clone this repo
* Connect the camera to PC or onboard computer using the ethernet cable that comes with it. The current implementation uses UDP communication.
* Power on the camera
* Do the PC wired network configuration. Make sure to assign a manual IP address to your computer
  * For example, IP `192.168.144.12`
  * Gateway `192.168.144.25`
  * Netmask `255.255.255.0`
* Done.
## Usage 

To use:
- cd into repo
- Open `mlobject_tracker.py`. Go to line 42. Change the path to YOLO model weight.
- Run:
```python
python3 mlobject_tracker.py
```
## Controls⌨️

To control using the camera here's the simple guidelines:

**WASD** - Control yawing and pitching 

**Q** - Zoom Out

**E** - Zoom In

**N** - Center the camera

**1,2,3** = Switch between FPV / Lock / Follow modes

**P** - take a photo from camera

**Esc** - Stop the inference
