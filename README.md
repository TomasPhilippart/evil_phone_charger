# Evil Phone Charging Station
Codebase for Research Project 2 - Building an Evil Phone Charging Station. 
Author: Tomás Philippart

Attack and threat model/setup:

![Attack model](docs/AttackAndThreatModel.png)

## video2text.py
Converts video into readable text, by frame. This still needs to be processed, however, to recover any sort of passwords.

### Usage: 
```
python3 video2text.py --filename <FILENAME> [--framerate <FRAMERATE>, --mode <tesseract|google_vision>]
```

Check results ``.txt`` under ``/results`` folder.

## framecapture.py: 
Records video object such as USB HDMI passthrough capture, which should be connected to your laptop.

### Usage:
```
python3 framecapture.py --filename <FILENAME> [--framerate <FRAMERATE>]
```

