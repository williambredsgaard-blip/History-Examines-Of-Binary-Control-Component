#!/usr/bin/env python3

import cv2
import numpy as np
import pytesseract
import sys
import time
import subprocess # NEW: For calling gnome-screenshot
import os         # NEW: For temporary file cleanup
from PIL import Image # NEW: For loading the screenshot image
from evdev import UInput, ecodes as e
from ultralytics import YOLO

try:
    ui = UInput({
        e.EV_REL: [e.REL_X, e.REL_Y],
        e.EV_KEY: [e.BTN_LEFT]
    }, name="CheckOKMouse")
except Exception as ex:
    print("Failed to initialize UInput:", ex)
    sys.exit(1)

model = YOLO("yolov8n.pt")
button_proxy_classes = [39, 63, 67]

def grab_screenshot_gnome():
    screenshot_path = "/tmp/screenshot_ok.png"
    
    try:
        # -f: specify output file
        subprocess.run(['gnome-screenshot', '-f', screenshot_path], check=True, capture_output=True)
    except FileNotFoundError:
        print("ERROR: The 'gnome-screenshot' command was not found.")
        print("Ensure 'gnome-screenshot' is installed on your system.")
        ui.close()
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: gnome-screenshot failed. Check X environment access.")
        print(f"Details: {e.stderr.decode().strip()}")
        ui.close()
        sys.exit(1)
    
    try:
        img = Image.open(screenshot_path)
        screenshot_array = np.array(img)
        return screenshot_array
    except Exception as e:
        print(f"ERROR: Failed to load or process screenshot image. Details: {e}")
        ui.close()
        sys.exit(1)
    finally:
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)

screenshot = grab_screenshot_gnome()

screen_height, screen_width, _ = screenshot.shape
    
frame = cv2.cvtColor(screenshot, cv2.COLOR_RGBA2BGR) # Changed BGRA to RGBA for Pillow compatibility

results = model(frame)[0]
ok_candidate = None

for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
    cls = int(cls)
    if cls not in button_proxy_classes or conf < 0.3:
        continue
    
    x1, y1, x2, y2 = map(int, box.cpu().numpy())
    roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 40, 200]))
    gray_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
    if gray_ratio < 0.3:
        continue

    text = pytesseract.image_to_string(roi, config="--psm 7").strip().upper()
    if "OK" in text:
        ok_candidate = (x1, y1, x2, y2)
        break

if ok_candidate:
    x1, y1, x2, y2 = ok_candidate
    center_x = x1 + (x2 - x1) // 2
    center_y = y1 + (y2 - y1) // 2

    dx = center_x - screen_width // 2
    dy = center_y - screen_height // 2

    ui.write(e.EV_REL, e.REL_X, dx)
    ui.write(e.EV_REL, e.REL_Y, dy)
    ui.syn()
    time.sleep(0.05)

    ui.write(e.EV_KEY, e.BTN_LEFT, 1)
    ui.syn()
    time.sleep(0.05)
    ui.write(e.EV_KEY, e.BTN_LEFT, 0)
    ui.syn()

    ui.close()
    print("OK button clicked.")
    sys.exit(0)
else:
    ui.close()
    print("OK button not found.")
    sys.exit(1)
