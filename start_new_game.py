#!/usr/bin/env python3

import easyocr
import cv2
import numpy as np
from PIL import ImageGrab
import subprocess
import time
import os
import sys

# Initialize EasyOCR with GPU if available
print("Loading EasyOCR...")
try:
    import torch
    use_gpu = torch.cuda.is_available()
    print(f"   CUDA available: {use_gpu}")
except ImportError:
    use_gpu = False
    print("   PyTorch not found, using CPU")

reader = easyocr.Reader(['en'], gpu=use_gpu)
print(f"✓ EasyOCR ready ({'GPU' if use_gpu else 'CPU'})")

def start_cs2_sequence():
    print("🚀 Starting CS2 auto-queue...")
    
    # Press ESC 6 times to ensure we're at main menu
    print("⎋ Pressing ESC 6 times to return to main menu...")
    for i in range(6):
        subprocess.run(['xdotool', 'key', 'Escape'])
        time.sleep(0.15)
        print(f"   {i+1}/6")
    
    time.sleep(0.5)  # Wait for menu to settle
    
    # Step 1: Click PLAY
    print("🎯 Finding PLAY...")
    screenshot = np.array(ImageGrab.grab())
    play_pos = find_text_position(screenshot, "PLAY")
    if play_pos:
        click_at(play_pos[0], play_pos[1], "PLAY")
        time.sleep(0.5)
    else:
        print("❌ PLAY not found")
        os._exit(1)
    
    # Step 2: Try Defusal Group Alpha
    print("🎯 Finding Defusal Group Alpha...")
    screenshot = np.array(ImageGrab.grab())
    defusal_pos = find_text_position(screenshot, "Defusal Group Alpha")
    if defusal_pos:
        click_at(defusal_pos[0], defusal_pos[1], "Defusal Group Alpha")
        time.sleep(0.5)
    
    # Step 3: Click DEATHMATCH
    print("🎯 Finding DEATHMATCH...")
    screenshot = np.array(ImageGrab.grab())
    deathmatch_pos = find_text_position(screenshot, "DEATHMATCH")
    if deathmatch_pos:
        click_at(deathmatch_pos[0], deathmatch_pos[1], "DEATHMATCH")
        time.sleep(0.5)
    
    # Step 4: Click green button
    print("🎯 Finding green button...")
    screenshot = np.array(ImageGrab.grab())
    green_button_pos = find_green_button(screenshot)
    if green_button_pos:
        click_at(green_button_pos[0], green_button_pos[1], "green button")
        print("✅ Queue started!")
        os._exit(0)
    else:
        print("❌ Green button not found")
        os._exit(1)

def find_text_position(img, target_text):
    results = reader.readtext(img)
    for (bbox, text, confidence) in results:
        if target_text.lower() in text.lower() and confidence > 0.4:
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x = int(np.mean(x_coords))
            y = int(np.mean(y_coords))
            print(f"   Found '{text}' at ({x}, {y}) conf={confidence:.2f}")
            return (x, y)
    print(f"   '{target_text}' not found")
    return None

def find_green_button(img):
    height, width = img.shape[:2]
    bottom_right = img[height//2:, width//2:]
    hsv = cv2.cvtColor(bottom_right, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            full_x = x + width//2 + w//2
            full_y = y + height//2 + h//2
            print(f"   Found green button at ({full_x}, {full_y})")
            return (full_x, full_y)
    return None

def click_at(x, y, button_name):
    print(f"   Clicking {button_name} at ({x}, {y})")
    subprocess.run(['xdotool', 'mousemove', '--sync', str(x), str(y)])
    time.sleep(0.1)
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(0.1)

def main():
    print("\n" + "=" * 50)
    print("CS2 Auto-Queue")
    print("=" * 50)
    
    start_cs2_sequence()


if __name__ == "__main__":
    main()
