#!/usr/bin/env python3

import subprocess
import sys
import os
import torch
import numpy as np
import pyautogui
import warnings
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

warnings.filterwarnings('ignore')


class GameStateDetector:
    
    GAME_PROMPTS = [
        "a screenshot of first person shooter gameplay with a gun visible",
        "a screenshot of FPS game combat with crosshair and weapons",
        "a screenshot of a player playing a shooting game in first person view",
        "a screenshot of competitive FPS multiplayer match gameplay",
        "a screenshot of CS2 Counter-Strike 2 gameplay during a round",
    ]
    
    MENU_PROMPTS = [
        "a screenshot of a video game main menu with buttons",
        "a screenshot of a game menu interface with options",
        "a screenshot of a game lobby or waiting screen",
        "a screenshot of game settings or inventory screen",
        "a screenshot of a loading screen",
        "a screenshot of a computer desktop or web browser",
    ]
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
        self._precompute()
    
    def _precompute(self):
        all_prompts = self.GAME_PROMPTS + self.MENU_PROMPTS
        with torch.no_grad():
            inputs = self.processor(text=all_prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.n_game = len(self.GAME_PROMPTS)
    
    def check(self) -> str:
        image = pyautogui.screenshot()
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarities = (image_features @ self.text_features.T).squeeze(0).cpu().numpy()
        
        game_score = float(np.max(similarities[:self.n_game]))
        menu_score = float(np.max(similarities[self.n_game:]))
        
        return "GAME" if game_score > menu_score else "MENU"


def main():
    try:
        detector = GameStateDetector()
        result = detector.check()
        if result == "GAME":
            os._exit(0)  # In game, nothing to do
        else:
            os._exit(1)  # In menu, action needed
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        os._exit(2)  # Error state


if __name__ == "__main__":
    main()
