#!/usr/bin/env python3
"""
CS2 WALKBOT by Daniel Marcu
===================================================

Works only on Linux/Ubuntu, it starts immediately
after running the script.

Controls:
    P = Start/Pause
    Q = Quit

Requirements:
- /dev/uinput write access (sudo usermod -aG input $USER)
"""

import cv2
import numpy as np
import time
import random
import sys
import os
import math
from collections import deque
from pathlib import Path
from PIL import Image
import warnings
import platform

warnings.filterwarnings('ignore')

# Dependency management
def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try: import torch
except ImportError: install("torch torchvision"); import torch

try: from transformers import pipeline
except ImportError: install("transformers"); from transformers import pipeline

try: from ultralytics import YOLO
except ImportError: install("ultralytics"); from ultralytics import YOLO

try: from mss import mss
except ImportError: install("mss"); from mss import mss

try: from pynput.keyboard import Key, Controller as KBController, Listener as KBListener
except ImportError: install("pynput"); from pynput.keyboard import Key, Controller as KBController, Listener as KBListener

try: from evdev import UInput, ecodes as e
except ImportError: install("evdev"); from evdev import UInput, ecodes as e

# ============================================================================
# ENEMY DETECTOR (YOLOv8 trained on CS2)
# ============================================================================

class EnemyDetector:
    def __init__(self, device: str):
        import os
        import time
        from pathlib import Path

        # ----- filesystem & params file -----
        self.data_dir = Path("navigation_data")
        self.data_dir.mkdir(exist_ok=True)
        self.params_file = self.data_dir / "enemy_detector.txt"
        self.weights_file = self.data_dir / "yolov8_csgo_cs2_model.pt"

        # ----- defaults -----old
        self.params = {
            "conf_threshold": 0.45,     # baseline detection confidence thresh
            "conf_min": 0.25,           # lower clamp for adaptive threshold
            "conf_max": 0.85,           # upper clamp
            "prioritize_head": 1,       # whether to prefer head detections (1 = yes)
            "use_local_weights": 1,     # if 1, try to load weights from navigation_data/
            "download_if_missing": 1,   # if 1 and local missing, attempt to download to weights_file
            "model_url": "https://github.com/siromermer/CS2-CSGO-Yolov8-Yolov7-ObjectDetection/raw/master/yolov8_csgo_cs2_model.pt",
            "nms_iou": 0.45,
            "smoothing_alpha": 0.2,     # EMA alpha for smoothing detection confidences
            "adapt_lr": 0.02,           # learning rate for adaptive conf threshold
            "prior_center_weight": 0.6, # weight for center proximity when ranking
            # persistent perf counters used for adaptation:
            "perf_hits": 0,
            "perf_misses": 0,
            "last_saved": 0.0
        }

        # ----- load params from .txt if present -----
        if self.params_file.exists():
            try:
                with open(self.params_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        v = v.strip()
                        # try cast to float or int when possible
                        try:
                            if "." in v:
                                val = float(v)
                                # if integer-like, cast to int
                                if val.is_integer():
                                    val = int(val)
                            else:
                                val = int(v)
                        except Exception:
                            val = v
                        self.params[k] = val
                # Ensure numeric defaults exist if missing
            except Exception as ex:
                print(f"[EnemyDetector] failed loading params: {ex}")

        # Ensure all keys exist (setdefault won't overwrite loaded values)
        for k, v in dict(self.params).items():
            self.params.setdefault(k, v)

        # ----- runtime state -----
        self.device = device
        self.model = None
        self.classes = ['ct_body', 'ct_head', 't_body', 't_head']
        self._ema_conf = None   # exponential moving average for confidences (single scalar)
        self._last_feedback_time = time.time()

        # ----- try to load model -----
        print("Loading YOLOv8 CS2 enemy detection...")
        try:
            # prefer local weights if requested
            from ultralytics import YOLO  # may raise
            model_path = None
            if int(self.params.get("use_local_weights", 1)) and self.weights_file.exists():
                model_path = str(self.weights_file)
            elif int(self.params.get("download_if_missing", 1)):
                # download synchronously if missing
                if not self.weights_file.exists():
                    try:
                        import urllib.request, ssl
                        print("  Downloading model to navigation_data/ (may take a moment)...")
                        ctx = ssl.create_default_context()
                        ctx.check_hostname = False
                        ctx.verify_mode = ssl.CERT_NONE
                        urllib.request.urlretrieve(self.params.get("model_url"), str(self.weights_file), context=ctx)
                        model_path = str(self.weights_file)
                    except Exception as exd:
                        print(f"  Warning: model download failed: {exd}")
                        model_path = None
                else:
                    model_path = str(self.weights_file)
            # fallback to model_path as None which allows YOLO() to try builtin names (not ideal)
            if model_path is None:
                # attempt default load (may raise)
                self.model = YOLO(self.params.get("model_url"))
            else:
                self.model = YOLO(model_path)

            # move to device if possible
            if self.model is not None and self.device == "cuda":
                try:
                    self.model.to("cuda")
                except Exception:
                    pass
            print("✓ Enemy detection ready")
        except Exception as ex:
            # If anything fails, keep model None but do not crash
            self.model = None
            print(f"⚠ Enemy detection unavailable: {ex}")

        # ----- schedule initial save of params so missing keys get persisted -----
        try:
            self.save_params()
        except Exception:
            pass

    # -------------------------
    # params persistence
    # -------------------------
    def save_params(self):
        """Write current params to navigation_data/enemy_detector.txt"""
        import time
        try:
            with open(self.params_file, "w") as f:
                for k, v in self.params.items():
                    f.write(f"{k}={v}\n")
            self.params['last_saved'] = time.time()
        except Exception as ex:
            print(f"[EnemyDetector] failed to save params: {ex}")

    def update_param(self, key, value):
        """Update single parameter and persist"""
        self.params[key] = value
        self.save_params()

    # -------------------------
    # feedback & adaptation
    # -------------------------
    def update_feedback(self, hit: bool):
        """
        Call this after an engagement:
          - hit=True  -> we consider detection useful (reinforce)
          - hit=False -> consider lowering sensitivity or adapt parameters
        This increments counters and adjusts conf_threshold adaptively.
        """
        import time, math
        lr = float(self.params.get("adapt_lr", 0.02))
        conf = float(self.params.get("conf_threshold", 0.45))

        if hit:
            self.params["perf_hits"] = int(self.params.get("perf_hits", 0)) + 1
            # raise threshold slightly (be more confident) up to conf_max
            conf = min(float(self.params.get("conf_max", 0.85)), conf * (1.0 + lr))
        else:
            self.params["perf_misses"] = int(self.params.get("perf_misses", 0)) + 1
            # lower threshold slightly so we catch more candidates
            conf = max(float(self.params.get("conf_min", 0.25)), conf * (1.0 - lr))

        self.params["conf_threshold"] = conf
        self._last_feedback_time = time.time()

        # occasionally persist
        if time.time() - float(self.params.get("last_saved", 0)) > 30:
            self.save_params()

    # -------------------------
    # detection
    # -------------------------
    def detect(self, frame: "np.ndarray"):
        """
        Run detection (if model present) and return list of detections.
        Each detection:
         {'x1','y1','x2','y2','cx','cy','w','h','conf','class','is_head'}
        """
        import numpy as np, math, time

        if self.model is None:
            return []

        # run model inference; adapt params for conf & nms
        conf_thresh = float(self.params.get("conf_threshold", 0.45))
        iou = float(self.params.get("nms_iou", 0.45))

        # ultralytics YOLO may accept parameters in call; handle gracefully
        try:
            # returns a Results object list
            # Note: the exact call signature depends on ultralytics version
            results = self.model(frame, conf=conf_thresh, iou=iou, verbose=False)
        except TypeError:
            # some versions require different kwargs
            try:
                results = self.model.predict(source=frame, conf=conf_thresh, iou=iou, verbose=False)
            except Exception as ex:
                print(f"[EnemyDetector] model inference failed: {ex}")
                return []
        except Exception as ex:
            print(f"[EnemyDetector] model inference exception: {ex}")
            return []

        detections = []
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame / 2.0
        center_y = h_frame / 2.0

        # iterate results
        try:
            for r in results:
                # some ultilities put boxes in r.boxes, others in r.boxes.xyxy - handle common layouts
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    try:
                        # get xyxy
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        conf = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else float(box.conf)
                        cls_idx = int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else int(box.cls)
                    except Exception:
                        # fallback to alternative attributes
                        try:
                            x1, y1, x2, y2 = map(float, box.xyxy)
                            conf = float(box.conf)
                            cls_idx = int(box.cls)
                        except Exception:
                            continue

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1
                    class_name = self.classes[cls_idx] if cls_idx < len(self.classes) else "unknown"
                    is_head = ("head" in class_name)

                    det = {
                        "x1": int(round(x1)), "y1": int(round(y1)),
                        "x2": int(round(x2)), "y2": int(round(y2)),
                        "cx": cx, "cy": cy, "w": w, "h": h,
                        "conf": conf, "class": class_name, "is_head": is_head
                    }

                    # Apply simple EMA smoothing of confidences across frames (single scalar)
                    if self._ema_conf is None:
                        self._ema_conf = conf
                    else:
                        alpha = float(self.params.get("smoothing_alpha", 0.2))
                        self._ema_conf = alpha * conf + (1 - alpha) * self._ema_conf
                        # optionally replace conf with EMA-smoothed confidence
                        det["conf_ema"] = self._ema_conf

                    detections.append(det)
        except Exception as ex:
            print(f"[EnemyDetector] parsing results failed: {ex}")
            return []

        # Ranking: prefer heads (if enabled), then higher conf (or EMA conf), then proximity to center
        def score_det(d):
            # head priority
            head_bonus = 1.0 if d.get("is_head", False) and int(self.params.get("prioritize_head", 1)) else 0.0
            conf_score = d.get("conf_ema", d.get("conf", 0.0))
            # distance penalty (closer to center is better), normalized 0..1
            dx = abs(d["cx"] - center_x) / max(1.0, center_x)
            dy = abs(d["cy"] - center_y) / max(1.0, center_y)
            dist_pen = (dx + dy) / 2.0
            center_score = max(0.0, 1.0 - dist_pen)
            return (head_bonus * 2.0) + conf_score + center_score * float(self.params.get("prior_center_weight", 0.6))

        # sort descending by score
        detections.sort(key=lambda d: score_det(d), reverse=True)

        # Optionally prune by conf_min (keep only reasonably confident)
        conf_min = float(self.params.get("conf_min", 0.25))
        filtered = [d for d in detections if d.get("conf", 0.0) >= conf_min]

        # Return filtered & sorted detections
        return filtered

    # -------------------------
    # utilities
    # -------------------------
    def set_weights_path(self, path: str):
        """
        Point to a specific local weights file and reload model.
        Example: detector.set_weights_path("navigation_data/my_weights.pt")
        """
        from pathlib import Path
        try:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(str(p))
            # update local pointer and attempt reload
            self.weights_file = p
            # attempt reload if ultralytics is present
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.weights_file))
                if self.device == "cuda":
                    try:
                        self.model.to("cuda")
                    except Exception:
                        pass
                print(f"[EnemyDetector] reloaded model from {p}")
            except Exception as ex:
                print(f"[EnemyDetector] failed to reload model: {ex}")
        except Exception as ex:
            print(f"[EnemyDetector] set_weights_path error: {ex}")


# Semantic categories
BLOCKED_CLASSES = {

    # --- Static surfaces / geometry ---
    'wall', 'floor', 'ceiling', 'ground', 'terrain', 'hill', 'slope',
    'cliff', 'rock_wall', 'stone_wall', 'mountain',

    # --- Buildings / structures ---
    'building', 'structure', 'tower', 'bunker', 'shed', 'garage',
    'hangar', 'warehouse', 'container_home', 'storage_unit',

    # --- Doors & windows ---
    'door', 'doorframe', 'shutter', 'window', 'window_frame',
    'glass_panel', 'metal_door', 'garage_door', 'gate',

    # --- Fences & railings ---
    'fence', 'chainlink_fence', 'metal_fence', 'wood_fence', 'barbed_wire',
    'railing', 'balcony_rail', 'guard_rail', 'handrail',

    # --- Obstacles / barriers ---
    'barrier', 'blocker', 'roadblock', 'gate_barrier', 'bollard',
    'turnstile', 'security_barrier', 'crowd_barrier',

    # --- Furniture (common blockers) ---
    'table', 'desk', 'chair', 'sofa', 'bed', 'couch', 'bench',
    'bookshelf', 'drawer', 'cabinet', 'locker', 'cupboard',
    'wardrobe', 'nightstand', 'dresser', 'counter', 'kitchen_island',

    # --- Props / clutter (common map assets) ---
    'box', 'crate', 'barrel', 'trashcan', 'dumpster', 'bucket',
    'paint_can', 'toolbox', 'fire_hydrant', 'traffic_cone',
    'traffic_barrel', 'plastic_container', 'metal_container',
    'pallet', 'wood_pallet', 'metal_pallet',

    # --- Industrial / mechanical ---
    'machinery', 'generator', 'compressor', 'air_conditioner',
    'industrial_fan', 'pump', 'furnace', 'tank', 'valve',
    'control_panel', 'concrete_mixer', 'scaffolding', 'forklift',

    # --- Vehicles (big blockers) ---
    'car', 'truck', 'bus', 'van', 'tractor', 'apc', 'jeep',
    'motorcycle', 'quadbike', 'armored_vehicle',

    # --- Vegetation ---
    'tree', 'bush', 'plant', 'hedge', 'shrub', 'potted_plant',
    'large_shrub', 'fallen_tree', 'vines',

    # --- Poles & vertical blockers ---
    'pole', 'streetlight', 'lamp_post', 'traffic_light', 'antenna',
    'flagpole', 'signpost', 'utility_pole',

    # --- Architectural interior elements ---
    'column', 'pillar', 'arch', 'support_beam', 'girder', 'truss',

    # --- Stairs / elevation blockers ---
    'stairs', 'steps', 'staircase', 'ladder', 'ramp', 'elevator_shaft',

    # --- Screens, boards ---
    'billboard', 'sign', 'poster_board', 'whiteboard', 'chalkboard',
    'tv_screen', 'monitor', 'terminal',

    # --- Shelves / storage ---
    'shelf', 'bookshelf', 'rack', 'storage_rack', 'server_rack',

    # --- Containers ---
    'shipping_container', 'cargo_container', 'barrel_stack',
    'supply_box', 'loot_crate',

    # --- Pipes / vents / ducts ---
    'pipe', 'vent', 'air_duct', 'gas_pipe', 'steam_pipe',

    # --- Rubble / debris ---
    'rubble', 'debris', 'broken_wall', 'wreckage', 'collapsed_structure',
    'concrete_block', 'brick_pile',

    # --- Natural obstacles ---
    'rock', 'boulder', 'log', 'driftwood', 'large_branch',

    # --- Water obstacles ---
    'water_tank', 'fountain', 'pool_edge',

    # --- Gameplay-specific blockers ---
    'bombsite_boundary', 'map_boundary', 'invisible_wall',
    'clip_brush', 'collision_proxy', 'hitbox_blocker',

    # --- Characters as obstacles ---
    'person', 'npc', 'enemy', 'teammate',

    # --- Misc ---
    'kiosk', 'vending_machine', 'atm', 'telephone_booth',
    'dispenser', 'server_cabinet',

    # --- Special CS2/CSGO props ---
    'sandbag', 'sandbag_wall', 'metal_crate', 'wooden_crate',
    'ammo_box', 'weapon_case', 'prop_door', 'prop_wall',
    'bomb_crate', 'barbed_fence', 'low_wall',
}

# ============================================================================
# VIRTUAL MOUSE (Linux/Ubuntu with evdev)
# ============================================================================

class VirtualMouse:
    def __init__(self):

        # persistence location
        self.data_dir = Path("navigation_data")
        self.data_dir.mkdir(exist_ok=True)
        self.params_file = self.data_dir / "virtual_mouse.txt"

        # defaults
        self.params = {
            "sensitivity": 1.0,        # global multiplier
            "axis_scale_x": 1.25,      # make horizontal slightly stronger by default
            "axis_scale_y": 1.0,
            "horiz_aggression": 1.15,  # extra multiplier applied to horizontal moves
            "invert_y": 0,
            "acceleration": 0.0,
            "smoothing_alpha_x": 0.28, # X uses lower smoothing -> more responsive
            "smoothing_alpha_y": 0.42, # Y can be smoother to avoid vertical jitter
            "max_move": 300,           # increased clamp to avoid being timid
            "deadzone_detect_px": 0.8, # if |dx_total| > this but integer move==0 -> underflow event
            "underflow_window": 40,    # history length to judge underflow frequency
            "underflow_ratio_threshold": 0.25, # fraction to trigger adaptation
            "adapt_lr": 0.06,          # adaptation step for axis_scale_x / horiz_aggression
            "min_axis_scale_x": 0.7,
            "max_axis_scale_x": 3.5,
            "min_horiz_aggression": 0.8,
            "max_horiz_aggression": 3.0,
            "tap_duration_ms": 12,
            "hold_min_time": 0.02,
            "emulate_if_no_uinput": 1
        }

        # runtime state
        self.ui = None
        self._uinput_ecodes = None
        self._fallback_controller = None
        self.available = False
        self.shooting = False

        # EMAs, residuals
        self._ema_x = 0.0
        self._ema_y = 0.0
        self._residual_x = 0.0
        self._residual_y = 0.0

        # underflow detection (when large fractional move gets rounded to 0)
        self._underflow_hist = deque(maxlen=int(self.params["underflow_window"]))
        self._last_params_save = 0.0

        # load file params (if present) and ensure keys
        try:
            if self.params_file.exists():
                with open(self.params_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        try:
                            if "." in v:
                                val = float(v)
                                if val.is_integer():
                                    val = int(val)
                            else:
                                val = int(v)
                        except Exception:
                            val = v
                        self.params[k] = val
        except Exception:
            pass

        # ensure all defaults exist (won't overwrite loaded values)
        for k, v in dict(self.params).items():
            self.params.setdefault(k, v)

        # setup uinput or fallback
        try:
            from evdev import UInput, ecodes as e
            # verify /dev/uinput existence and permission
            if not os.path.exists('/dev/uinput'):
                raise FileNotFoundError('/dev/uinput not found')
            if not os.access('/dev/uinput', os.W_OK):
                raise PermissionError('/dev/uinput not writable')

            self.ui = UInput({
                e.EV_REL: [e.REL_X, e.REL_Y],
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
            }, name='CS2-Combat-Mouse')
            self._uinput_ecodes = e
            self.available = True
            # short settle
            time.sleep(0.03)
            # print success
            print("✓ Virtual mouse (uinput) initialized")
        except Exception as ex:
            # fallback to pynput if allowed
            if int(self.params.get("emulate_if_no_uinput", 1)):
                try:
                    from pynput.mouse import Controller
                    self._fallback_controller = Controller()
                    self.available = True
                    print("⚠ uinput unavailable, falling back to pynput controller")
                except Exception as ex2:
                    print(f"⚠ Virtual mouse init failed (uinput & pynput): {ex} / {ex2}")
                    self.available = False
            else:
                print(f"⚠ Virtual mouse init failed: {ex}")
                self.available = False

        # persist params (may include defaults not in file)
        try:
            self.save_params()
        except Exception:
            pass

    # ---------------------------
    # persist / update
    # ---------------------------
    def save_params(self):
        """Persist current tuning parameters to file."""
        import time
        try:
            with open(self.params_file, "w") as f:
                for k, v in self.params.items():
                    f.write(f"{k}={v}\n")
            self._last_params_save = time.time()
        except Exception as ex:
            print(f"[VirtualMouse] failed saving params: {ex}")

    def update_param(self, key, value):
        self.params[key] = value
        self.save_params()

    # ---------------------------
    # core movement
    # ---------------------------
    def move(self, dx: int, dy: int):
        """
        Apply horizontal-aggressive, simpler, adaptive movement:
         - horizontal gets axis_scale_x * horiz_aggression multipliers
         - separate smoothing for X and Y
         - detect underflow (large fractional but zero integer movement) and adapt
        """
        import math, time

        if not self.available:
            return

        # no-op quick return
        if dx == 0 and dy == 0:
            return

        # read params and cast to floats
        s = float(self.params.get("sensitivity", 1.0))
        ax = float(self.params.get("axis_scale_x", 1.0))
        ay = float(self.params.get("axis_scale_y", 1.0))
        horiz_aggr = float(self.params.get("horiz_aggression", 1.0))
        invert_y = int(self.params.get("invert_y", 0))
        accel = float(self.params.get("acceleration", 0.0))
        alpha_x = float(self.params.get("smoothing_alpha_x", 0.28))
        alpha_y = float(self.params.get("smoothing_alpha_y", 0.42))
        max_move = float(self.params.get("max_move", 300.0))
        deadzone_detect = float(self.params.get("deadzone_detect_px", 0.8))

        # scale inputs; be braver horizontally
        dx_scaled = dx * s * ax * horiz_aggr
        dy_scaled = dy * s * ay * (-1.0 if invert_y else 1.0)

        # acceleration (kept simple)
        if accel != 0.0:
            mag = (abs(dx_scaled) + abs(dy_scaled)) / 2.0
            factor = 1.0 + accel * (mag / max(1.0, max_move))
            dx_scaled *= factor
            dy_scaled *= factor

        # EMA smoothing applied separately
        self._ema_x = alpha_x * dx_scaled + (1.0 - alpha_x) * self._ema_x
        self._ema_y = alpha_y * dy_scaled + (1.0 - alpha_y) * self._ema_y

        # add residuals from previous fractional moves
        dx_total = self._ema_x + self._residual_x
        dy_total = self._ema_y + self._residual_y

        # clamp
        dx_clamped = max(-max_move, min(max_move, dx_total))
        dy_clamped = max(-max_move, min(max_move, dy_total))

        # integer movement and store residuals
        ix = int(math.copysign(int(abs(dx_clamped) + 0.5), dx_clamped)) if abs(dx_clamped) >= 0.5 else 0
        iy = int(math.copysign(int(abs(dy_clamped) + 0.5), dy_clamped)) if abs(dy_clamped) >= 0.5 else 0

        # preserve sub-pixel residuals more accurately
        self._residual_x = dx_total - ix
        self._residual_y = dy_total - iy

        # underflow detection: we attempted a significant fractional motion but integer result was zero
        underflow_event = (abs(dx_total) >= deadzone_detect and ix == 0)
        self._underflow_hist.append(1 if underflow_event else 0)

        # apply move to device
        try:
            if self.ui and self._uinput_ecodes:
                e = self._uinput_ecodes
                if ix != 0:
                    self.ui.write(e.EV_REL, e.REL_X, int(ix))
                if iy != 0:
                    self.ui.write(e.EV_REL, e.REL_Y, int(iy))
                if ix != 0 or iy != 0:
                    self.ui.syn()
            elif self._fallback_controller:
                try:
                    self._fallback_controller.move(ix, iy)
                except Exception:
                    try:
                        x, y = self._fallback_controller.position
                        self._fallback_controller.position = (int(x + ix), int(y + iy))
                    except Exception:
                        pass
            else:
                # nothing to do
                return
        except Exception:
            # non-fatal
            return

        # ---------------------------
        # Adaptive adjustment: if many underflow events recently, increase horizontal gain
        # ---------------------------
        # compute underflow ratio over history
        hist = list(self._underflow_hist)
        if len(hist) >= int(self.params.get("underflow_window", 40)) // 4:  # start adapting after a few samples
            ratio = sum(hist) / max(1, len(hist))
            thresh = float(self.params.get("underflow_ratio_threshold", 0.25))
            if ratio > thresh:
                # adapt axis_scale_x and horiz_aggression upward gently
                lr = float(self.params.get("adapt_lr", 0.06))
                new_ax = min(float(self.params.get("max_axis_scale_x", 3.5)),
                             ax * (1.0 + lr))
                new_hagg = min(float(self.params.get("max_horiz_aggression", 3.0)),
                               horiz_aggr * (1.0 + lr * 0.6))
                # update params in-memory and persist occasionally
                self.params["axis_scale_x"] = new_ax
                self.params["horiz_aggression"] = new_hagg
                # reset underflow history to avoid runaway increases
                self._underflow_hist.clear()
                # save occasionally (throttle)
                import time
                if time.time() - self._last_params_save > 20.0:
                    try:
                        self.save_params()
                    except Exception:
                        pass
                    self._last_params_save = time.time()

        # small safeguard: ensure axis_scale_x / horiz_aggression remain within allowed bounds
        self.params["axis_scale_x"] = max(float(self.params.get("min_axis_scale_x", 0.7)),
                                          min(float(self.params.get("max_axis_scale_x", 3.5)),
                                              float(self.params.get("axis_scale_x"))))
        self.params["horiz_aggression"] = max(float(self.params.get("min_horiz_aggression", 0.8)),
                                              min(float(self.params.get("max_horiz_aggression", 3.0)),
                                                  float(self.params.get("horiz_aggression"))))

    # ---------------------------
    # tap / hold / helpers
    # ---------------------------
    def tap(self, button='left'):
        import time
        if not self.available:
            return
        dur = float(self.params.get("tap_duration_ms", 12)) / 1000.0
        try:
            if self.ui and self._uinput_ecodes:
                e = self._uinput_ecodes
                code = e.BTN_LEFT if button == 'left' else e.BTN_RIGHT
                self.ui.write(e.EV_KEY, code, 1); self.ui.syn()
                time.sleep(max(0.002, dur))
                self.ui.write(e.EV_KEY, code, 0); self.ui.syn()
            elif self._fallback_controller:
                from pynput.mouse import Button
                b = Button.left if button == 'left' else Button.right
                self._fallback_controller.press(b)
                time.sleep(max(0.002, dur))
                self._fallback_controller.release(b)
        except Exception:
            pass

    def hold(self, pressed: bool):
        import time
        if not self.available:
            return
        now = time.time()
        if pressed:
            if not self.shooting:
                try:
                    if self.ui and self._uinput_ecodes:
                        e = self._uinput_ecodes
                        self.ui.write(e.EV_KEY, e.BTN_LEFT, 1); self.ui.syn()
                    elif self._fallback_controller:
                        from pynput.mouse import Button
                        self._fallback_controller.press(Button.left)
                except Exception:
                    pass
                self.shooting = True
                self._last_hold_time = now
        else:
            min_hold = float(self.params.get("hold_min_time", 0.02))
            if self.shooting and (now - getattr(self, "_last_hold_time", 0)) < min_hold:
                try:
                    time.sleep(min_hold - (now - getattr(self, "_last_hold_time", 0)))
                except Exception:
                    pass
            if self.shooting:
                try:
                    if self.ui and self._uinput_ecodes:
                        e = self._uinput_ecodes
                        self.ui.write(e.EV_KEY, e.BTN_LEFT, 0); self.ui.syn()
                    elif self._fallback_controller:
                        from pynput.mouse import Button
                        self._fallback_controller.release(Button.left)
                except Exception:
                    pass
                self.shooting = False

    # ---------------------------
    # convenience
    # ---------------------------
    def set_sensitivity(self, sens: float):
        self.params['sensitivity'] = float(sens)
        try:
            self.save_params()
        except Exception:
            pass

    def close(self):
        try:
            if getattr(self, "shooting", False):
                try:
                    self.hold(False)
                except Exception:
                    pass
            if getattr(self, "ui", None):
                try:
                    self.ui.close()
                except Exception:
                    pass
                self.ui = None
            # fallback controller needs no explicit close
        except Exception:
            pass
        finally:
            self.available = False
            try:
                self.save_params()
            except Exception:
                pass



# ============================================================================
# PERCEPTION SYSTEM
# ============================================================================

class Perception:
    PARAMS_PATH = "navigation_data/perception_params.txt"

    def __init__(self):
        # Ensure folder exists
        os.makedirs("navigation_data", exist_ok=True)
        
        # Load or initialize parameters
        self.params = self._load_params()

        # Device selection
        self.device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            props = torch.cuda.get_device_properties(0)
            free_mem = props.total_memory - torch.cuda.memory_allocated(0)
            if free_mem > 2 * 1024**3:
                self.device = "cuda"
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                print(f"GPU: {props.name} ({free_mem/1024**3:.1f}GB free)")
        print(f"Device: {self.device.upper()}")

        dev_id = 0 if self.device == "cuda" else -1
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load models
        print("Loading SegFormer-B4...")
        self.seg_pipe = pipeline(
            "image-segmentation",
            model="nvidia/segformer-b4-finetuned-ade-512-512",
            device=dev_id, torch_dtype=dtype
        )

        print("Loading Depth Anything V2...")
        self.depth_pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=dev_id, torch_dtype=dtype
        )

        # Warmup
        print("Warming up models...")
        dummy = Image.new('RGB', (256, 256), color='gray')
        self.seg_pipe(dummy)
        self.depth_pipe(dummy)

        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("✓ Ready!\n")

        # Enemy detection
        self.enemy_detector = EnemyDetector(self.device)

    # ---------------------------
    # Parameter management (plain text)
    # ---------------------------
    def _load_params(self):
        params = {}
        if os.path.exists(self.PARAMS_PATH):
            try:
                with open(self.PARAMS_PATH, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        # Convert lists if needed
                        if value.startswith("[") and value.endswith("]"):
                            params[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                        else:
                            try:
                                params[key] = float(value)
                            except ValueError:
                                params[key] = value
                print("Loaded tuning parameters.")
                return params
            except Exception as e:
                print(f"Failed to load params: {e}")
        # Default parameters
        return {
            "blocked_classes": ["wall", "fence", "rock"],  # example default
            "depth_scale": 1.0
        }

    def save_params(self):
        with open(self.PARAMS_PATH, "w") as f:
            for key, value in self.params.items():
                if isinstance(value, list):
                    value_str = "[" + ", ".join(map(str, value)) + "]"
                else:
                    value_str = str(value)
                f.write(f"{key}={value_str}\n")
        print("Saved tuning parameters.")

    def update_param(self, key, value):
        self.params[key] = value
        self.save_params()

    # ---------------------------
    # Perception methods
    # ---------------------------
    def get_depth(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (768, 432))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self.depth_pipe(Image.fromarray(rgb))
        depth = np.array(result["depth"]).astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth *= self.params.get("depth_scale", 1.0)
        return cv2.resize(depth, (w, h))

    def get_segmentation(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (512, 512))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        results = self.seg_pipe(Image.fromarray(rgb))
        blocked = np.zeros((512, 512), dtype=np.float32)

        blocked_classes = self.params.get("blocked_classes", [])
        for seg in results:
            label = seg['label'].lower()
            if any(b in label for b in blocked_classes):
                mask = np.array(seg['mask']).astype(np.float32) / 255.0
                blocked = np.maximum(blocked, mask)

        return cv2.resize(blocked, (w, h))

    def detect_enemies(self, frame: np.ndarray):
        return self.enemy_detector.detect(frame)

# ============================================================================
# NAVIGATOR
# ============================================================================

class Navigator:
    def __init__(self):
        import os, time
        from pathlib import Path
        from collections import deque
        from mss import mss

        # ---------------------------
        # Config & parameter file
        # ---------------------------

        self.data_dir = Path("navigation_data")
        self.data_dir.mkdir(exist_ok=True)
        self.params_file = self.data_dir / "navigation_init.txt"


        # Default parameters
        default_params = {
            "history_len": 20,
            "gradient_len": 20,  # FIX: Increased for better trend prediction
            "gap_len": 15,  # FIX: Increased for smoother corridor tracking
            "stuck_history_len": 50,
            "screen_margin": 15,
            "turn_sensitivity": 0.5,
            "gap_threshold": 0.3,
            "learning_rate": 0.08,  # FIX: Boosted for faster adaptation
            "aim_smoothing": 0.15,
            "fire_confidence": 0.50
        }

        self.params = default_params.copy()

        # Load if file exists
        if self.params_file.exists():
            try:
                with open(self.params_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        try:
                            value = float(value)
                            if value.is_integer():
                                value = int(value)
                        except ValueError:
                            pass
                        self.params[key] = value
                print(f"Loaded parameters from {self.params_file.name}")
            except Exception as e:
                print(f"Failed to load params: {e}")

        # ---------------------------
        # Ensure essential keys exist
        # ---------------------------
        self.params.setdefault('obs_threshold', 0.4)  # FIX: Lowered for earlier obstacle detection
        self.params.setdefault('emergency_threshold', 0.8)
        self.params.setdefault('turn_commitment', 15)       # frames to commit to a turn
        self.params.setdefault('history_len', 10)
        self.params.setdefault('gradient_len', 20)  # FIX: Increased
        self.params.setdefault('gap_len', 15)  # FIX: Increased
        self.params.setdefault('learning_rate', 0.08)  # FIX: Increased
        self.params.setdefault('threshold_factor', 0.25)  # FIX: Lowered for more gaps
        self.params.setdefault('kernel_ratio', 0.08)  # FIX: Increased for smoother gaps
        self.params.setdefault('strip_top_ratio', 0.33)
        self.params.setdefault('min_gap_ratio', 0.08)  # FIX: Lowered for narrow distant paths
        self.params.setdefault('gradient_factor', 1.0)      # gradient scaling factor
        self.params.setdefault('stuck_threshold', 0.005)  # FIX: Raised to reduce false positives
        self.params.setdefault('recovery_time', 1.5)  # FIX: Extended for better recovery

        # Save helper
        def save_params():
            with open(self.params_file, "w") as f:
                for k, v in self.params.items():
                    f.write(f"{k}={v}\n")
            print(f"Saved parameters to {self.params_file.name}")
        self.save_params = save_params

        # ---------------------------
        # Screen capture
        # ---------------------------
        self.sct = mss()
        mon = self.sct.monitors[1]
        margin = self.params.get("screen_margin", 15)
        self.region = {
            "left": margin, "top": 25,
            "width": mon["width"] - margin * 2,
            "height": mon["height"] - 50
        }
        self.w, self.h = self.region['width'], self.region['height']
        print(f"Screen: {self.w}x{self.h}")

        # ---------------------------
        # Perception
        # ---------------------------
        self.perception = Perception()

        # ---------------------------
        # Controls
        # ---------------------------
        self.kb = KBController()
        self.mouse = VirtualMouse()
        self.held_keys = set()
        print(f"Mouse: {'✓' if self.mouse.available else '✗'}")

        # ---------------------------
        # State
        # ---------------------------
        self.velocity = 0.0
        self.score_history = {k: deque(maxlen=self.params['history_len']) for k in ['left', 'center', 'right']}
        self.gradient_history = deque(maxlen=self.params['gradient_len'])
        self.gap_history = deque(maxlen=self.params['gap_len'])
        self.turn_commitment = 0
        self.committed_dir = None
        self.in_corridor = False
        self.corridor_center = 0.5
        self.stuck_history = deque(maxlen=self.params.get("stuck_history_len", 50))
        self.stuck_counter = 0
        self.recovery_mode = False
        self.recovery_start = 0

        # ---------------------------
        # Timing & caching
        # ---------------------------
        self.last_jump = 0
        self.frame_count = 0
        self.last_seg_frame = -100
        self.last_enemy_frame = -100
        self.cached_blocked = None
        self.cached_enemies = []

        # ---------------------------
        # Combat
        # ---------------------------
        self.shooting = False

        # ---------------------------
        # Performance tracking for adaptive learning
        # ---------------------------
        self.perf_metrics = {
            'shots_fired': 0,
            'time_on_target': 0,
            'enemies_engaged': 0,
            'stuck_events': 0,
            'recovery_count': 0,
            'total_frames': 0
        }
        self.last_perf_update = time.time()
        self.last_param_save = time.time()

        # ---------------------------
        # Learning functions
        # ---------------------------
        def update_perf_metric(key, value=1):
            self.perf_metrics[key] = self.perf_metrics.get(key, 0) + value
            self.perf_metrics['total_frames'] += 1
        self.update_perf_metric = update_perf_metric

        def learn():
            stuck_ratio = self.perf_metrics['stuck_events'] / max(1, self.perf_metrics['total_frames'])
            target_ratio = self.perf_metrics['time_on_target'] / max(1, self.perf_metrics['total_frames'])
            lr = self.params.get("learning_rate", 0.05)

            # Adaptive tuning of turn sensitivity
            if stuck_ratio > 0.05:
                self.params['turn_sensitivity'] = max(0.1, self.params['turn_sensitivity'] * (1 - lr))
            if target_ratio < 0.3:
                self.params['turn_sensitivity'] = min(1.0, self.params['turn_sensitivity'] * (1 + lr))

            # Periodically save parameters
            now = time.time()
            if now - self.last_param_save > 60:
                self.save_params()
                self.last_param_save = now
        self.learn = learn

    
    def _load_config(self):
        config_file = self.config_dir / "params.txt"
        defaults = {
            'sensitivity': 22.0,
            'turn_speed': 3.0,
            'history_len': 8,
            'gradient_len': 15,
            'gap_len': 10,
            'obs_threshold': 0.48,
            'emergency_threshold': 0.65,
            'turn_commitment': 18,
            'corridor_commitment': 25,
            'stuck_threshold': 0.001,
            'recovery_time': 0.9,
            'aim_smoothing': 0.15,
            'fire_confidence': 0.50,
            'headshot_priority': 1.0,
            'learning_rate': 0.05,
            'performance_window': 100
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    for line in f:
                        if '=' in line:
                            k, v = line.strip().split('=', 1)
                            if k in defaults:
                                # Convert to int for length parameters, float for others
                                if k.endswith('_len') or k.endswith('_commitment') or k.endswith('_window'):
                                    defaults[k] = int(float(v))
                                else:
                                    defaults[k] = float(v)
            except: pass
        else:
            with open(config_file, 'w') as f:
                for k, v in defaults.items():
                    f.write(f"{k}={v}\n")
        
        return defaults
    
    def capture(self):
        img = np.array(self.sct.grab(self.region))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def press(self, key):
        if key not in self.held_keys:
            self.kb.press(key)
            self.held_keys.add(key)
    
    def release(self, key):
        if key in self.held_keys:
            self.kb.release(key)
            self.held_keys.discard(key)
    
    def release_all(self):
        for k in list(self.held_keys):
            try: self.kb.release(k)
            except: pass
        self.held_keys.clear()
        self.velocity = 0
        if self.shooting:
            self.mouse.hold(False)
            self.shooting = False
    
    def tap(self, key):
        self.kb.press(key)
        time.sleep(0.03)
        self.kb.release(key)
    
    def find_gaps(self, depth, blocked, last_move_success=True):
        """
        Detect navigable gaps (corridors, doorways) with autonomous learning.
        last_move_success: True if previous chosen gap led to progress, False if stuck/collision
        """
        import numpy as np

        # Ensure all required tuning parameters exist
        self.params.setdefault('threshold_factor', 0.25)  # FIX: Lowered to detect more gaps
        self.params.setdefault('kernel_ratio', 0.08)  # FIX: Increased for smoother detection
        self.params.setdefault('strip_top_ratio', 0.33)
        self.params.setdefault('min_gap_ratio', 0.08)  # FIX: Lowered for narrow distant paths
        self.params.setdefault('learning_rate', 0.08)  # FIX: Increased

        # ---------------------------
        # Tuneable strips
        # ---------------------------
        top_ratio = self.params.get("strip_top_ratio", 0.33)
        bot_ratio = self.params.get("strip_bot_ratio", 0.67)
        strip_top, strip_bot = int(self.h * top_ratio), int(self.h * bot_ratio)

        depth_strip = depth[strip_top:strip_bot, :].mean(axis=0)
        block_strip = blocked[strip_top:strip_bot, :].mean(axis=0) if blocked is not None else np.zeros(self.w)

        # ---------------------------
        # Openness measure (FIX: Weight distant areas higher for future paths)
        # ---------------------------
        openness = (1 - depth_strip) * (1 - block_strip) * (1 + (1 - depth_strip) * 0.5)  # Boost low-depth (distant)

        # ---------------------------
        # Add far-strip blend for future paths
        # ---------------------------
        far_top, far_bot = int(self.h * 0.1), int(self.h * 0.3)  # FIX: New far horizon for distant detection
        far_depth = depth[far_top:far_bot, :].mean(axis=0)
        far_block = blocked[far_top:far_bot, :].mean(axis=0) if blocked is not None else np.zeros(self.w)
        far_openness = (1 - far_depth) * (1 - far_block)
        openness = 0.7 * openness + 0.3 * far_openness  # FIX: Blend mid + far

        # ---------------------------
        # Adaptive smoothing
        # ---------------------------
        kernel_ratio = self.params.get("kernel_ratio", 0.08)
        kernel_size = max(1, int(self.w * kernel_ratio))
        kernel = np.ones(kernel_size) / kernel_size
        openness = np.convolve(openness, kernel, mode='same')

        # ---------------------------
        # Adaptive threshold
        # ---------------------------
        threshold_factor = self.params.get("threshold_factor", 0.25)
        threshold = openness.mean() + openness.std() * threshold_factor

        # ---------------------------
        # Detect gaps
        # ---------------------------
        min_gap_ratio = self.params.get("min_gap_ratio", 0.08)
        gaps = []
        in_gap = False
        gap_start = 0

        for i, val in enumerate(openness):
            if val > threshold and not in_gap:
                gap_start = i
                in_gap = True
            elif val <= threshold and in_gap:
                gap_w = i - gap_start
                if gap_w >= self.w * min_gap_ratio:
                    center = (gap_start + i) / 2 / self.w
                    score = openness[gap_start:i].mean()
                    gaps.append({'center': center, 'width': gap_w / self.w, 'score': score})
                in_gap = False

        if in_gap and (self.w - gap_start) >= self.w * min_gap_ratio:
            center = (gap_start + self.w) / 2 / self.w
            gaps.append({'center': center, 'width': (self.w - gap_start) / self.w, 'score': openness[gap_start:].mean()})

        # ---------------------------
        # Sort gaps by score
        # ---------------------------
        gaps.sort(key=lambda g: g['score'], reverse=True)

        # ---------------------------
        # Autonomous learning (FIX: More aggressive on failure)
        # ---------------------------
        lr = self.params.get("learning_rate", 0.08)

        # If last chosen gap caused a problem, decrease sensitivity
        if not last_move_success:
            self.params['threshold_factor'] = min(1.0, self.params['threshold_factor'] * (1 + lr * 1.2))  # FIX: Stronger penalty
            self.params['kernel_ratio'] = min(0.2, self.params['kernel_ratio'] * (1 + lr))
            self.params['min_gap_ratio'] = max(0.05, self.params['min_gap_ratio'] * (1 - lr))
        else:  # if last move successful, increase sensitivity slightly
            self.params['threshold_factor'] = max(0.05, self.params['threshold_factor'] * (1 - lr * 0.5))
            self.params['kernel_ratio'] = max(0.01, self.params['kernel_ratio'] * (1 - lr))
            self.params['min_gap_ratio'] = min(0.3, self.params['min_gap_ratio'] * (1 + lr))

        # Clamp strip ratios
        self.params['strip_top_ratio'] = min(max(self.params.get('strip_top_ratio', 0.33), 0.1), 0.5)
        self.params['strip_bot_ratio'] = min(max(self.params.get('strip_bot_ratio', 0.67), 0.5), 0.9)

        # Periodically save updated parameters
        import time
        now = time.time()
        if now - getattr(self, 'last_param_save', 0) > 60:
            if hasattr(self, 'save_params'):
                self.save_params()
            self.last_param_save = now

        return gaps

    
    def analyze(self, frame):

        self.frame_count += 1

        # ---------------------------
        # Depth & perception
        # ---------------------------
        depth = self.perception.get_depth(frame)

        # Segmentation every seg_interval frames
        seg_interval = self.params.get('seg_interval', 3)
        if self.frame_count - getattr(self, 'last_seg_frame', -100) >= seg_interval:
            self.cached_blocked = self.perception.get_segmentation(frame)
            self.last_seg_frame = self.frame_count

        # Enemy detection every enemy_interval frames
        enemy_interval = self.params.get('enemy_interval', 2)
        if self.frame_count - getattr(self, 'last_enemy_frame', -100) >= enemy_interval:
            self.cached_enemies = self.perception.detect_enemies(frame)
            self.last_enemy_frame = self.frame_count

        blocked = self.cached_blocked if self.cached_blocked is not None else np.zeros((self.h, self.w))

        # ---------------------------
        # Gap detection (with own tuning parameters)
        # ---------------------------
        gap_width_limit = self.params.get('gap_width_limit', 0.35)
        gaps = self.find_gaps(depth, blocked, getattr(self, 'last_move_success', True))  # FIX: Pass feedback
        if gaps and gaps[0]['width'] < gap_width_limit:
            self.gap_history.append(gaps[0]['center'])
            self.in_corridor = True
            if len(self.gap_history) >= self.params.get('corridor_history_len', 3):
                self.corridor_center = sum(self.gap_history) / len(self.gap_history)
        else:
            self.in_corridor = False

        # ---------------------------
        # Zone analysis with own weights (FIX: Finer zones, add far)
        # ---------------------------
        zone_count = 15  # FIX: Increased to 15 for finer detail
        zone_w = self.w // zone_count
        top_ratio = self.params.get('zone_top_ratio', 0.33)
        bot_ratio = self.params.get('zone_bot_ratio', 0.67)
        mid_top, mid_bot = int(self.h * top_ratio), int(self.h * bot_ratio)
        near_top = int(self.h * self.params.get('zone_near_top_ratio', 0.45))
        far_top, far_bot = int(self.h * 0.1), int(self.h * 0.3)  # FIX: New far zone for future paths

        zones = []
        for i in range(zone_count):
            x1, x2 = i * zone_w, (i + 1) * zone_w
            d = depth[mid_top:mid_bot, x1:x2].mean()
            b = blocked[mid_top:mid_bot, x1:x2].mean()
            nd = depth[near_top:mid_bot, x1:x2].mean()
            nb = blocked[near_top:mid_bot, x1:x2].mean()
            fd = depth[far_top:far_bot, x1:x2].mean()  # FIX: Far depth
            fb = blocked[far_top:far_bot, x1:x2].mean()  # FIX: Far blocked
            combined_weight = self.params.get('combined_weight', 0.5)
            mid_combined = d + b * combined_weight
            far_combined = fd + fb * combined_weight
            combined = 0.6 * mid_combined + 0.4 * far_combined  # FIX: Blend mid + far
            zones.append({'combined': combined, 'near': nd + nb * combined_weight})

        # Aggregate zone scores
        left_indices = slice(0, zone_count // 3)
        center_indices = slice(zone_count // 3, 2 * zone_count // 3)
        right_indices = slice(2 * zone_count // 3, zone_count)

        left_raw = sum(z['combined'] for z in zones[left_indices]) / len(zones[left_indices])
        center_raw = sum(z['combined'] for z in zones[center_indices]) / len(zones[center_indices])
        right_raw = sum(z['combined'] for z in zones[right_indices]) / len(zones[right_indices])

        # Smooth with history
        self.score_history['left'].append(left_raw)
        self.score_history['center'].append(center_raw)
        self.score_history['right'].append(right_raw)

        left = np.mean(self.score_history['left'])
        center = np.mean(self.score_history['center'])
        right = np.mean(self.score_history['right'])

        # Near obstacles
        near_left = np.mean([zones[i]['near'] for i in range(zone_count // 3 - 2, zone_count // 3)])
        near_center = np.mean([zones[i]['near'] for i in range(zone_count // 3, 2 * zone_count // 3)])
        near_right = np.mean([zones[i]['near'] for i in range(2 * zone_count // 3, 2 * zone_count // 3 + 2)])

        # ---------------------------
        # Gradient & trend analysis (FIX: Longer trend)
        # ---------------------------
        gradient = left - right
        self.gradient_history.append(gradient)
        smooth_gradient = np.mean(self.gradient_history)

        trend_len = 10  # FIX: Increased to 10 for better prediction
        if len(self.gradient_history) >= 2 * trend_len:
            recent = list(self.gradient_history)[-trend_len:]
            older = list(self.gradient_history)[-2*trend_len:-trend_len]
            gradient_trend = np.mean(recent) - np.mean(older)
        else:
            gradient_trend = 0

        # ---------------------------
        # Adaptive obstacle thresholds
        # ---------------------------
        obs_threshold = self.params.get('obs_threshold', 0.4)  # FIX: Updated default
        emergency_threshold = self.params.get('emergency_threshold', 0.8)

        # ---------------------------
        # Autonomous parameter tuning for analyze
        # ---------------------------
        lr = self.params.get('learning_rate', 0.08)  # FIX: Updated

        # Example: if near_center frequently triggers emergency but agent navigates successfully, reduce threshold
        if near_center > emergency_threshold:
            if getattr(self, 'last_move_success', True):
                self.params['emergency_threshold'] = min(1.0, self.params['emergency_threshold'] + lr * 0.05)
            else:
                self.params['emergency_threshold'] = max(0.5, self.params['emergency_threshold'] - lr * 0.05)

        # Smooth obstacle threshold similarly
        if near_center < obs_threshold:
            self.params['obs_threshold'] = max(0.1, self.params['obs_threshold'] - lr * 0.02)
        else:
            self.params['obs_threshold'] = min(0.9, self.params['obs_threshold'] + lr * 0.02)

        # Periodically save
        now = time.time()
        if now - getattr(self, 'last_param_save', 0) > 60:
            if hasattr(self, 'save_params'):
                self.save_params()
            self.last_param_save = now

        # FIX: Track move success for feedback (openness increase)
        current_open = (left + center + right) / 3  # Simple avg openness
        self.last_move_success = current_open > getattr(self, 'prev_open', 0)
        self.prev_open = current_open

        return {
            'depth': depth,
            'blocked': blocked,
            'gaps': gaps,
            'left': left,
            'center': center,
            'right': right,
            'gradient': smooth_gradient,
            'gradient_trend': gradient_trend,
            'obstacle_left': near_left > obs_threshold,
            'obstacle_center': near_center > obs_threshold,
            'obstacle_right': near_right > obs_threshold,
            'emergency': near_center > emergency_threshold,
            'enemies': self.cached_enemies
        }
    
    def smooth_turn(self, target, urgency=0.5):
        """
        Adaptive velocity-based smooth turning with its own tuneable parameters.
        Parameters are stored in navigation_data/smooth_turn_params.txt
        """
        import os
        import time
        from pathlib import Path

        # ---------------------------
        # Parameter setup
        # ---------------------------
        if not hasattr(self, 'turn_data_dir'):
            self.turn_data_dir = Path("navigation_data")
            self.turn_data_dir.mkdir(exist_ok=True)
            self.turn_params_file = self.turn_data_dir / "smooth_turn_params.txt"

        # Default parameters
        default_turn_params = {
            "alpha_base": 0.2,          # base smoothing factor
            "sensitivity": 1.0,         # scaling for velocity to pixels
            "turn_speed": 5.0,           # pixels per unit velocity
            "min_velocity": 0.02,       # deadzone
            "velocity_decay": 0.4,      # damping for small velocity
            "urgency_scale": 1.0,       # scale alpha by urgency
            "learning_rate": 0.02       # how fast to adapt
        }

        # Load or initialize
        if not hasattr(self, 'turn_params'):
            self.turn_params = default_turn_params.copy()
            if self.turn_params_file.exists():
                try:
                    with open(self.turn_params_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line or "=" not in line:
                                continue
                            key, value = line.split("=", 1)
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                            self.turn_params[key] = value
                    print(f"Loaded smooth_turn parameters from {self.turn_params_file.name}")
                except Exception as e:
                    print(f"Failed to load smooth_turn params: {e}")

        # Save helper
        def save_turn_params():
            with open(self.turn_params_file, "w") as f:
                for k, v in self.turn_params.items():
                    f.write(f"{k}={v}\n")
        self.save_turn_params = save_turn_params

        # ---------------------------
        # Adaptive smoothing
        # ---------------------------
        alpha_base = self.turn_params['alpha_base']
        urgency_scale = self.turn_params.get('urgency_scale', 1.0)
        alpha = alpha_base * urgency * urgency_scale
        self.velocity = self.velocity * (1 - alpha) + target * alpha

        # Deadzone handling
        if abs(self.velocity) < self.turn_params['min_velocity']:
            self.velocity *= self.turn_params['velocity_decay']
            return

        # Pixel movement
        pixels = self.velocity * self.turn_params['sensitivity'] * self.turn_params['turn_speed']
        self.mouse.move(int(pixels), 0)

        # ---------------------------
        # Autonomous tuning
        # ---------------------------
        # Adjust sensitivity and turn_speed based on recent effectiveness
        if hasattr(self, 'last_move_success'):
            lr = self.turn_params.get('learning_rate', 0.02)
            if not self.last_move_success:
                # If turn was too weak, increase sensitivity
                self.turn_params['sensitivity'] = min(3.0, self.turn_params['sensitivity'] * (1 + lr))
                self.turn_params['turn_speed'] = min(15.0, self.turn_params['turn_speed'] * (1 + lr))
            else:
                # If turn overshoots, decrease sensitivity slightly
                self.turn_params['sensitivity'] = max(0.5, self.turn_params['sensitivity'] * (1 - lr/2))
                self.turn_params['turn_speed'] = max(3.0, self.turn_params['turn_speed'] * (1 - lr/2))

        # Periodically save parameters every 60 seconds
        now = time.time()
        if not hasattr(self, 'last_turn_param_save'):
            self.last_turn_param_save = now
        if now - self.last_turn_param_save > 60:
            self.save_turn_params()
            self.last_turn_param_save = now
    
    def combat(self, enemies):
        now = time.time()

        # ---------------------------
        # Initialize in-memory combat params (safe defaults; no .txt modifications)
        # ---------------------------
        if not hasattr(self, "combat_params"):
            self.combat_params = {}

        cp = self.combat_params

        # DEFAULTS (only set if missing)
        cp.setdefault("fire_confidence", 0.7)
        cp.setdefault("conf_min", 0.35)
        cp.setdefault("conf_max", 0.95)

        # Sloppiness-related defaults
        cp.setdefault("initial_sloppy", 1)
        cp.setdefault("sloppy_fire_tolerance", 24)
        cp.setdefault("sloppy_fire_tolerance_current", cp["sloppy_fire_tolerance"])
        cp.setdefault("sloppy_decay_rate", 0.015)
        cp.setdefault("sloppy_min_tolerance", 4)

        # Probabilistic early-fire
        cp.setdefault("fire_while_moving_prob", 0.35)
        cp.setdefault("fire_prob_decay", 0.01)

        # Aim aggression (start fast/sloppy -> decay toward final)
        cp.setdefault("initial_aim_smoothing", 0.6)
        cp.setdefault("current_aim_smoothing", cp["initial_aim_smoothing"])
        cp.setdefault("initial_aim_speed", 1.8)
        cp.setdefault("current_aim_speed", cp["initial_aim_speed"])
        cp.setdefault("final_aim_smoothing", 0.18)
        cp.setdefault("aim_aggression_decay", 0.012)

        # Vertical control to prevent aiming at floor
        cp.setdefault("vertical_gain", 0.6)        # scale vertical movement relative to horizontal
        cp.setdefault("max_vertical_move", 40)     # clamp max vertical pixels per frame

        # Body/head aiming offsets (fractions of box height)
        cp.setdefault("body_aim_fraction", 0.35)   # aim at upper chest (~35% from top of box)
        cp.setdefault("head_aim_fraction", 0.50)   # head center fallback (use cy usually)

        # Existing combat defaults (safety)
        cp.setdefault("w_head", 2.5)
        cp.setdefault("w_conf", 1.0)
        cp.setdefault("w_center", 1.2)
        cp.setdefault("w_size", 0.7)
        cp.setdefault("aim_smoothing", cp["current_aim_smoothing"])
        cp.setdefault("aim_speed", cp["current_aim_speed"])
        cp.setdefault("burst_mode", 1)
        cp.setdefault("burst_count", 3)
        cp.setdefault("burst_interval", 0.08)
        cp.setdefault("hold_min_time", 0.05)
        cp.setdefault("adapt_lr", 0.03)
        cp.setdefault("smoothing_alpha", 0.15)
        cp.setdefault("micro_jitter", 0.0)
        cp.setdefault("recoil_comp", 0.0)
        cp.setdefault("save_interval", 60.0)

        # ---------------------------
        # No enemies -> stop shooting and exit
        # ---------------------------
        if not enemies:
            if getattr(self, "shooting", False):
                try:
                    self.mouse.hold(False)
                except Exception:
                    pass
                self.shooting = False
            return False

        # update engagement metric
        self.perf_metrics['enemies_engaged'] = self.perf_metrics.get('enemies_engaged', 0) + 1

        # ---------------------------
        # Candidate selection: score targets
        # ---------------------------
        cx, cy = self.w / 2.0, self.h / 2.0

        def score_target(e):
            conf = float(e.get('conf', 0.0))
            head_bonus = float(cp.get("w_head", 2.5)) if e.get('is_head', False) else 0.0
            dx = (e['cx'] - cx) / max(1.0, cx)
            dy = (e['cy'] - cy) / max(1.0, cy)
            dist_norm = math.hypot(dx, dy)
            size = math.sqrt(max(1.0, e.get('w', 1.0) * e.get('h', 1.0)))
            score = (head_bonus * 10.0) + float(cp.get("w_conf", 1.0)) * conf \
                    + float(cp.get("w_size", 0.7)) * (size / max(1.0, max(self.w, self.h))) \
                    + float(cp.get("w_center", 1.2)) * (1.0 - max(0.0, 1.0 - dist_norm))
            return score

        # filter by confidence threshold
        fc = float(cp.get("fire_confidence", 0.7))
        candidates = [e for e in enemies if float(e.get('conf', 0.0)) >= fc]

        # if none pass confidence, optionally lower a bit in-memory (non-persistent)
        if not candidates:
            misses = float(self.perf_metrics.get('misses', 0))
            hits = float(self.perf_metrics.get('hits', 0))
            total = max(1.0, hits + misses)
            miss_rate = misses / total
            if miss_rate > 0.6 and fc > float(cp.get("conf_min", 0.35)):
                cp["fire_confidence"] = max(float(cp.get("conf_min", 0.35)), fc * (1.0 - cp.get("adapt_lr", 0.03)))
                fc = cp["fire_confidence"]
            if not candidates:
                if getattr(self, "shooting", False):
                    try:
                        self.mouse.hold(False)
                    except Exception:
                        pass
                    self.shooting = False
                return False

        # choose best candidate
        candidates.sort(key=lambda x: score_target(x), reverse=True)
        best = candidates[0]

        # ---------------------------
        # Compute pixel offsets and adaptive aim params
        # ---------------------------
        target_x = best['cx']
        if best.get('is_head', False):
            # prefer using detection's cy (head center) but allow small shift if needed
            target_y = best.get('cy', best.get('y1', 0) + best.get('h', 0) * cp.get("head_aim_fraction", 0.5))
        else:
            # aim at upper chest/neck area (not the bottom of the box) to avoid looking at floor
            body_frac = float(cp.get("body_aim_fraction", 0.35))
            target_y = best.get('y1', 0) + best.get('h', 0) * body_frac

        dx_pix = target_x - cx
        dy_pix = target_y - cy

        # recoil compensation (upwards)
        recoil = float(cp.get("recoil_comp", 0.0))
        if getattr(self, "recent_shot_time", 0) and now - getattr(self, "recent_shot_time", 0) < 0.25:
            dy_pix -= recoil

        # micro jitter (optional)
        jitter = float(cp.get("micro_jitter", 0.0))
        if jitter > 0:
            dx_pix += (random.random() - 0.5) * jitter
            dy_pix += (random.random() - 0.5) * jitter

        # use current aggressive->precise decaying params
        aim_smooth = float(cp.get("current_aim_smoothing", cp.get("initial_aim_smoothing", 0.6)))
        aim_speed = float(cp.get("current_aim_speed", cp.get("initial_aim_speed", 1.8)))
        aim_smooth = max(0.01, min(0.95, aim_smooth))
        aim_speed = max(0.1, min(5.0, aim_speed))

        # horizontal movement
        move_x = int(dx_pix * aim_smooth * aim_speed)

        # vertical movement scaled down to avoid looking at floor, then clamped
        vertical_gain = float(cp.get("vertical_gain", 0.6))
        max_vert = float(cp.get("max_vertical_move", 40.0))
        raw_move_y = dy_pix * aim_smooth * aim_speed * vertical_gain
        # clamp raw vertical move
        if raw_move_y > 0:
            move_y = int(min(max_vert, raw_move_y))
        else:
            move_y = int(max(-max_vert, raw_move_y))

        # tiny fallback nudges
        if move_x == 0 and abs(dx_pix) >= 1.0:
            move_x = 1 if dx_pix > 0 else -1
        if move_y == 0 and abs(dy_pix) >= 1.0:
            # nudge smaller vertically to avoid big downward jumps
            move_y = 1 if dy_pix > 0 else -1
            move_y = int(move_y * vertical_gain)

        # apply mouse move (safe)
        try:
            self.mouse.move(move_x, move_y)
        except Exception:
            pass

        # ---------------------------
        # Sloppy tolerance -> allow_fire logic
        # ---------------------------
        tol = float(cp.get("sloppy_fire_tolerance_current", cp.get("sloppy_fire_tolerance", 24)))
        on_target = (abs(dx_pix) <= tol and abs(dy_pix) <= tol)

        allow_fire = False
        if on_target:
            allow_fire = True
        else:
            if int(cp.get("initial_sloppy", 1)):
                p = float(cp.get("fire_while_moving_prob", 0.35))
                if random.random() < p:
                    allow_fire = True

        # ---------------------------
        # Firing decision (burst or hold)
        # ---------------------------
        if int(cp.get("burst_mode", 1)):
            if not hasattr(self, "_burst_remain") or self._burst_remain <= 0:
                if allow_fire:
                    self._burst_remain = int(max(1, cp.get("burst_count", 3)))
                    self._last_burst_shot = 0.0
            if getattr(self, "_burst_remain", 0) > 0:
                last_shot = getattr(self, "_last_burst_shot", 0.0)
                if now - last_shot >= float(cp.get("burst_interval", 0.08)):
                    try:
                        if hasattr(self.mouse, "tap"):
                            self.mouse.tap()
                        else:
                            self.mouse.hold(True)
                            time.sleep(0.01)
                            self.mouse.hold(False)
                    except Exception:
                        try:
                            self.mouse.hold(True)
                            time.sleep(0.01)
                            self.mouse.hold(False)
                        except Exception:
                            pass
                    self._burst_remain -= 1
                    self._last_burst_shot = now
                    self.recent_shot_time = now
                    self.perf_metrics['shots_fired'] = self.perf_metrics.get('shots_fired', 0) + 1
                    if on_target:
                        self.perf_metrics['time_on_target'] = self.perf_metrics.get('time_on_target', 0) + 1
                    self.shooting = True
            else:
                if getattr(self, "shooting", False):
                    try:
                        self.mouse.hold(False)
                    except Exception:
                        pass
                    self.shooting = False
        else:
            if allow_fire:
                if not getattr(self, "shooting", False):
                    try:
                        self.mouse.hold(True)
                    except Exception:
                        pass
                    self.shooting = True
                    self.perf_metrics['shots_fired'] = self.perf_metrics.get('shots_fired', 0) + 1
                    self.recent_shot_time = now
            else:
                if getattr(self, "shooting", False):
                    last = getattr(self, "recent_shot_time", 0.0)
                    if now - last >= float(cp.get("hold_min_time", 0.05)):
                        try:
                            self.mouse.hold(False)
                        except Exception:
                            pass
                        self.shooting = False

        # ---------------------------
        # Learning: decay sloppy params on success (on_target considered a success proxy)
        # ---------------------------
        if on_target:
            if int(cp.get("initial_sloppy", 1)):
                cur_tol = float(cp.get("sloppy_fire_tolerance_current", cp.get("sloppy_fire_tolerance", 24)))
                decay = float(cp.get("sloppy_decay_rate", 0.015))
                min_tol = float(cp.get("sloppy_min_tolerance", 4))
                new_tol = max(min_tol, cur_tol * (1.0 - decay))
                cp["sloppy_fire_tolerance_current"] = new_tol

                p = float(cp.get("fire_while_moving_prob", 0.35))
                p = max(0.0, p - float(cp.get("fire_prob_decay", 0.01)))
                cp["fire_while_moving_prob"] = p

                cur_smooth = float(cp.get("current_aim_smoothing", cp.get("initial_aim_smoothing", 0.6)))
                target_smooth = float(cp.get("final_aim_smoothing", 0.18))
                aggr_decay = float(cp.get("aim_aggression_decay", 0.012))
                cur_smooth = max(target_smooth, cur_smooth * (1.0 - aggr_decay))
                cp["current_aim_smoothing"] = cur_smooth
                cp["aim_smoothing"] = cur_smooth

                cur_speed = float(cp.get("current_aim_speed", cp.get("initial_aim_speed", 1.8)))
                cur_speed = max(float(cp.get("aim_speed", 1.0)), cur_speed * (1.0 - aggr_decay * 0.6))
                cp["current_aim_speed"] = cur_speed
                cp["aim_speed"] = cur_speed

            self.perf_metrics['time_on_target'] = self.perf_metrics.get('time_on_target', 0) + 1
            self.perf_metrics['hits'] = self.perf_metrics.get('hits', 0) + 1
        else:
            self.perf_metrics['misses'] = self.perf_metrics.get('misses', 0) + 1

        # ---------------------------
        # Gentle adaptive adjustments to fire_confidence based on EMA of on-target ratio
        # ---------------------------
        if "aim_on_target_ema" not in self.perf_metrics:
            self.perf_metrics["aim_on_target_ema"] = 1.0 if on_target else 0.0
        else:
            alpha = float(cp.get("smoothing_alpha", 0.15))
            self.perf_metrics["aim_on_target_ema"] = alpha * (1.0 if on_target else 0.0) + (1 - alpha) * self.perf_metrics["aim_on_target_ema"]

        ema = float(self.perf_metrics.get("aim_on_target_ema", 0.0))
        lr = float(cp.get("adapt_lr", 0.03))
        hits = float(self.perf_metrics.get('time_on_target', 0))
        shots = float(self.perf_metrics.get('shots_fired', 1))
        hit_rate = hits / max(1.0, shots)

        if ema < 0.2 and cp.get("fire_confidence", 0.7) > cp.get("conf_min", 0.35):
            cp["fire_confidence"] = max(cp.get("conf_min", 0.35), cp["fire_confidence"] * (1.0 - lr))
        elif ema > 0.6 and hit_rate > 0.4:
            cp["fire_confidence"] = min(cp.get("conf_max", 0.95), cp["fire_confidence"] * (1.0 + lr * 0.5))

        # clamp important parameters to safe ranges
        cp["sloppy_fire_tolerance_current"] = max(float(cp.get("sloppy_min_tolerance", 4)), min(200.0, float(cp.get("sloppy_fire_tolerance_current", 24))))
        cp["fire_confidence"] = max(float(cp.get("conf_min", 0.35)), min(float(cp.get("conf_max", 0.95)), float(cp.get("fire_confidence"))))
        cp["current_aim_smoothing"] = max(0.01, min(0.95, float(cp.get("current_aim_smoothing", cp.get("final_aim_smoothing", 0.18)))))
        cp["current_aim_speed"] = max(0.1, min(5.0, float(cp.get("current_aim_speed", 1.0))))

        cp["last_updated"] = now

        return True

    
    def adapt_parameters(self):
        """Adaptive learning: tune parameters based on performance"""
        now = time.time()
        
        # Update every 5 seconds
        if now - self.last_perf_update < 5.0:
            return
        
        self.last_perf_update = now
        metrics = self.perf_metrics
        
        if metrics['total_frames'] < 50:  # Need minimum data
            return
        
        lr = self.params['learning_rate']
        
        # Adapt aim smoothing based on time on target
        if metrics['enemies_engaged'] > 0:
            hit_rate = metrics['time_on_target'] / max(metrics['shots_fired'], 1)
            
            # If hitting well, can be more aggressive (lower smoothing)
            if hit_rate > 0.7:
                self.params['aim_smoothing'] = max(0.10, self.params['aim_smoothing'] - lr * 0.02)
            # If missing, need more smoothing
            elif hit_rate < 0.4:
                self.params['aim_smoothing'] = min(0.25, self.params['aim_smoothing'] + lr * 0.03)
        
        # Adapt stuck recovery based on stuck events
        stuck_rate = metrics['stuck_events'] / max(metrics['total_frames'], 1)
        if stuck_rate > 0.01:  # Getting stuck too often
            self.params['recovery_time'] = min(1.5, self.params['recovery_time'] + lr * 0.1)
            self.params['turn_commitment'] = max(12, int(self.params['turn_commitment'] - lr * 20))
        elif stuck_rate < 0.001:  # Moving smoothly, can commit more
            self.params['turn_commitment'] = min(25, int(self.params['turn_commitment'] + lr * 10))
        
        # Adapt sensitivity based on combat engagement
        if metrics['enemies_engaged'] > 10:
            engagement_rate = metrics['time_on_target'] / metrics['enemies_engaged']
            if engagement_rate < 0.3:  # Not tracking well
                self.params['sensitivity'] = min(28, self.params['sensitivity'] + lr * 5)
            elif engagement_rate > 0.6:  # Tracking well
                self.params['sensitivity'] = max(18, self.params['sensitivity'] - lr * 3)
        
        # Save adapted parameters every 30 seconds
        if now - self.last_param_save > 30.0:
            self.last_param_save = now
            self._save_params()
            print(f"\n📊 Adapted params: aim={self.params['aim_smoothing']:.3f}, "
                  f"sens={self.params['sensitivity']:.1f}, "
                  f"commit={self.params['turn_commitment']}")
    
    def _save_params(self):
        """Save current parameters to file"""
        config_file = self.config_dir / "params.txt"
        try:
            with open(config_file, 'w') as f:
                for k, v in self.params.items():
                    f.write(f"{k}={v}\n")
        except:
            pass
    
    def navigate(self, analysis):
        """Main navigation logic with adaptive learning and tunable parameters"""
        now = time.time()

        # ---------------------------
        # COMBAT PRIORITY - engage enemies first
        # ---------------------------
        if self.combat(analysis['enemies']):
            self.press('w')
            self.release('a')
            self.release('d')
            return

        # ---------------------------
        # Stuck detection (FIX: Higher threshold, use depth var)
        # ---------------------------
        self.stuck_history.append(analysis['center'])
        if len(self.stuck_history) >= 35:
            var = np.var(list(self.stuck_history)[-35:])
            if var < self.params['stuck_threshold'] and analysis['obstacle_center']:
                self.stuck_counter += 2
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        # ---------------------------
        # Recovery mode (FIX: Extended time, add random strafe)
        # ---------------------------
        if self.stuck_counter > 18 or self.recovery_mode:
            if not self.recovery_mode:
                self.recovery_mode = True
                self.recovery_start = now
                self.committed_dir = 'left' if analysis['left'] < analysis['right'] else 'right'

            if now - self.recovery_start < self.params['recovery_time']:
                self.release('w')
                self.press('s')
                turn = -1.0 if self.committed_dir == 'left' else 1.0
                self.smooth_turn(turn, 1.0)
                strafe_dir = random.choice(['a', 'd'])  # FIX: Random strafe for better unstuck
                self.press(strafe_dir)
                self.release('d' if strafe_dir == 'a' else 'a')
                if random.random() < 0.15:
                    self.tap(Key.space)
                return
            else:
                self.recovery_mode = False
                self.stuck_counter = 0
                self.release('s')
                self.release('a')
                self.release('d')
                self.velocity = 0

        # ---------------------------
        # Emergency handling
        # ---------------------------
        if analysis['emergency']:
            if self.turn_commitment <= 0:
                if analysis['gaps']:
                    self.committed_dir = 'right' if analysis['gaps'][0]['center'] > 0.5 else 'left'
                else:
                    self.committed_dir = 'left' if analysis['left'] < analysis['right'] else 'right'
                self.turn_commitment = int(self.params['turn_commitment'])

            self.turn_commitment -= 1
            turn = -1.0 if self.committed_dir == 'left' else 1.0
            self.smooth_turn(turn, 1.0)
            self.release('w')
            time.sleep(0.005)
            self.press('w')

            if now - self.last_jump > 0.3:
                self.tap(Key.space)
                self.last_jump = now
            return

        # ---------------------------
        # Normal navigation
        # ---------------------------
        self.press('w')
        target = 0.0
        urgency = 0.5

        # Corridor mode
        if self.in_corridor and analysis['gaps']:
            offset = self.corridor_center - 0.5
            target = offset * 3.5
            target = max(-0.75, min(0.75, target))
            urgency = 0.7
            self.turn_commitment = 0

            # Adaptive corridor learning
            lr = self.params.get('learning_rate', 0.08)
            self.corridor_center = (1 - lr) * self.corridor_center + lr * analysis['gaps'][0]['center']

        # Obstacle ahead
        elif analysis['obstacle_center']:
            if self.turn_commitment <= 0:
                if analysis['gaps']:
                    self.committed_dir = 'left' if analysis['gaps'][0]['center'] < 0.5 else 'right'
                else:
                    self.committed_dir = 'left' if analysis['left'] < analysis['right'] else 'right'
                self.turn_commitment = int(self.params['turn_commitment'])

            self.turn_commitment -= 1
            target = -0.95 if self.committed_dir == 'left' else 0.95
            urgency = 0.85

            # Adaptive learning: obstacle avoidance
            lr = self.params.get('learning_rate', 0.08)
            self.params['obs_threshold'] = min(0.9, self.params.get('obs_threshold', 0.5) + lr * 0.02)

        # Side obstacles
        elif analysis['obstacle_left'] and not analysis['obstacle_right']:
            target = 0.6
            urgency = 0.7
            self.turn_commitment = 0
        elif analysis['obstacle_right'] and not analysis['obstacle_left']:
            target = -0.6
            urgency = 0.7
            self.turn_commitment = 0

        # Predictive steering using gradient and trend
        elif abs(analysis['gradient']) > 0.01:
            target = analysis['gradient'] * 2.5 + analysis['gradient_trend'] * 10.0
            target = max(-0.5, min(0.5, target))
            urgency = 0.5
            self.turn_commitment = 0

            # Adaptive gradient learning
            lr = self.params.get('learning_rate', 0.08)
            self.params['gradient_factor'] = (1 - lr) * self.params.get('gradient_factor', 1.0) + lr * analysis['gradient']

        # ---------------------------
        # Apply turning
        # ---------------------------
        self.smooth_turn(target, urgency)

        # ---------------------------
        # Strafe assist
        # ---------------------------
        if target < -0.25:
            self.press('a')
            self.release('d')
        elif target > 0.25:
            self.press('d')
            self.release('a')
        else:
            self.release('a')
            self.release('d')

        # ---------------------------
        # Jumping for obstacles or random variation
        # ---------------------------
        if analysis['obstacle_center'] and now - self.last_jump > 0.35:
            self.tap(Key.space)
            self.last_jump = now
        elif now - self.last_jump > 0.9 and random.random() < 0.1:
            self.tap(Key.space)
            self.last_jump = now


    
    def draw_debug(self, frame, analysis):
        """Debug visualization"""
        depth_color = cv2.applyColorMap((analysis['depth'] * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        output = cv2.addWeighted(frame, 0.4, depth_color, 0.6, 0)
        
        # Enemy boxes
        for enemy in analysis['enemies']:
            color = (0, 0, 255) if enemy['is_head'] else (0, 165, 255)
            cv2.rectangle(output, (enemy['x1'], enemy['y1']), (enemy['x2'], enemy['y2']), color, 2)
            label = f"{enemy['class']} {enemy['conf']:.2f}"
            cv2.putText(output, label, (enemy['x1'], enemy['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Gaps (FIX: Thicker lines for visibility)
        for i, gap in enumerate(analysis['gaps'][:3]):
            x = int(gap['center'] * self.w)
            color = (0, 255, 0) if i == 0 else (0, 200, 200)
            cv2.line(output, (x, self.h//3), (x, 2*self.h//3), color, 3)  # FIX: Thicker

        # Crosshair
        cv2.drawMarker(output, (self.w//2, self.h//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        # Status
        info = [
            f"Vel: {self.velocity:+.2f}",
            f"Grad: {analysis['gradient']:+.3f}",
            f"Enemies: {len(analysis['enemies'])}"
        ]
        if self.in_corridor:
            info.append("CORRIDOR")
        if self.recovery_mode:
            info.append("RECOVERY")
        if self.shooting:
            info.append("🔴 FIRING")
        
        for i, txt in enumerate(info):
            cv2.putText(output, txt, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        
        return output

# ============================================================================
# MAIN BOT
# ============================================================================

class CS2Bot:
    def __init__(self, show_debug=True):
        print("=" * 60)
        print("  CS2 SMART NAVIGATION + COMBAT BOT")
        print("  AI-Powered Pathfinding | Precision Targeting")
        print("=" * 60)
        print()
        
        self.nav = Navigator()
        self.show_debug = show_debug
        self.running = True
        self.active = True
        
        if show_debug:
            cv2.namedWindow("CS2 Nav Bot", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("CS2 Nav Bot", 640, 360)
            #cv2.imshow("CS2 Nav Bot", np.zeros((360, 640, 3), dtype=np.uint8))
            cv2.waitKey(1)
    
    def on_key(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                c = key.char.lower()
                if c == 'p':
                    if not self.active:
                        print("\n🎮 Starting in 3 seconds...")
                        time.sleep(3)
                        self.active = True
                        print("🟢 ACTIVE\n")
                    else:
                        self.active = False
                        self.nav.release_all()
                        print("\n⏸️  PAUSED\n")
                elif c == 'q':
                    self.nav.release_all()
                    self.running = False
                    return False
        except: pass
    
    def run(self):
        print("Controls: P = Start/Pause, Q = Quit")
        print("Ready!\n")
        
        listener = KBListener(on_press=self.on_key)
        listener.start()
        
        frame_times = deque(maxlen=30)
        
        try:
            while self.running:
                if not self.active:
                    time.sleep(0.05)
                    if self.show_debug:
                        cv2.waitKey(1)
                    continue
                
                t0 = time.time()
                
                frame = self.nav.capture()
                analysis = self.nav.analyze(frame)
                self.nav.navigate(analysis)
                
                dt = time.time() - t0
                frame_times.append(dt)
                fps = 1.0 / (sum(frame_times) / len(frame_times))
                
                if self.show_debug:
                    dbg = self.nav.draw_debug(frame, analysis)
                    cv2.putText(dbg, f"FPS: {fps:.1f}", (10, dbg.shape[0] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("CS2 Nav Bot", dbg)
                    cv2.waitKey(1)
        
        except KeyboardInterrupt:
            pass
        finally:
            self.nav.release_all()
            self.nav.mouse.close()
            #cv2.destroyAllWindows()
            listener.stop()
            print("\nBot stopped.")

if __name__ == "__main__":
    import argparse
    
    print("\n" + "=" * 60)
    print("  CS2 WALKBOT by Daniel Marcu")
    print("  Ubuntu Edition")
    print("=" * 60)
    print(f"  OS: {platform.system()} {platform.release()}")
    print("=" * 60 + "\n")
    
    # Check permissions
    if not os.access('/dev/uinput', os.W_OK):
        print("⚠ WARNING: No write access to /dev/uinput")
        print("  Run these commands:")
        print("  1. sudo usermod -aG input $USER")
        print("  2. Logout and login again")
        print("  3. Verify with: groups | grep input\n")
    
    parser = argparse.ArgumentParser(description="CS2 Navigation + Combat Bot (Ubuntu)")
    parser.add_argument('--no-debug', action='store_true', help='Hide debug window')
    args = parser.parse_args()
    
    bot = CS2Bot(show_debug=not args.no_debug)
    bot.run()