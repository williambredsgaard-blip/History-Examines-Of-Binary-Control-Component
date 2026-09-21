#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import time
import os
import signal

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
LOOP = ROOT / "supervisor_check_loop.py"

IS_WINDOWS = sys.platform.startswith("win")

TORCH_PACKAGES = [
    "torch",
    "torchvision",
]

PIP_PACKAGES = [
    "numpy",
    "pillow",
    "opencv-python",
    "easyocr",
    "pyautogui",
    "mss",
    "pytesseract",
    "transformers",
    "ultralytics",
    "pynput",
]

SYSTEM_DEPS = [
    "build-essential",
    f"python{sys.version_info.major}.{sys.version_info.minor}-dev",
    "pkg-config",
    "libudev-dev",
    "gnome-screenshot",
    "xdotool", 
    "python3-tk", 
    "ffmpeg", 
    "mpg123", 
]

process = None

def run(cmd, capture=False, check=False):
    return subprocess.run(cmd, capture_output=capture, text=capture, check=check)

def cleanup(sig=None, frame=None):
    print("\n🛑 Shutting down...")
    if process and process.poll() is None:
        process.terminate()
        process.wait()
    sys.exit(0)

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     ██████╗███████╗██████╗     ██████╗  ██████╗ ████████╗║
║    ██╔════╝██╔════╝╚════██╗    ██╔══██╗██╔═══██╗╚══██╔══╝║
║    ██║     ███████╗ █████╔╝    ██████╔╝██║   ██║   ██║   ║
║    ██║     ╚════██║██╔═══╝     ██╔══██╗██║   ██║   ██║   ║
║    ╚██████╗███████║███████╗    ██████╔╝╚██████╔╝   ██║   ║
║     ╚═════╝╚══════╝╚══════╝    ╚═════╝  ╚═════╝    ╚═╝   ║
║                                                          ║
║                       by Daniel Marcu                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

def install_system_dependency(package_name):
    if IS_WINDOWS:
        return True

    print(f"\n🚨 Attempting to install required system package: {package_name}")
    print("This requires administrative privileges (sudo password).")
    try:
        subprocess.run(['apt', 'list', package_name], check=True, capture_output=True, text=True) 
        subprocess.call(['sudo', 'apt', 'install', '-y', package_name])
        print(f"System package {package_name} installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print(f"ERROR: System package {package_name} not found in apt repository.")
        return False
    except FileNotFoundError:
        print("ERROR: 'sudo' or 'apt' command not found. Cannot install system package.")
        return False
    except Exception as e:
        print(f"ERROR: Failed to install system package {package_name}. {e}")
        return False

def install_prerequisite_system_deps():
    if IS_WINDOWS:
        print("Running on Windows: Skipping Linux system package installation (dpkg/apt).")
        return True

    print("Checking for required prerequisite system packages...")
    all_ok = True
    for pkg in SYSTEM_DEPS:
        print(f" • {pkg} ... ", end="", flush=True)
        try:
            subprocess.run(['dpkg', '-s', pkg], check=True, capture_output=True, text=True)
            print("already installed")
        except subprocess.CalledProcessError:
            print("missing, installing...")
            if not install_system_dependency(pkg):
                all_ok = False
                print(f"WARNING: Could not install required system package {pkg}. Proceeding, but Python package installation may fail.")
    return all_ok

def check_xserver_access():
    if IS_WINDOWS:
        print("Running on Windows: Skipping X Server check.")
        return True

    print("\n🌐 Checking X Server Access for Screen Capture...")
    
    if 'DISPLAY' not in os.environ:
        print("⚠️ WARNING: $DISPLAY environment variable is NOT set. Screen capture (mss) will likely fail.")
        print("   If you are on SSH, you need to set up X forwarding (e.g., use 'ssh -X').")
        print("   Skipping 'xhost +' check as it cannot run without a display.")
        return False

    try:
        run(['which', 'xhost'], check=True, capture=True)
    except subprocess.CalledProcessError:
        print("❌ ERROR: The 'xhost' command is not available. Install it (e.g., 'sudo apt install x11-xserver-utils').")
        return False

    print(" • Executing 'xhost +' to grant general X access... ", end="", flush=True)
    try:
        result = run(['xhost', '+'], check=True, capture=True)
        if result.returncode == 0:
            print("SUCCESS")
            return True
        else:
            print("FAIL")
            print(f"   Details: {result.stderr or result.stdout}")
            print("   Please run 'xhost +' manually in the terminal before starting the script.")
            return False
            
    except subprocess.CalledProcessError as e:
        print("FAIL (Command Error)")
        print(f"   Details: {e.stderr or e.stdout}")
        print("   Please run 'xhost +' manually in the terminal before starting the script.")
        return False
    except Exception as e:
        print(f"FAIL (Unexpected Error: {e})")
        print("   Please run 'xhost +' manually in the terminal before starting the script.")
        return False

def check_uinput_access():
    if IS_WINDOWS:
        print("Running on Windows: Skipping /dev/uinput check.")
        return True

    print("\n⌨️ Checking UInput Access for Input Simulation...")
    uinput_path = Path("/dev/uinput")

    if not uinput_path.exists():
        print("❌ ERROR: The device file /dev/uinput does not exist.")
        print("   Attempting to load the 'uinput' kernel module (requires sudo).")
        try:
            subprocess.call(['sudo', 'modprobe', 'uinput'])
            if not uinput_path.exists():
                print("   Module load failed or module is unavailable. Please check kernel settings.")
                return False
        except FileNotFoundError:
            print("   ERROR: 'sudo' or 'modprobe' command not found. Cannot load module.")
            return False

    if uinput_path.exists():
        try:
            if os.access(uinput_path, os.W_OK):
                print(" • User has write permissions. SUCCESS")
                return True
            else:
                print("⚠️ WARNING: User lacks write permissions for /dev/uinput.")
                print("   Attempting to apply temporary fix: 'sudo chmod a+rw /dev/uinput'")
                print("   (This fix is temporary and requires your sudo password.)")
                
                result = subprocess.run(['sudo', 'chmod', 'a+rw', str(uinput_path)], check=False, capture_output=True, text=True)
                
                if result.returncode == 0:
                    if os.access(uinput_path, os.W_OK):
                        print("   Temporary fix applied. SUCCESS (Permissions will reset on reboot).")
                        return True
                    else:
                        print("   Failed to gain access even after 'chmod'.")
                        print(f"   Details: {result.stderr.strip()}")
                        return False
                else:
                    print("   'sudo chmod' failed. This requires your sudo password.")
                    print(f"   Details: {result.stderr.strip()}")
                    return False

        except Exception as e:
            print(f"FAIL (Unexpected Error during permission check: {e})")
            return False
    
    return False

def create_venv():
    if VENV.exists():
        print("Using existing virtualenv")
    else:
        print("Creating virtualenv...", end=" ", flush=True)
        try:
            run([sys.executable, "-m", "venv", str(VENV)], check=True, capture=True) 
            print("done")
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip() if isinstance(e.stderr, str) else ""
            stdout_output = e.stdout.strip() if isinstance(e.stdout, str) else ""

            is_ensurepip_error = ("ensurepip" in stderr_output and "available" in stderr_output) or \
                                 ("ensurepip" in stdout_output and "available" in stdout_output)

            if is_ensurepip_error and not IS_WINDOWS:
                python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
                required_package = f"{python_version}-venv"
                
                print("fail (Missing OS Dependency)")
                if install_system_dependency(required_package):
                    print("Retrying virtualenv creation...", end=" ", flush=True)
                    try:
                        run([sys.executable, "-m", "venv", str(VENV)], check=True, capture=True)
                        print("done")
                    except subprocess.CalledProcessError:
                        print("fail")
                        print("FATAL ERROR: Virtual environment creation failed even after installing the OS package. Aborting.")
                        sys.exit(1)
                else:
                    print("FATAL ERROR: Cannot automatically resolve OS dependency. Aborting.")
                    sys.exit(1)
            else:
                print("fail")
                print(f"FATAL ERROR: Virtual environment creation failed with an unknown error.")
                print(f"Details (Stderr): {stderr_output or 'N/A'}")
                print(f"Details (Stdout): {stdout_output or 'N/A'}")
                sys.exit(1)

    if IS_WINDOWS:
        vpy = VENV / "Scripts" / "python.exe"
    else:
        vpy = VENV / "bin" / "python"

    if not vpy.exists():
        print("ERROR: venv python not found. Aborting.")
        sys.exit(1)

    run([str(vpy), "-m", "ensurepip", "--upgrade"], check=True)
    run([str(vpy), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    return str(vpy)

def install_packages(vpy):
    print("Installing PyTorch packages...")
    torch_cmd = [vpy, "-m", "pip", "install"] + TORCH_PACKAGES + ["--index-url", "https://download.pytorch.org/whl/cpu"]
    result = run(torch_cmd, capture=True)
    if result.returncode == 0:
        print(" • torch & torchvision ... ok")
    else:
        print(" • torch & torchvision ... fail")
        err = (result.stderr or result.stdout or "").splitlines()
        if err:
            print("   " + err[-1].strip())

    print("Installing remaining Python packages...")
    for pkg in PIP_PACKAGES:
        print(f" • {pkg} ... ", end="", flush=True)
        result = run([vpy, "-m", "pip", "install", pkg], capture=True) 
        if result.returncode == 0:
            print("ok")
        else:
            print("fail")
            err = (result.stderr or result.stdout or "").splitlines()
            if err:
                print("   " + err[-1].strip())

def countdown(n=3):
    print("\nStarting in", end=" ", flush=True)
    for i in range(n, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1)
    print("go\n")

def launch_loop(vpy):
    global process
    if not LOOP.exists():
        print(f"ERROR: loop script not found: {LOOP}")
        sys.exit(1)

    input("Press ENTER to start (Ctrl+C to stop)...")
    countdown(3)

    process = subprocess.Popen([vpy, str(LOOP)])
    try:
        process.wait()
    except KeyboardInterrupt:
        cleanup()

def main():
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, cleanup)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, cleanup)

    print_banner()
    print("=== Preparing environment ===")
    print("=== 15+ minutes ===")
    
    install_prerequisite_system_deps() 
    
    if not check_xserver_access():
        print("\nFATAL ERROR: Cannot guarantee X server access. Screen capture will likely fail.")
        sys.exit(1)

    if not check_uinput_access():
        print("\nFATAL ERROR: Cannot guarantee UInput access. Input simulation (mouse/keyboard control) will likely fail.")
        sys.exit(1)

    vpy = create_venv()
    install_packages(vpy)
    print("=== Ready ===")

    print("🎮 Starting supervisor loop... (Ctrl+C to stop)\n")
    launch_loop(vpy)

if __name__ == "__main__":
    main()