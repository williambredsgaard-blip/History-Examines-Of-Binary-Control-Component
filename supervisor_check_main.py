import time
import subprocess
import sys
import os

def press_key_xdotool(key, times=1):
    """Press a key using xdotool."""
    try:
        for _ in range(times):
            subprocess.run(['xdotool', 'key', key], check=True)
            time.sleep(0.1)
        return True
    except subprocess.CalledProcessError as ex:
        print(f"ERROR: xdotool failed: {ex}")
        return False
    except FileNotFoundError:
        print("ERROR: xdotool not found. Install with: sudo apt install xdotool")
        return False

# Start supervisor and wait for it to finish
print("Starting supervisor_check.py...")
supervisor_process = subprocess.Popen([sys.executable, 'supervisor_check.py'])
supervisor_exit_code = supervisor_process.wait()

# Check if supervisor finished successfully
if supervisor_exit_code != 0:
    print(f"Supervisor failed with exit code: {supervisor_exit_code}")
    press_key_xdotool('Escape', 6)
    sys.exit(1)

print("Supervisor succeeded. Starting main.py...")
os.execvp(sys.executable, [sys.executable, 'main.py', '--no-debug'])