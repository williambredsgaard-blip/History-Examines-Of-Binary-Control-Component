#!/usr/bin/env python3
import subprocess
import sys
import os

def run_supervisor():
    print("🔍 Checking CS2 state...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    check_script = os.path.join(script_dir, 'check_menu_game.py')
    
    result = subprocess.run([sys.executable, check_script], 
                          capture_output=True, text=True)
    
    exit_code = result.returncode
    print(f"📊 Detection exit code: {exit_code}")
    
    if exit_code == 0:  # GAME
        print("🎮 Already in game")
        print("SUCCESS")
        os._exit(0)
    elif exit_code == 1:  # MENU
        print("🚀 Starting new game sequence...")
        start_script = os.path.join(script_dir, 'start_new_game.py')
        start_result = subprocess.run([sys.executable, start_script])
        if start_result.returncode == 0:
            print("✅ Game sequence completed")
            print("SUCCESS")
            os._exit(0)
        else:
            print("❌ Game sequence failed")
            print("ERROR")
            os._exit(1)
    else:  # Error (exit code 2 or other)
        print(f"⚠️ Detection error (code {exit_code})")
        if result.stderr:
            print(f"🔧 stderr: {result.stderr[:500]}")
        print("ERROR")
        os._exit(1)

def main():
    run_supervisor()

if __name__ == "__main__":
    main()