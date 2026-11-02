#!/bin/sh
# Starts the EMG arm controller with the first Python that has the xArm packages.
# Usage: ./run.sh [--arm usb] [--band bridge|wifi:<ip>|serial|myo|sim] [--key KEY]
cd "$(dirname "$0")" || exit 1
for py in .venv/bin/python ../Vision-Guided-Robotic-System/.venv/bin/python python3; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c "import xarm, hid" >/dev/null 2>&1; then
    exec "$py" emg_arm_control.py "$@"
  fi
done
echo "No Python with xarm+hidapi found - starting anyway (simulation / BioWave-only features)."
echo "Install with:  python3 -m pip install -r requirements.txt"
exec python3 emg_arm_control.py "$@"
