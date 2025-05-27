import psutil
import time
import subprocess

# Replace with the PID you want to monitor
target_pid = 30560

# Path to your Python executable and script
python_exe = ""
script_path = ""
script1_path = ""

def wait_for_pid_and_run(pid, command):
    try:
        p = psutil.Process(pid)
        print(f"Monitoring PID {pid}...")
        while p.is_running():
            time.sleep(10)
        print(f"PID {pid} has exited. Running command...")
        subprocess.run(command)
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} is not running.")

# Build command
command = [python_exe, script_path]
command1 = [python_exe, script1_path]

wait_for_pid_and_run(target_pid, command)
wait_for_pid_and_run(target_pid, command1)
