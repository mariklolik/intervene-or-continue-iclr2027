import os
import signal
import subprocess
import sys
import time


def main() -> int:
    deadline = float(sys.argv[1])
    if time.time() >= deadline:
        return 124
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, lambda signum, frame: sys.exit(128 + signum))
    process = subprocess.Popen(sys.argv[2:], start_new_session=True)
    try:
        return process.wait(timeout=max(0.001, deadline - time.time()))
    except subprocess.TimeoutExpired:
        return 124
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)


if __name__ == "__main__":
    sys.exit(main())
