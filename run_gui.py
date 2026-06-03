import os
import subprocess
import sys
import time
import webbrowser


def main():
    print("Starting Trading SDK GUI...")
    # Add src to python path so it works correctly
    os.environ["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))

    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.trading_sdk.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
    )

    # Wait for server to start
    time.sleep(1.5)

    print("\nOpening browser at http://127.0.0.1:8000 ...")
    webbrowser.open("http://127.0.0.1:8000")

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down GUI...")
        server_process.terminate()


if __name__ == "__main__":
    main()
