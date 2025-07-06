import argparse
import subprocess
import sys
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start API, dashboard, and optionally the orchestrator"
    )
    parser.add_argument("--api-port", type=int, default=8000, help="Port for FastAPI service")
    parser.add_argument("--dash-port", type=int, default=8501, help="Port for Streamlit dashboard")
    parser.add_argument("--orchestrator", action="store_true", help="Start the orchestrator CLI")
    parser.add_argument("--ngrok", action="store_true", help="Expose dashboard via ngrok tunnel")
    args = parser.parse_args()

    processes = []

    # Start FastAPI service
    api_cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(args.api_port)]
    api_proc = subprocess.Popen(api_cmd)
    processes.append(api_proc)
    time.sleep(2)

    # Optionally start orchestrator
    if args.orchestrator:
        orch_cmd = [sys.executable, "-m", "core.orchestrator"]
        orch_proc = subprocess.Popen(orch_cmd)
        processes.append(orch_proc)
        time.sleep(1)

    env = os.environ.copy()
    env["ZAN_API_URL"] = f"http://localhost:{args.api_port}"

    # Start Streamlit dashboard
    dash_cmd = ["streamlit", "run", "🏠 Home.py", "--server.port", str(args.dash_port)]
    dash_proc = subprocess.Popen(dash_cmd, env=env)
    processes.append(dash_proc)

    # Optional ngrok tunnel
    if args.ngrok:
        ngrok_cmd = ["ngrok", "http", str(args.dash_port)]
        ngrok_proc = subprocess.Popen(ngrok_cmd)
        processes.append(ngrok_proc)

    try:
        dash_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
