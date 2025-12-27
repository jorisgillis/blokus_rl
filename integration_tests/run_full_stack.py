"""
Run the full stack (backend + frontend).
"""

import os
import signal
import subprocess
import sys
import time


def run_backend():
    """Run the FastAPI backend."""
    print("🚀 Starting FastAPI backend...")
    backend_process = subprocess.Popen(
        [sys.executable, "backend/main.py"],
        cwd="/Users/joris/Documents/Programming/blokus",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    return backend_process


def run_frontend():
    """Run the frontend server."""
    print("🎨 Starting frontend server...")
    frontend_process = subprocess.Popen(
        [sys.executable, "frontend/server.py"],
        cwd="/Users/joris/Documents/Programming/blokus",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    return frontend_process


def test_connection():
    """Test if both servers are running."""
    import requests

    print("🔍 Testing connections...")

    # Test backend
    try:
        response = requests.get("http://localhost:8000/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("❌ Backend returned unexpected status")
    except requests.exceptions.RequestException:
        print("❌ Backend not responding")

    # Test frontend
    try:
        response = requests.get("http://localhost:3000/", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is running")
        else:
            print("❌ Frontend returned unexpected status")
    except requests.exceptions.RequestException:
        print("❌ Frontend not responding")


def main():
    """Main function to run the full stack."""
    print("🎮 Blokus Full Stack Application")
    print("=" * 40)

    try:
        # Start backend
        backend_process = run_backend()

        # Wait a bit for backend to start
        time.sleep(2)

        # Start frontend
        frontend_process = run_frontend()

        # Wait a bit for frontend to start
        time.sleep(2)

        # Test connections
        test_connection()

        print("\n🌟 Full stack is running!")
        print("📱 Frontend: http://localhost:3000")
        print("🔌 Backend:  http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/api/docs")
        print("\nPress Ctrl+C to stop all servers...")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")

        # Stop processes
        try:
            os.killpg(os.getpgid(backend_process.pid), signal.SIGTERM)
            os.killpg(os.getpgid(frontend_process.pid), signal.SIGTERM)
        except:
            pass

        print("✅ Servers stopped")

    except Exception as e:
        print(f"❌ Error: {e}")

        # Clean up processes
        try:
            os.killpg(os.getpgid(backend_process.pid), signal.SIGTERM)
            os.killpg(os.getpgid(frontend_process.pid), signal.SIGTERM)
        except:
            pass


if __name__ == "__main__":
    main()
