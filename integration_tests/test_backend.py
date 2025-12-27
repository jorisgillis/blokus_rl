#!/usr/bin/env python
"""
Test script for FastAPI backend.
"""

import os
import signal
import subprocess
import time

import requests


def test_backend():
    """Test the FastAPI backend."""
    print("Starting FastAPI backend...")

    # Start the backend in a subprocess
    process = subprocess.Popen(
        ["python", "backend/main.py"],
        cwd="/Users/joris/Documents/Programming/blokus",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )

    # Wait for server to start
    time.sleep(3)

    try:
        # Test the endpoints
        base_url = "http://localhost:8000"

        print("Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Root: {response.status_code} - {response.json()}")

        print("Testing hello endpoint...")
        response = requests.get(f"{base_url}/api/hello", timeout=5)
        print(f"Hello: {response.status_code} - {response.json()}")

        print("Testing personalized hello...")
        response = requests.get(f"{base_url}/api/hello/BlokusPlayer", timeout=5)
        print(f"Personalized: {response.status_code} - {response.json()}")

        print("Testing status endpoint...")
        response = requests.get(f"{base_url}/api/status", timeout=5)
        print(f"Status: {response.status_code} - {response.json()}")

        print("✅ All backend tests passed!")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Backend test failed: {e}")
        return False

    finally:
        # Clean up: terminate the subprocess
        print("Stopping backend...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()


if __name__ == "__main__":
    test_backend()
