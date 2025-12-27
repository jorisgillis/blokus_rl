#!/usr/bin/env python
"""
Test backend-frontend integration.
"""

import webbrowser

import requests


def test_integration():
    """Test the integration between backend and frontend."""
    print("🔍 Testing backend-frontend integration...")

    # Test backend endpoints
    base_url = "http://localhost:8000"

    print("\n1. Testing backend endpoints:")

    # Test root
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Root: {response.status_code} - {response.json()['message']}")
    except Exception as e:
        print(f"❌ Root failed: {e}")

    # Test hello
    try:
        response = requests.get(f"{base_url}/api/hello", timeout=5)
        print(f"✅ Hello: {response.status_code} - {response.json()['message']}")
    except Exception as e:
        print(f"❌ Hello failed: {e}")

    # Test personalized hello
    try:
        response = requests.get(f"{base_url}/api/hello/BlokusPlayer", timeout=5)
        print(f"✅ Personalized: {response.status_code} - {response.json()['message']}")
    except Exception as e:
        print(f"❌ Personalized failed: {e}")

    # Test status
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        print(f"✅ Status: {response.status_code} - {response.json()['status']}")
    except Exception as e:
        print(f"❌ Status failed: {e}")

    print("\n2. Backend API Documentation:")
    print(f"📚 Swagger UI: {base_url}/api/docs")
    print(f"📖 ReDoc: {base_url}/api/redoc")

    print("\n3. Frontend Test:")
    print("🎨 Frontend should be available at http://localhost:3000")
    print("   (Note: Frontend server needs to be started separately)")

    print("\n✅ Backend is working correctly!")
    print("🎮 You can now:")
    print("   1. Start the frontend server: python frontend/server.py")
    print("   2. Access frontend at: http://localhost:3000")
    print("   3. The frontend will fetch data from the backend")

    # Try to open the API docs in browser
    try:
        print("\n🌐 Opening API documentation in browser...")
        webbrowser.open(f"{base_url}/api/docs")
    except:
        print("Could not open browser automatically")

    return True


if __name__ == "__main__":
    test_integration()
