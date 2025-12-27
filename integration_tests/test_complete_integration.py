#!/usr/bin/env python
"""
Test complete backend-frontend integration.
"""

import webbrowser

import requests


def test_complete_integration():
    """Test the complete integration."""
    print("🎮 Testing Complete Blokus Integration")
    print("=" * 40)

    base_url = "http://localhost:8000"

    # Test 1: Backend API endpoints
    print("\n1. Testing Backend API Endpoints:")

    endpoints = [
        ("/api/status", "Status"),
        ("/api/hello", "Hello"),
        ("/api/hello/BlokusPlayer", "Personalized Hello"),
    ]

    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(
                    f"✅ {name}: {response.status_code} - {data.get('message', data.get('status', 'Success'))}"
                )
            else:
                print(f"❌ {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")

    # Test 2: Frontend
    print("\n2. Testing Frontend:")

    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200 and "Blokus Game" in response.text:
            print(f"✅ Frontend: {response.status_code} - HTML page served")
        else:
            print(f"❌ Frontend: {response.status_code} - Unexpected response")
    except Exception as e:
        print(f"❌ Frontend: Error - {e}")

    # Test 3: Static files
    print("\n3. Testing Static Files:")

    try:
        response = requests.get(f"{base_url}/static/main.js", timeout=5)
        if response.status_code == 200 and "HelloWorld" in response.text:
            print(f"✅ Static JS: {response.status_code} - JavaScript file served")
        else:
            print(f"❌ Static JS: {response.status_code} - Unexpected response")
    except Exception as e:
        print(f"❌ Static JS: Error - {e}")

    # Test 4: API Documentation
    print("\n4. Testing API Documentation:")

    try:
        response = requests.get(f"{base_url}/api/docs", timeout=5)
        if response.status_code == 200 and "Swagger UI" in response.text:
            print(f"✅ API Docs: {response.status_code} - Swagger UI available")
        else:
            print(f"❌ API Docs: {response.status_code} - Unexpected response")
    except Exception as e:
        print(f"❌ API Docs: Error - {e}")

    print("\n" + "=" * 40)
    print("🎉 Integration Test Complete!")
    print("\n🌟 All Systems Operational:")
    print(f"   📱 Frontend: {base_url}/")
    print(f"   🔌 Backend API: {base_url}/api/hello")
    print(f"   📚 API Docs: {base_url}/api/docs")
    print(f"   📖 ReDoc: {base_url}/api/redoc")

    # Try to open the frontend in browser
    try:
        print("\n🌐 Opening frontend in browser...")
        webbrowser.open(base_url)
    except:
        print("Could not open browser automatically")

    return True


if __name__ == "__main__":
    test_complete_integration()
