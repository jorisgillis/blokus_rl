"""
Testable FastAPI app without automatic server startup.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient


# Create the FastAPI app
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, Blokus World!"}


@app.get("/api/hello")
async def hello():
    return {"message": "Hello from Blokus API!"}


@app.get("/api/hello/{name}")
async def hello_name(name: str):
    return {"message": f"Hello, {name}! Welcome to Blokus!"}


# Test the app directly
def test_app():
    """Test the FastAPI app using TestClient."""
    client = TestClient(app)

    print("Testing FastAPI app...")

    # Test root endpoint
    response = client.get("/")
    print(f"Root: {response.status_code} - {response.json()}")
    assert response.status_code == 200
    assert response.json()["message"] == "Hello, Blokus World!"

    # Test hello endpoint
    response = client.get("/api/hello")
    print(f"Hello: {response.status_code} - {response.json()}")
    assert response.status_code == 200
    assert response.json()["message"] == "Hello from Blokus API!"

    # Test personalized hello
    response = client.get("/api/hello/Player1")
    print(f"Personalized: {response.status_code} - {response.json()}")
    assert response.status_code == 200
    assert "Player1" in response.json()["message"]

    print("✅ All FastAPI tests passed!")
    return True


if __name__ == "__main__":
    test_app()
