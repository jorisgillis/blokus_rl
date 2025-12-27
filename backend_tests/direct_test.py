"""
Direct test without uvicorn.run() to debug the issue.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Direct test working"}


@app.get("/api/hello")
async def hello():
    return {"message": "Hello from direct test"}


# Test directly without running server
def test_direct():
    client = TestClient(app)

    print("Testing direct FastAPI app...")

    # Test root
    response = client.get("/")
    print(f"Root: {response.status_code} - {response.json()}")

    # Test hello
    response = client.get("/api/hello")
    print(f"Hello: {response.status_code} - {response.json()}")

    print("✅ Direct test passed!")


if __name__ == "__main__":
    test_direct()
