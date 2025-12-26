"""
Minimal backend test to isolate the issue.
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Minimal test working"}


if __name__ == "__main__":
    print("Starting minimal backend test...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Use different port
        log_level="info",
    )
