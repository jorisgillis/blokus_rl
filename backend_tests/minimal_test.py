"""
Minimal backend test to isolate the issue.
"""

import uvicorn
from fastapi import FastAPI

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
