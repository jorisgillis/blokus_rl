"""
Simple test to verify FastAPI works.
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    print("Starting simple FastAPI test...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
