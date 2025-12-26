"""
Working backend with alternative approach.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Blokus Game API",
    description="Backend API for the Blokus board game",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


# Define API routes first (they take precedence)
@app.get("/api/hello")
async def hello():
    return {"message": "Hello from Blokus API!"}


@app.get("/api/hello/{name}")
async def hello_name(name: str):
    return {"message": f"Hello, {name}! Welcome to Blokus!"}


@app.get("/api/status")
async def status():
    return {
        "status": "running",
        "message": "Blokus API is up and running!",
        "endpoints": [
            {"path": "/", "description": "Root endpoint"},
            {"path": "/api/hello", "description": "Hello World"},
            {"path": "/api/hello/{name}", "description": "Personalized hello"},
            {"path": "/api/status", "description": "API status"},
        ],
    }


# Mount static files for frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
if os.path.exists(frontend_path):
    # Mount static files on /static path
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    logger.info(f"Mounted static files from: {frontend_path}")

    # Serve index.html for root route
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    # Catch-all route for frontend routing (for single-page applications)
    # This will serve index.html for any non-API routes
    @app.get("/{full_path:path}")
    async def serve_frontend_routes(full_path: str, request: Request):
        # Serve index.html for all non-API routes (for frontend routing)
        return FileResponse(os.path.join(frontend_path, "index.html"))
else:
    logger.warning(f"Frontend not found at: {frontend_path}")

    # Fallback root route if frontend not found
    @app.get("/")
    async def root():
        return {"message": "Hello, Blokus World!"}


if __name__ == "__main__":
    logger.info("Starting Blokus backend...")

    # Try running with different configuration
    try:
        uvicorn.run(
            "working_backend:app",  # Use just the filename
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            # Try without reload first
            reload=False,
            # Try with different workers
            workers=1,
        )
    except Exception as e:
        logger.error(f"Failed to start backend: {e}")
        raise
