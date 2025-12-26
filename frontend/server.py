"""
Simple server to serve the frontend.
"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os


class FrontendHandler(SimpleHTTPRequestHandler):
    """Custom handler for frontend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="frontend/public", **kwargs)


def serve_frontend(port=3000):
    """Serve the frontend on the specified port."""
    print(f"Serving frontend on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")

    # Change to the frontend directory
    os.chdir("frontend/public")

    with ThreadingHTTPServer(("", port), FrontendHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nFrontend server stopped")


if __name__ == "__main__":
    serve_frontend()
