"""
run_dashboard.py - Local launcher for the Quantum Kernel & Hardware Co-Design Explorer.

Usage:
    python interactive_dashboard/run_dashboard.py
"""
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))


def run():
    os.chdir(DASHBOARD_DIR)
    url = f"http://localhost:{PORT}/index.html"
    print(f"Launching Quantum Kernel & Hardware Co-Design Explorer...")
    print(f"Serving at: {url}")
    print("Press Ctrl+C to stop the server.")
    
    webbrowser.open(url)
    server = HTTPServer(("localhost", PORT), SimpleHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
        server.server_close()


if __name__ == "__main__":
    run()
