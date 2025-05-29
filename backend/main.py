# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ecg_simulator.api import app as ecg_app  # Import your existing app
import uvicorn

# Create the main FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount or include your ECG simulator routes
# Option 1: If ecg_app is a FastAPI app, mount it
app.mount("/api", ecg_app)

# Option 2: If you have routers, include them instead
# from ecg_simulator.api import router
# app.include_router(router)

# Add a simple root endpoint to test
@app.get("/")
def read_root():
    return {"message": "ECG Simulator API is running"}

# Local development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=[
            "*/.git/*",
            "*/__pycache__/*",
            "*.pyc",
            "*/.pytest_cache/*",
            "*/.vscode/*",
            "*/.idea/*"
        ],
        reload_delay=1,
        reload_includes=["*.py", "*.html", "*.css", "*.js"]
    )