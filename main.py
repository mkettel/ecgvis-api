# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ECG Simulator API", version="1.0.0")

# CORS middleware - restrict to production domains for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://opti-ecg.com",
        "https://www.opti-ecg.com",
        "https://optical-ekg-chi.vercel.app",
        "http://localhost:3000",  # For local development
        "http://127.0.0.1:3000"   # For local development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "ECG Simulator API is running", "status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Import your ECG routes
try:
    from ecg_simulator.api import app as ecg_app
    # Since your ECG routes already have /api prefix, mount at root
    app.mount("/api", ecg_app)
except ImportError as e:
    print(f"Import error: {e}")
    # Add a debug endpoint to see what's wrong
    @app.get("/debug")
    def debug():
        return {"error": str(e), "message": "ECG simulator import failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)