# main.py
from ecg_simulator.api import app

# This block only runs when you execute the file directly (python main.py)
# Vercel will import the 'app' but won't execute this block
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # This refers to the 'app' variable in this file
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