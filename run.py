import os
import sys
import subprocess

def main():
    print("[HarvestLenz] Starting FastAPI Bootstrapper...")
    
    # 1. Determine paths
    backend_dir = r"c:\Users\sssha\harv\backend\backend"
    
    # 2. Check virtual environment
    venv_dir = os.path.join(backend_dir, "venv")
    if not os.path.exists(venv_dir):
        print(f"[INFO] Creating virtual environment at {venv_dir}...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        
    # 3. Determine python executable inside venv
    if sys.platform == "win32":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")
        
    # 4. Change CWD to backend_dir
    os.chdir(backend_dir)
    
    # 5. Launch FastAPI development server via uvicorn
    port = sys.argv[1] if len(sys.argv) > 1 else "8001"
    print(f"[INFO] Launching FastAPI Dev Server on http://127.0.0.1:{port} ...")
    try:
        cmd = [venv_python, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", port, "--reload"]
        subprocess.check_call(cmd)
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user.")
        
if __name__ == "__main__":
    main()
