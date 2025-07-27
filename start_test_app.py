#!/usr/bin/env python3
"""
Simple startup script for the Personality Predictor test application.
"""

import subprocess
import sys
import time
import signal
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    # Check if Node.js is installed
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        print("✅ Node.js is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js is not installed. Please install Node.js v16 or higher.")
        return False
    
    # Check if npm is installed
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        print("✅ npm is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ npm is not installed. Please install npm.")
        return False
    
    return True

def check_model_files():
    """Check if the required model files exist."""
    print("🔍 Checking model files...")
    
    model_files = [
        "best_comprehensive_model.pkl",
        "preprocessor_comprehensive.pkl"
    ]
    
    missing_files = []
    for file in model_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        print("Please ensure all model files are in the current directory.")
        return False
    
    print("✅ All model files found")
    return True

def install_frontend_dependencies():
    """Install frontend dependencies if needed."""
    print("📦 Installing frontend dependencies...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("Installing npm packages...")
        try:
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            print("✅ Frontend dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install frontend dependencies")
            return False
    else:
        print("✅ Frontend dependencies already installed")
    
    return True

def start_backend_server():
    """Start the FastAPI backend server."""
    print("🚀 Starting backend server...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return None
    
    try:
        # Install backend dependencies first
        print("Installing backend dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      cwd=backend_dir, check=True)
        
        # Start the backend server
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Backend server started on http://localhost:8000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Backend server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend server: {e}")
        return None

def start_frontend_server():
    """Start the React frontend development server."""
    print("🚀 Starting frontend server...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None
    
    try:
        # Start the frontend server
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Frontend server started on http://localhost:3000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Frontend server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend server: {e}")
        return None

def cleanup(backend_process, frontend_process):
    """Clean up processes on exit."""
    print("\n🛑 Shutting down servers...")
    
    if backend_process:
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
    
    if frontend_process:
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_process.kill()
    
    print("✅ Servers stopped")

def main():
    """Main function to start the application."""
    print("🧠 Personality Predictor - Test Application")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        sys.exit(1)
    
    # Install frontend dependencies
    if not install_frontend_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting application servers...")
    
    # Start backend server
    backend_process = start_backend_server()
    if not backend_process:
        print("❌ Failed to start backend server")
        sys.exit(1)
    
    # Start frontend server
    frontend_process = start_frontend_server()
    if not frontend_process:
        print("❌ Failed to start frontend server")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n🎉 Application is running!")
    print("📱 Frontend: http://localhost:3000")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the servers")
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        cleanup(backend_process, frontend_process)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend server stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(backend_process, frontend_process)

if __name__ == "__main__":
    main() 