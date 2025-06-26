#!/usr/bin/env python3
"""
Test runner that starts each Python server in the servers directory
and runs mock_client.py against each one.
"""

import os
import subprocess
import time
import sys
import signal
import glob
from pathlib import Path

# Configuration
SERVERS_DIR = "servers"
CLIENT_SCRIPT = "mock_client.py"
SERVER_PORT = 5001
STARTUP_WAIT_TIME = 3  # seconds to wait for server to start
SHUTDOWN_WAIT_TIME = 2  # seconds to wait for server to shutdown

def find_python_servers():
    """Find all Python files in the servers directory."""
    servers_path = Path(SERVERS_DIR)
    if not servers_path.exists():
        print(f"❌ Servers directory '{SERVERS_DIR}' not found")
        sys.exit(1)
    
    python_files = list(servers_path.glob("*.py"))
    if not python_files:
        print(f"❌ No Python files found in '{SERVERS_DIR}' directory")
        sys.exit(1)
    
    return python_files

def check_client_exists():
    """Check if the mock client script exists."""
    if not Path(CLIENT_SCRIPT).exists():
        print(f"❌ Client script '{CLIENT_SCRIPT}' not found")
        sys.exit(1)

def is_port_in_use(port):
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_server(port, timeout=10):
    """Wait for server to start accepting connections."""
    import socket
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex(('localhost', port)) == 0:
                    return True
        except:
            pass
        time.sleep(0.5)
    return False

def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        # Try to kill the process group
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(1)
        # If still running, force kill
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # Process already dead
    except ProcessLookupError:
        pass  # Process already dead
    except PermissionError:
        # Fallback to killing just the main process
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

def run_server_test(server_file):
    """Start a server, test it with the mock client, then kill it."""
    server_name = server_file.name
    print(f"\n🚀 Testing server: {server_name}")
    print("=" * 50)
    
    # Check if port is already in use
    if is_port_in_use(SERVER_PORT):
        print(f"❌ Port {SERVER_PORT} is already in use. Please free it first.")
        return False
    
    server_process = None
    try:
        # Start the server (only when needed)
        print(f"📡 Starting server: {server_file}")
        server_process = subprocess.Popen(
            [sys.executable, str(server_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Wait for server to start
        print(f"⏳ Waiting for server to start on port {SERVER_PORT}...")
        if not wait_for_server(SERVER_PORT):
            print(f"❌ Server {server_name} failed to start within timeout")
            return False
        
        print(f"✅ Server {server_name} started successfully")
        
        # Run the mock client against this specific server
        print(f"🧪 Running mock client against {server_name}...")
        client_result = subprocess.run(
            [sys.executable, CLIENT_SCRIPT],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for client
        )
        
        if client_result.returncode == 0:
            print(f"✅ Mock client test PASSED for {server_name}")
            print("Client output:")
            print(client_result.stdout)
            return True
        else:
            print(f"❌ Mock client test FAILED for {server_name}")
            print("Client stdout:")
            print(client_result.stdout)
            print("Client stderr:")
            print(client_result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Mock client test TIMED OUT for {server_name}")
        return False
    except Exception as e:
        print(f"❌ Error testing server {server_name}: {e}")
        return False
    
    finally:
        # Always kill the server after testing (cleanup)
        if server_process:
            print(f"🛑 Stopping server {server_name}...")
            kill_process_tree(server_process.pid)
            time.sleep(SHUTDOWN_WAIT_TIME)  # Wait for cleanup

def main():
    """Main function to run all server tests."""
    print("🔧 Face Recognition Server Test Runner")
    print("=" * 50)
    
    # Find all Python servers
    server_files = find_python_servers()
    print(f"📁 Found {len(server_files)} Python server(s) in '{SERVERS_DIR}':")
    for server_file in server_files:
        print(f"   - {server_file.name}")
    
    # Check if client exists
    check_client_exists()
    print(f"✅ Mock client script '{CLIENT_SCRIPT}' found")
    
    # Run tests for each server
    results = {}
    failed_servers = []
    
    for server_file in server_files:
        success = run_server_test(server_file)
        results[server_file.name] = success
        
        if not success:
            failed_servers.append(server_file.name)
            print(f"\n❌ CRITICAL: Test failed for {server_file.name}")
            print("Exiting due to failure...")
            sys.exit(1)
    
    # Print final results
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    
    for server_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{server_name}: {status}")
    
    if failed_servers:
        print(f"\n❌ {len(failed_servers)} server(s) failed testing")
        sys.exit(1)
    else:
        print(f"\n🎉 All {len(server_files)} server(s) passed testing!")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)