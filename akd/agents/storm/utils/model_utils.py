import subprocess
import time
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, AnyMessage

def format_messages(messages):
    return [
        {"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id} if isinstance(message, ToolMessage)
        else {"role": "user", "content": message.content} if isinstance(message, HumanMessage)
        else {"role": "system", "content": message.content} if isinstance(message, SystemMessage)
        else {"role": "assistant", "content": message.content} 
        for message in messages
    ]

def kill_ollama_server():
    """Kill the Ollama server process."""
    try:
        subprocess.run(["pkill", "-f", "ollama"], check=True)
        print("Ollama server killed successfully.")
    except subprocess.CalledProcessError:
        print("No Ollama server process found or failed to kill.")

def start_ollama_server():
    """Start the Ollama server."""
    try:
        # Start the server (adjust the command if needed)
        os.environ["OLLAMA_MODELS"] = ''
        subprocess.Popen(["./bin/ollama", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Ollama server started successfully.")
        time.sleep(10)
    except Exception as e:
        print(f"Failed to start Ollama server: {e}")

def restart_ollama_server():
    """Kill and restart the Ollama server."""
    print("Restarting Ollama server...")
    kill_ollama_server()
    time.sleep(2)  # Allow time for the process to terminate
    start_ollama_server()