import time
import os
import json
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

print("Code Commenter is running...")

# Configure your local Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"
FILE_TO_WATCH = "pytorch_tensor.py"  # Replace with your actual file
MODEL_NAME = "deepseek-coder:1.3b"  # Replace with your installed Ollama model

class WritingMonitor(FileSystemEventHandler):
    def on_modified(self, event):
        # Check if the modified file is the one we're monitoring
        if event.src_path.endswith(FILE_TO_WATCH):
            try:
                with open(FILE_TO_WATCH, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:  # Ensure the last line is not empty
                        self.check_writing_quality(last_line)
            except Exception as e:
                print("\033[91m[Error] Could not read file:\033[0m", e)

    def check_writing_quality(self, text):
        prompt = ("Explain to me in detail the latest line of code: " + text)

        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={"model": MODEL_NAME, "prompt": prompt},
                timeout=30,  # Increased timeout to 30 seconds
                stream=True
            )
            response.raise_for_status()  # Ensure we catch HTTP errors

            collected_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_data = json.loads(line.decode("utf-8"))  # Decode each JSON object
                        if "response" in json_data:
                            collected_text += json_data["response"]  # Append responses
                    except json.JSONDecodeError:
                        print("\033[91m[Error] Failed to parse JSON chunk from Ollama.\033[0m")
                        print("Raw chunk:", line.decode("utf-8"))

            result = collected_text.strip()
            if result:
                print("\033[94m[Explanation]\033[0m", result)  # Print explanation in blue
            else:
                print("\033[91m[Error] No response received from Ollama.\033[0m")

        except requests.exceptions.ConnectionError:
            print("\033[91m[Error] Could not connect to Ollama. Is `ollama serve` running?\033[0m")
        except requests.exceptions.Timeout:
            print("\033[91m[Error] Ollama API request timed out.\033[0m")
        except requests.RequestException as e:
            print("\033[91m[Error] HTTP error:\033[0m", e)

def check_prerequisites():
    """Check if the file exists and if Ollama is running."""
    if not os.path.exists(FILE_TO_WATCH):
        print(f"\033[91m[Error] File '{FILE_TO_WATCH}' not found. Please create it first.\033[0m")
        return False

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code != 200:
            print("\033[91m[Error] Ollama API is not responding properly.\033[0m")
            return False
    except requests.exceptions.ConnectionError:
        print("\033[91m[Error] Could not connect to Ollama. Is it running?\033[0m")
        return False
    except requests.exceptions.Timeout:
        print("\033[91m[Error] Connection to Ollama timed out.\033[0m")
        return False

    return True

if __name__ == "__main__":
    print("[DEBUG] Starting script...")
    if not check_prerequisites():
        print("[DEBUG] Prerequisites check failed.")
        exit(1)
    else:
        print("[DEBUG] Prerequisites check passed.")
    
    print(f"\033[92m[OK] Monitoring '{FILE_TO_WATCH}' for changes...\033[0m")
    
    event_handler = WritingMonitor()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    print("[DEBUG] Observer started.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[DEBUG] Keyboard interrupt received. Stopping observer...")
        observer.stop()
    observer.join()
    print("[DEBUG] Observer stopped.")
if __name__ == "__main__":
    if not check_prerequisites():
        exit(1)

    print(f"\033[92m[OK] Monitoring '{FILE_TO_WATCH}' for changes...\033[0m")
    
    event_handler = WritingMonitor()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()