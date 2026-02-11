import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path.cwd()))

from smartstress_langgraph.llm.client import generate_chat

def test_connection():
    print("Testing Gemini API connection...")
    messages = [{"role": "user", "content": "Hello, are you working?"}]
    
    try:
        response = generate_chat(messages)
        print("\n--- Success! ---")
        print(f"Response: {response}")
    except Exception as e:
        print("\n--- Failed! ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

if __name__ == "__main__":
    test_connection()
