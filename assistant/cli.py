import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import sys

# Add parent directory to sys.path to enable direct execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from assistant.agent import GeminiAgent, load_env

def main():
    load_env()
    print("=" * 60)
    print("        VIKMO Auto-Parts Dealer Assistant CLI")
    print("=" * 60)
    print("Commands:")
    print("  - Type 'exit' or 'quit' to end the session.")
    print("  - Type 'reset' to clear chat history & reset catalogue stock.")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("[Error] GEMINI_API_KEY environment variable is not set!")
        print("Please set the GEMINI_API_KEY variable in your terminal and try again.")
        print("Example (PowerShell): $env:GEMINI_API_KEY='your_key_here'")
        print("Example (CMD): set GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    print("Initializing agent and loading index...")
    try:
        agent = GeminiAgent()
        print("Agent loaded successfully! You can start chatting now.")
    except Exception as e:
        print(f"[Error] Failed to initialize agent: {e}")
        sys.exit(1)

    print("-" * 60)
    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        query = user_input.strip()
        if not query:
            continue

        if query.lower() in ['exit', 'quit']:
            print("Exiting session. Goodbye!")
            break

        if query.lower() == 'reset':
            agent.reset()
            print("Assistant: Conversation memory and product catalogue stock have been reset.")
            print("-" * 60)
            continue

        print("Assistant is thinking...")
        try:
            response = agent.process_message(query)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\n[Error] Failed to process message: {e}")
        print("-" * 60)

if __name__ == "__main__":
    main()
