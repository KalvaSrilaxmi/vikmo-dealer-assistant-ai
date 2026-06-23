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
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    load_env()
    
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    demo_mode = os.environ.get("DEMO_MODE", "false").strip().lower() in ("true", "yes", "1")
    fallback_pipeline = os.environ.get("FALLBACK_PROVIDERS", "groq,demo").strip().upper()
    
    print("=" * 60)
    print("        VIKMO Auto-Parts Dealer Assistant CLI")
    print("=" * 60)
    if demo_mode:
        print("  * Active Mode: DEMO MODE (Local Rule-based, Keyless)")
    else:
        print(f"  * Primary LLM Provider: {provider.upper()}")
        print(f"  * Fallback Pipeline: {fallback_pipeline}")
        
    print("Commands:")
    print("  - Type 'exit' or 'quit' to end the session.")
    print("  - Type 'reset' to clear chat history & reset catalogue stock.")
    print("=" * 60)

    # Print warnings for missing keys instead of exiting
    if not demo_mode:
        if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            print("[Info] GEMINI_API_KEY not found. Agent will attempt fallbacks.")
        elif provider == "groq" and not os.environ.get("GROQ_API_KEY"):
            print("[Info] GROQ_API_KEY not found. Agent will attempt fallbacks.")

    print("Initializing agent and loading retriever index...")
    try:
        agent = GeminiAgent()
        print(f"Agent loaded successfully! Active provider: {agent.active_provider_name.upper()}")
        print("You can start chatting now.")
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
            try:
                print(f"\nAssistant: {response}")
            except UnicodeEncodeError:
                try:
                    print(f"\nAssistant: {response.replace('₹', 'INR')}")
                except Exception:
                    # Fallback to ascii representation
                    print(f"\nAssistant: {response.encode('ascii', errors='replace').decode('ascii')}")
        except Exception as e:
            print(f"\n[Error] Failed to process message: {e}")
        print("-" * 60)

if __name__ == "__main__":
    main()
