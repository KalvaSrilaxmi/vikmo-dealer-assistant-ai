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
    
    provider_name = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    if os.environ.get("DEMO_MODE", "false").strip().lower() in ("true", "yes", "1"):
        provider_name = "demo"
    fallback_pipeline_str = os.environ.get("FALLBACK_PROVIDERS", "groq,ollama,demo").strip()
    fallback_chain_list = [p.strip().capitalize() for p in fallback_pipeline_str.split(",") if p.strip()]
    fallback_chain = " -> ".join(fallback_chain_list)
    
    print("=" * 60)
    print("        VIKMO Auto-Parts Dealer Assistant CLI")
    print("=" * 60)
    print("Commands:")
    print("  - Type 'exit' or 'quit' to end the session.")
    print("  - Type 'reset' to clear chat history & reset catalogue stock.")
    print("=" * 60)

    # Print warnings for missing keys instead of exiting
    if provider_name == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        print("[Info] GEMINI_API_KEY not found. Agent will attempt fallbacks.")
    elif provider_name == "groq" and not os.environ.get("GROQ_API_KEY"):
        print("[Info] GROQ_API_KEY not found. Agent will attempt fallbacks.")

    print("Initializing agent and loading retriever index...")
    try:
        agent = GeminiAgent()
        print("\n" + "=" * 60)
        print(f"Active Provider : {agent.active_provider_name.capitalize()}")
        print(f"Model           : {agent.active_model_name}")
        print(f"Fallback Chain  : {fallback_chain}")
        print("=" * 60)
        print("\nAgent loaded successfully! You can start chatting now.")
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
            provider_lbl = f"{agent.active_provider_name.capitalize()} - {agent.active_model_name}"
            try:
                print(f"\nAssistant (via {provider_lbl}): {response}")
            except UnicodeEncodeError:
                try:
                    print(f"\nAssistant (via {provider_lbl}): {response.replace('₹', 'INR')}")
                except Exception:
                    # Fallback to ascii representation
                    print(f"\nAssistant (via {provider_lbl}): {response.encode('ascii', errors='replace').decode('ascii')}")
        except Exception as e:
            print(f"\n[Error] Failed to process message: {e}")
        print("-" * 60)

if __name__ == "__main__":
    main()
