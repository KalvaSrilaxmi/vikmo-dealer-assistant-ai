import os
import sys

# Ensure local imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from assistant.agent import GeminiAgent

def test():
    print("Initializing agent...")
    agent = GeminiAgent()
    print("Active provider:", agent.active_provider_name)
    print("Model name:", agent.active_model_name)
    print("Sending 'hi' to agent...")
    response = agent.process_message("hi")
    print("\n--- Response ---")
    print(response)
    print("----------------")
    print("Tool calls logged:", agent.tool_calls_log)

if __name__ == "__main__":
    test()
