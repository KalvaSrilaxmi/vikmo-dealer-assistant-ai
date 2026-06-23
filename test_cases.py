import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from assistant.agent import GeminiAgent

agent = GeminiAgent()
print("Active provider:", agent.active_provider_name)
print()

queries = [
    "Do you have TVS brake pads for the Bajaj Pulsar 150?",
    "I need some brake pads.",
    "Do you have any filters in stock?",
]

for q in queries:
    agent.reset()
    print(f"Query: {q}")
    resp = agent.process_message(q)
    print(f"Tools called: {agent.tool_calls_log}")
    print(f"Response: {resp[:120]}")
    print("-" * 60)
