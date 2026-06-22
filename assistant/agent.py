import os
import json
import google.generativeai as genai
from assistant.retriever import CatalogueRetriever
from assistant.memory import ConversationMemory
import assistant.tools as assistant_tools

# System Instruction Persona and Rules
SYSTEM_INSTRUCTION = """You are VIKMO Auto-Parts Assistant, a knowledgeable and agentic virtual assistant representing VIKMO Auto Parts. Your target users are auto-parts dealers.

Strict Operational Guidelines:
1. Grounding: You must ground all answers regarding products, prices, brand, vehicle fitment, and stock levels strictly in the provided RAG context or the output of the tools. Do NOT make up or hallucinate SKUs, prices, brands, or stock levels.
2. Guardrails: You only handle requests related to auto parts, catalogue search, checking stock, and order creation. If the user asks about out-of-domain topics (e.g., weather, general news, recipes, off-topic chats), politely decline to answer, stating that you can only assist with auto-parts queries.
3. Ambiguity: If a user's request is ambiguous (e.g., "I need brake pads", "I want an oil filter"), do not assume the part or vehicle. Instead, ask a clarifying question to determine the vehicle make/model (e.g., "For which vehicle?").
4. Pricing & Currency: All prices are in Indian Rupees (INR, ₹). Format them clearly.
5. Order Creation: To place an order, you must call the `create_order` tool. You must collect the dealer's name and a list of line items (with SKUs and quantities). If the SKU, quantity, or dealer name is missing, ask clarifying questions before calling the tool. Format the success receipt as structured Markdown (total price, items, order ID).
6. Tool Calls: You MUST invoke the appropriate tool (`find_parts_by_vehicle` or `check_stock`) whenever the user asks about parts for a specific vehicle, or asks about the price/stock/details of a specific SKU/part. Do not just reply using the provided semantic search context; you must call the tools to query the active database.

Expected Conversation Flow Example:
User: I need brake pads.
Assistant: For which vehicle?
User: Bajaj Pulsar 150
Assistant: I found the following compatible parts:
  1. SKU: BRK-1002 | Brake Pad Set — Bajaj Pulsar 150 | Brand: TVS
  Which SKU would you like more details about?
User: BRK-1002
Assistant: BRK-1002 is in stock.
  Price: ₹1460
  Available Quantity: 136
  Would you like to place an order?
User: Yes, order 5 units.
Assistant: Could you please provide your dealer name to complete the order?
User: ABC Motors
Assistant: Order created successfully.
  Order ID: ORD-E102F89A
  Total Amount: ₹7300
"""

def load_env():
    # Attempt to load from a local .env file in the workspace root if it exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(base_dir)
    env_path = os.path.join(workspace_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    parts = line.split("=", 1)
                    k = parts[0].strip()
                    v = parts[1].strip().strip('"').strip("'")
                    os.environ[k] = v

def clean_args(val):
    """
    Recursively converts protobuf MapComposite/List objects to native Python dicts/lists.
    This prevents protobuf type errors inside the python tools.
    """
    if hasattr(val, 'items') or isinstance(val, dict):
        return {k: clean_args(v) for k, v in val.items()}
    elif isinstance(val, list) or (hasattr(val, '__iter__') and not isinstance(val, (str, bytes))):
        return [clean_args(x) for x in val]
    else:
        return val

class GeminiAgent:
    """
    Orchestrates the conversational agent loop using Gemini, FAISS RAG,
    conversation memory, and tool routing.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        load_env()
        # Load API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it before running the agent.")
        genai.configure(api_key=api_key)
        
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_INSTRUCTION
        )
        
        self.retriever = CatalogueRetriever()
        self.memory = ConversationMemory()
        
        # Mapping functions for tool calling execution
        self.tool_map = {
            "check_stock": assistant_tools.check_stock,
            "find_parts_by_vehicle": assistant_tools.find_parts_by_vehicle,
            "create_order": assistant_tools.create_order
        }
        
        # Tools list for Gemini config
        self.tools_list = [
            assistant_tools.check_stock,
            assistant_tools.find_parts_by_vehicle,
            assistant_tools.create_order
        ]
        self.tool_calls_log = []

    def reset(self):
        """Clears memory and resets the inventory state."""
        self.memory.clear()
        self.tool_calls_log.clear()
        # Reset inventory state by deleting state file to force copy from source
        if os.path.exists(assistant_tools.STATE_PATH):
            try:
                os.remove(assistant_tools.STATE_PATH)
            except Exception:
                pass
        assistant_tools.load_catalogue_state()

    def process_message(self, user_message: str) -> str:
        """
        Processes a user message, performs RAG retrieval, invokes Gemini,
        handles tool execution loop, and updates conversation history.
        """
        # 1. Query the semantic retriever to get grounding context for the user's input
        rag_results = []
        try:
            rag_results = self.retriever.retrieve(user_message, top_k=4)
        except Exception as e:
            print(f"[Warning] RAG retrieval failed: {e}")
            
        # Format the RAG context block
        context_block = "\n[Grounded Catalogue Context (Top Semantic Matches)]:\n"
        if rag_results:
            for item in rag_results:
                context_block += (
                    f"- SKU: {item['sku']} | Name: {item['name']} | Category: {item['category']} | "
                    f"Brand: {item['brand']} | Fitment: {item['vehicle_fitment']} | "
                    f"Price: ₹{item['price_inr']} | Stock: {item['stock']} | Description: {item['description']}\n"
                )
        else:
            context_block += "No direct matches found in catalogue.\n"
            
        # 2. Append User Message with context block
        # We present the context block to the model alongside the user query
        augmented_message = f"{user_message}\n\n{context_block}"
        self.memory.add_user_message(augmented_message)
        
        # 3. Execution loop (handles potential multi-step tool calls)
        max_turns = 5
        turn = 0
        
        # Format the memory history for Gemini API
        contents = list(self.memory.get_history())

        while turn < max_turns:
            response = self.model.generate_content(
                contents=contents,
                tools=self.tools_list
            )
            
            # Check if there are function calls requested
            function_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
            
            if not function_calls:
                # No tools to call, append response to memory and return
                final_text = response.text if response.text else "I am sorry, I encountered an issue generating a response."
                
                # Clean up context block in memory for clean multi-turn tracking
                # Replace the last user message with just the original query to avoid bloated context history
                history = self.memory.get_history()
                if history and history[-1]['role'] == 'user' and augmented_message in history[-1]['parts'][0]:
                    history[-1]['parts'] = [user_message]
                
                self.memory.add_model_response(final_text)
                return final_text
            
            # Record the model's tool calls in memory/history
            parts_dicts = []
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    parts_dicts.append({
                        "function_call": {
                            "name": part.function_call.name,
                            "args": clean_args(dict(part.function_call.args))
                        }
                    })
                elif part.text:
                    parts_dicts.append(part.text)
                    
            self.memory.add_raw_message({
                "role": "model",
                "parts": parts_dicts
            })
            
            # Execute all function calls and build the response content parts
            tool_response_parts = []
            for function_call in function_calls:
                fn_name = function_call.name
                fn_args = clean_args(dict(function_call.args))
                
                print(f"[Agent] Executing tool '{fn_name}' with args: {fn_args}")
                self.tool_calls_log.append(fn_name)
                
                if fn_name in self.tool_map:
                    try:
                        tool_result = self.tool_map[fn_name](**fn_args)
                    except Exception as e:
                        tool_result = {"status": "ERROR", "error": str(e)}
                else:
                    tool_result = {"status": "ERROR", "error": f"Tool '{fn_name}' not implemented."}
                
                # Format function response part for Gemini API as dict
                tool_response_parts.append({
                    "function_response": {
                        "name": fn_name,
                        "response": tool_result
                    }
                })
                
            # Record the function responses in history
            self.memory.add_raw_message({
                "role": "user",
                "parts": tool_response_parts
            })
            
            # Rebuild contents from history for the next turn
            contents = list(self.memory.get_history())
            
            turn += 1
            
        return "I apologize, but I reached the maximum tool execution loops without completing the task."
