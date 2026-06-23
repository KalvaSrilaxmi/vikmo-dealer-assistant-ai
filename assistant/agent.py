import os
import json
from assistant.retriever import CatalogueRetriever
from assistant.memory import ConversationMemory
import assistant.tools as assistant_tools
from assistant.llm import GeminiProvider, GroqProvider, OllamaProvider, DemoProvider

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
User: Fictional Cruiser 500
Assistant: I found the following compatible parts:
  1. SKU: XYZ-9999 | Brake Pad Set — Fictional Cruiser 500 | Brand: TVS
  Which SKU would you like more details about?
User: XYZ-9999
Assistant: XYZ-9999 is in stock.
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
    Orchestrates the conversational agent loop, supporting local RAG,
    resilient LLM abstraction (Gemini, Groq, Ollama), keyless Demo Mode,
    and automatic failover handling.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        load_env()
        self.model_name = model_name
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
        
        # Read provider settings and fallback pipeline
        self.fallback_pipeline = [p.strip().lower() for p in os.environ.get("FALLBACK_PROVIDERS", "gemini,groq,demo").split(",") if p.strip()]
        
        # Initialize selected provider
        self.active_provider_name = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
        if os.environ.get("DEMO_MODE", "false").strip().lower() in ("true", "yes", "1"):
            self.active_provider_name = "demo"
            
        self.provider = None
        self._init_active_provider()

    @property
    def active_model_name(self) -> str:
        if self.provider:
            return self.provider.model_name
        return "Unknown"

    def _init_active_provider(self):
        """
        Attempts to initialize the active provider. If initialization fails,
        automatically routes to the fallback pipeline.
        """
        try:
            self.provider = self._create_provider_instance(self.active_provider_name)
        except Exception as e:
            self._route_to_fallback(error_msg=str(e))

    def _create_provider_instance(self, name: str):
        if name == "gemini":
            return GeminiProvider(system_instruction=SYSTEM_INSTRUCTION, model_name=self.model_name)
        elif name == "groq":
            return GroqProvider(system_instruction=SYSTEM_INSTRUCTION)
        elif name == "ollama":
            return OllamaProvider(system_instruction=SYSTEM_INSTRUCTION)
        elif name == "demo":
            return DemoProvider(system_instruction=SYSTEM_INSTRUCTION)
        else:
            raise ValueError(f"Unknown LLM provider: {name}")

    def _route_to_fallback(self, error_msg: str = "Unknown error"):
        """
        Pops the next fallback provider from the pipeline and instantiates it.
        """
        old_provider = self.active_provider_name.upper()
        print(f"\n[Warning] {old_provider} failed ({error_msg})")
        
        while self.fallback_pipeline:
            next_provider = self.fallback_pipeline.pop(0)
            print(f"Switching to {next_provider.capitalize()}...")
            try:
                self.provider = self._create_provider_instance(next_provider)
                self.active_provider_name = next_provider
                print(f"Active Provider : {self.active_provider_name.capitalize()}")
                print(f"Model           : {self.provider.model_name}\n")
                return
            except Exception as e:
                print(f"[Warning] Failed to initialize fallback '{next_provider}': {e}")
                
        # If pipeline exhausted, raise RuntimeError
        raise RuntimeError("Fallback pipeline exhausted. No functional LLM providers available.")

    def reset(self):
        """Clears memory and resets the inventory state."""
        self.memory.clear()
        self.tool_calls_log.clear()
        if os.path.exists(assistant_tools.STATE_PATH):
            try:
                os.remove(assistant_tools.STATE_PATH)
            except Exception:
                pass
        assistant_tools.load_catalogue_state()

    def process_message(self, user_message: str) -> str:
        """
        Processes a user message, performs RAG retrieval, invokes active LLM,
        handles tool execution loop, and updates conversation history.
        Implements automatic fallback if providers crash or hit quotas.
        """
        # 1. Query the semantic retriever to get grounding context for the user's input
        # Skip RAG context injection for simple greetings and out-of-domain queries to prevent RAG pollution
        cleaned_msg = user_message.strip().lower()
        is_greeting = cleaned_msg in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "yo", "sup"] or cleaned_msg.startswith(("hi ", "hello ", "hey "))
        
        if is_greeting:
            greeting_resp = "Hi, welcome to VIKMO Auto Parts! How can I help you today?"
            self.memory.add_user_message(user_message)
            self.memory.add_model_response(greeting_resp)
            return greeting_resp

        ood_keywords = ["weather", "recipe", "cookie", "poem", "news", "cricket", "song", "joke", "capital of", "france"]
        is_ood = any(kw in cleaned_msg for kw in ood_keywords)
        
        # Check if the query is ambiguous (asks for parts/stock/filters but does not specify vehicle fitment or SKU, and no context exists)
        intent_keywords = ["part", "pad", "brake", "filter", "plug", "tyre", "chain", "oil", "mirror", "lever", "cable", "accessory", "accessories", "horn", "light", "clutch", "stock", "price", "avail"]
        has_intent = any(kw in cleaned_msg for kw in intent_keywords)
        
        vehicle_brands_models = [
            "pulsar", "seltos", "meteor", "apache", "unicorn", "hornet", "shine", "splendor", "swift", "gixxer", "baleno", "creta", "alto", "duke", "dominar", "himalayan", "fz", "r15", "mt-15", "classic", "bullet", "city", "activa", "jupiter", "platina", "passion", "access", "dio",
            "yamaha", "honda", "suzuki", "bajaj", "royal enfield", "ktm", "kia", "maruti", "hyundai", "tvs", "hero"
        ]
        has_vehicle = any(v in cleaned_msg for v in vehicle_brands_models)
        
        import re
        has_sku = bool(re.search(r'\b[a-zA-Z]{3,4}-\d{4}\b', cleaned_msg))
        
        # Check history for context
        has_history_sku = False
        has_history_vehicle = False
        for msg in self.memory.get_history():
            parts = msg.get("parts", [])
            for part in parts:
                if isinstance(part, str):
                    part_lower = part.lower()
                    if any(v in part_lower for v in vehicle_brands_models):
                        has_history_vehicle = True
                    if re.search(r'\b[a-zA-Z]{3,4}-\d{4}\b', part_lower):
                        has_history_sku = True
                elif isinstance(part, dict):
                    has_history_sku = True
                    
        is_ambiguous = has_intent and not (has_vehicle or has_sku or has_history_sku or has_history_vehicle)
        
        rag_results = []
        if not (is_ood or is_ambiguous):
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
        augmented_message = f"{user_message}\n\n{context_block}"
        self.memory.add_user_message(augmented_message)
        active_tools = None if (is_greeting or is_ood or is_ambiguous) else self.tools_list
        
        # 3. Execution loop (handles potential multi-step tool calls)
        max_turns = 5
        turn = 0
        
        contents = list(self.memory.get_history())

        while turn < max_turns:
            # Generate content using resilient provider wrapping
            response = None
            retries = len(self.fallback_pipeline) + 2 # Allow trying active plus all fallbacks
            while retries > 0:
                try:
                    response = self.provider.generate_content(
                        contents=contents,
                        tools=active_tools
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    try:
                        self._route_to_fallback(error_msg=err_str)
                    except Exception as fe:
                        print(f"[Error] Fallback routing failed: {fe}")
                        return "I am sorry, all configured LLM providers are currently unavailable."
                    contents = list(self.memory.get_history()) # Refresh history format if needed
                    retries -= 1
                    
            if response is None:
                return "I am sorry, I encountered issues communicating with all configured LLM providers."

            if not response.tool_calls:
                # No tools to call, append response to memory and return
                final_text = response.text if response.text else "I am sorry, I encountered an issue generating a response."
                
                # Clean up context block in memory for clean multi-turn tracking
                history = self.memory.get_history()
                if history and history[-1]['role'] == 'user' and augmented_message in history[-1]['parts'][0]:
                    history[-1]['parts'] = [user_message]
                
                self.memory.add_model_response(final_text)
                return final_text
            
            # Record the model's tool calls in memory/history
            parts_dicts = []
            if response.text:
                parts_dicts.append(response.text)
            for tc in response.tool_calls:
                parts_dicts.append({
                    "function_call": {
                        "id": tc.get("id"),
                        "name": tc["name"],
                        "args": clean_args(tc["args"])
                    }
                })
                    
            self.memory.add_raw_message({
                "role": "model",
                "parts": parts_dicts
            })
            
            # Execute all function calls and build the response content parts
            tool_response_parts = []
            for tc in response.tool_calls:
                fn_name = tc["name"]
                fn_args = clean_args(tc["args"])
                
                # Decode JSON strings to python objects if using simplified tools schema
                if fn_name == "create_order":
                    if "items_json" in fn_args:
                        try:
                            fn_args["items"] = json.loads(fn_args["items_json"])
                            del fn_args["items_json"]
                        except Exception:
                            pass
                    if isinstance(fn_args.get("items"), str):
                        try:
                            fn_args["items"] = json.loads(fn_args["items"])
                        except Exception:
                            pass
                
                print(f"[Agent] Executing tool '{fn_name}' with args: {fn_args}")
                self.tool_calls_log.append(fn_name)
                
                if fn_name in self.tool_map:
                    try:
                        tool_result = self.tool_map[fn_name](**fn_args)
                    except Exception as e:
                        tool_result = {"status": "ERROR", "error": str(e)}
                else:
                    tool_result = {"status": "ERROR", "error": f"Tool '{fn_name}' not implemented."}
                
                tool_response_parts.append({
                    "function_response": {
                        "id": tc.get("id"),
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
