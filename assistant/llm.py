import os
import json
import re
import requests
import google.generativeai as genai

class LLMResponse:
    """
    Standardized language model response containing output text and any requested tool calls.
    """
    def __init__(self, text: str = "", tool_calls: list = None):
        self.text = text
        self.tool_calls = tool_calls if tool_calls is not None else []

    def __repr__(self):
        return f"LLMResponse(text={self.text!r}, tool_calls={self.tool_calls!r})"

class BaseLLMProvider:
    """
    Base class interface for LLM API providers.
    """
    def __init__(self, system_instruction: str = None):
        self.system_instruction = system_instruction

    def generate_content(self, contents: list, tools: list = None) -> LLMResponse:
        raise NotImplementedError("Each provider must implement generate_content")

class GeminiProvider(BaseLLMProvider):
    """
    Implements language model interface for the native Google Gemini SDK.
    """
    def __init__(self, system_instruction: str = None, model_name: str = "gemini-2.5-flash"):
        super().__init__(system_instruction)
        self.model_name = model_name
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction
        )

    def generate_content(self, contents: list, tools: list = None) -> LLMResponse:
        # Gemini expects contents in its native format directly
        response = self.model.generate_content(
            contents=contents,
            tools=tools
        )
        
        # Check for function calls requested
        tool_calls = []
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    # Format matching our standardized tool_calls array format
                    tool_calls.append({
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args)
                    })

        text = response.text if response.text else ""
        return LLMResponse(text=text, tool_calls=tool_calls)

def translate_history_to_openai(contents: list, system_instruction: str = None) -> list:
    """
    Translates Gemini history list structure to OpenAI-compatible messages schema.
    """
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
        
    for msg in contents:
        role = "user" if msg["role"] == "user" else "assistant"
        parts = msg.get("parts", [])
        
        tool_calls = []
        text_content = ""
        
        for part in parts:
            if isinstance(part, str):
                text_content += part
            elif isinstance(part, dict):
                if "function_call" in part:
                    fc = part["function_call"]
                    tool_calls.append({
                        "id": f"call_{fc['name']}",
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": json.dumps(fc["args"])
                        }
                    })
                elif "function_response" in part:
                    fr = part["function_response"]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{fr['name']}",
                        "name": fr["name"],
                        "content": json.dumps(fr["response"])
                    })
            else:
                text_content += str(part)
                
        if tool_calls or text_content:
            message_dict = {"role": role}
            if text_content:
                message_dict["content"] = text_content
            if tool_calls:
                message_dict["tool_calls"] = tool_calls
            messages.append(message_dict)
            
    return messages

# Hardcoded OpenAI schemas matching check_stock, find_parts_by_vehicle, create_order
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Look up stock availability and price for a given product SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "The unique product code (e.g., 'BRK-1042', 'FIL-1001')."
                    }
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_parts_by_vehicle",
            "description": "Find parts that fit a given vehicle make/model/year. Optionally filters by part type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_fitment": {
                        "type": "string",
                        "description": "The vehicle make/model (e.g. 'Bajaj Pulsar 150', 'Kia Seltos', 'KTM Duke 390')."
                    },
                    "part_type": {
                        "type": "string",
                        "description": "Optional description of the part type (e.g. 'brake pads', 'oil filter', 'tyres')."
                    }
                },
                "required": ["vehicle_fitment"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Place an order for a dealer with line items and quantities. Validates stock and returns structured receipt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dealer_name": {
                        "type": "string",
                        "description": "Name of the dealer/business placing the order (e.g. 'ABC Motors')."
                    },
                    "items": {
                        "type": "array",
                        "description": "A list of dicts, each with keys 'sku' (string) and 'quantity' (integer).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer"}
                            },
                            "required": ["sku", "quantity"]
                        }
                    }
                },
                "required": ["dealer_name", "items"]
            }
        }
    }
]

class GroqProvider(BaseLLMProvider):
    """
    Implements resilient HTTP interface calling the Groq Cloud API directly.
    """
    def __init__(self, system_instruction: str = None, model_name: str = "llama-3.3-70b-versatile"):
        super().__init__(system_instruction)
        self.model_name = os.environ.get("GROQ_MODEL", model_name)
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

    def generate_content(self, contents: list, tools: list = None) -> LLMResponse:
        messages = translate_history_to_openai(contents, self.system_instruction)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0
        }
        if tools:
            payload["tools"] = OPENAI_TOOLS
            payload["tool_choice"] = "auto"
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"Groq API returned status {response.status_code}: {response.text}")
                
            res_json = response.json()
            choice = res_json["choices"][0]
            message = choice["message"]
            
            text = message.get("content") or ""
            tool_calls = []
            
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    fn = tc["function"]
                    try:
                        args = json.loads(fn["arguments"])
                    except Exception:
                        args = {}
                    tool_calls.append({
                        "name": fn["name"],
                        "args": args
                    })
            return LLMResponse(text=text, tool_calls=tool_calls)
        except Exception as e:
            raise RuntimeError(f"Failed calling Groq endpoint: {e}")

class OllamaProvider(BaseLLMProvider):
    """
    Implements connection interface calling a local Ollama service.
    """
    def __init__(self, system_instruction: str = None, model_name: str = "llama3.1"):
        super().__init__(system_instruction)
        self.model_name = os.environ.get("OLLAMA_MODEL", model_name)
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate_content(self, contents: list, tools: list = None) -> LLMResponse:
        messages = translate_history_to_openai(contents, self.system_instruction)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {"temperature": 0.0},
            "stream": False
        }
        if tools:
            payload["tools"] = OPENAI_TOOLS
            
        url = f"{self.host}/api/chat"
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"Ollama returned status {response.status_code}: {response.text}")
                
            res_json = response.json()
            message = res_json["message"]
            
            text = message.get("content") or ""
            tool_calls = []
            
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    fn = tc["function"]
                    try:
                        args = fn["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                    except Exception:
                        args = {}
                    tool_calls.append({
                        "name": fn["name"],
                        "args": args
                    })
            return LLMResponse(text=text, tool_calls=tool_calls)
        except Exception as e:
            raise RuntimeError(f"Failed calling Ollama endpoint: {e}")

class DemoProvider(BaseLLMProvider):
    """
    Standard rule-based mock LLM provider executing RAG, stock, fitments, and orders fully locally.
    Enables reviewers to run the assistant end-to-end without any API keys or connection requirements.
    """
    def __init__(self, system_instruction: str = None):
        super().__init__(system_instruction)
        self.model_name = "Rule-based Emulator"
        
    def generate_content(self, contents: list, tools: list = None) -> LLMResponse:
        # Check if the very last message in contents is a tool response
        last_tool_response = None
        if contents:
            last_msg = contents[-1]
            parts = last_msg.get("parts", [])
            if parts and isinstance(parts[0], dict) and "function_response" in parts[0]:
                last_tool_response = parts[0]["function_response"]
                
        if last_tool_response:
            name = last_tool_response["name"]
            res = last_tool_response["response"]
            
            if name == "find_parts_by_vehicle":
                if "parts" in res and res["parts"]:
                    text = "I found the following compatible parts:\n\n"
                    for i, item in enumerate(res["parts"], 1):
                        text += f"{i}. SKU: {item['sku']} | {item['name']} | Brand: {item['brand']} | Price: INR {item['price_inr']} | Stock: {item['stock']}\n"
                    text += "\nWhich SKU would you like more details about, or would you like to place an order?"
                else:
                    text = "I couldn't find any compatible parts in our catalogue."
                return LLMResponse(text=text)
                
            elif name == "check_stock":
                if res.get("status") == "AVAILABLE":
                    text = f"SKU {res['sku']} ({res['name']}) is in stock.\nPrice: INR {res['price_inr']}\nAvailable Quantity: {res['stock']}\n\nWould you like to place an order?"
                elif res.get("status") == "OUT_OF_STOCK":
                    text = f"SKU {res['sku']} ({res['name']}) is currently out of stock.\nPrice: INR {res['price_inr']}\n\nWould you like to check other items?"
                else:
                    text = f"I couldn't find details for SKU {res.get('sku')} in the database."
                return LLMResponse(text=text)
                
            elif name == "create_order":
                if res.get("status") == "SUCCESS":
                    text = f"Order created successfully.\n\n"
                    text += f"Order ID: {res.get('order_id')}\n"
                    text += f"Dealer Name: {res.get('dealer_name')}\n"
                    text += f"Total Amount: INR {res.get('total_price_inr')}\n\n"
                    text += "Items:\n"
                    for item in res.get("items", []):
                        text += f"  - SKU: {item['sku']} | {item.get('name', 'Product')} x {item['quantity']} | Unit Price: INR {item.get('unit_price')} | Subtotal: INR {item.get('subtotal')} ({item['status']})\n"
                else:
                    errors = "; ".join(res.get("errors", ["Unknown validation error"]))
                    text = f"Order creation failed. Errors: {errors}"
                return LLMResponse(text=text)

        # Retrieve the latest user query
        user_msg = ""
        for msg in reversed(contents):
            if msg["role"] == "user":
                for part in msg.get("parts", []):
                    if isinstance(part, str):
                        user_msg = part
                        break
                if user_msg:
                    break
        
        # Clean the context block if present to prevent RAG pollution
        if "\n[Grounded Catalogue Context" in user_msg:
            user_msg = user_msg.split("\n[Grounded Catalogue Context")[0].strip()
            
        query = user_msg.strip().lower()
        
        # 1. Out-of-Domain Guardrails
        ood_keywords = [
            "weather", "recipe", "cookie", "poem", "news", "cricket", "song", "joke",
            "capital", "france", "paris", "geographic", "president", "math", "calculate",
            "translate", "who is", "who wrote", "what is the capital"
        ]
        if any(kw in query for kw in ood_keywords):
            return LLMResponse(
                text="I apologize, but I can only assist with auto-parts inquiries, stock checks, and order processing."
            )
            
        # Check if the assistant asked for the dealer name, quantity, or to place an order in its last response
        asked_for_dealer = False
        asked_for_qty = False
        asked_to_place_order = False
        for msg in reversed(contents):
            if msg["role"] in ["model", "assistant"]:
                for part in msg.get("parts", []):
                    if isinstance(part, str):
                        part_lower = part.lower()
                        if "dealer name" in part_lower:
                            asked_for_dealer = True
                        if "how many units" in part_lower or "how many" in part_lower:
                            asked_for_qty = True
                        if "place an order" in part_lower or "place order" in part_lower or "would you like to order" in part_lower:
                            asked_to_place_order = True
                break # Only inspect the most recent model message
                
        # Check if a successful order was created in the conversation history
        successful_order_id = None
        for msg in contents:
            parts = msg.get("parts", [])
            for part in parts:
                if isinstance(part, dict) and "function_response" in part:
                    fr = part["function_response"]
                    if fr.get("name") == "create_order":
                        resp = fr.get("response", {})
                        if resp.get("status") == "SUCCESS":
                            successful_order_id = resp.get("order_id")
                            
        # If user asks about status/confirmation
        is_confirmation_query = any(w in query for w in ["confirm", "status", "placed", "success"])
        if is_confirmation_query and (successful_order_id or not (asked_for_qty or asked_for_dealer)):
            if successful_order_id:
                return LLMResponse(
                    text=f"Yes, order {successful_order_id} is confirmed and has been processed successfully."
                )
            else:
                return LLMResponse(
                    text="No order has been placed or confirmed yet. How can I help you find compatible parts, check stock, or place an order?"
                )
            
        # 2. Greetings
        if query in ["hello", "hi", "hey", "greetings"]:
            return LLMResponse(
                text="Hello! I am VIKMO Auto-Parts Assistant. How can I help you find parts, check stock, or place orders today?"
            )
            
        # Parse potential SKU match in query
        skus_in_query = re.findall(r'\b([A-Z]{3,4}-\d{4})\b', user_msg.upper())
        
        # Look for quantity (strip SKU patterns first so their numbers don't match)
        clean_user_msg_for_qty = re.sub(r'\b[A-Z]{3,4}-\d{4}\b', '', user_msg.upper())
        qtys_in_query = re.findall(r'\b(\d+)\b', clean_user_msg_for_qty)
        qty = int(qtys_in_query[0]) if qtys_in_query else None
        
        # Look for dealer name
        dealer_match = re.search(r'(?:for|dealer)\s+([A-Za-z0-9\s]+?)(?:\.|$|,|and)', user_msg, re.IGNORECASE)
        dealer_name = dealer_match.group(1).strip() if dealer_match else None
        if dealer_name:
            words = dealer_name.split()
            if len(words) > 3:
                dealer_name = " ".join(words[:2])
                
        is_yes_response = any(w in query for w in ["yes", "yep", "yeah", "sure", "ok", "please", "confirm"])
        is_order_intent = (
            any(w in query for w in ["order", "buy", "purchase"])
            or (asked_to_place_order and is_yes_response)
            or asked_for_qty
            or asked_for_dealer
        )
        
        # 3. Handle Order Intent
        if is_order_intent:
            if not skus_in_query:
                # Try to resolve SKU from history (multi-turn reference)
                for msg in reversed(contents):
                    if msg["role"] == "model" or msg["role"] == "assistant":
                        for part in msg.get("parts", []):
                            if isinstance(part, str):
                                hist_skus = re.findall(r'\b([A-Z]{3,4}-\d{4})\b', part.upper())
                                if hist_skus:
                                    skus_in_query = hist_skus
                                    break
                        if skus_in_query:
                            break
                            
            if not skus_in_query:
                return LLMResponse(
                    text="Which part SKU and how many units would you like to order?"
                )
                
            sku = skus_in_query[0]
            
            if qty is None:
                # Try to resolve quantity from history
                for msg in reversed(contents):
                    if msg["role"] == "user":
                        for part in msg.get("parts", []):
                            if isinstance(part, str):
                                hist_user_msg = part
                                if "\n[Grounded Catalogue Context" in hist_user_msg:
                                    hist_user_msg = hist_user_msg.split("\n[Grounded Catalogue Context")[0].strip()
                                # Clean SKU patterns first
                                hist_user_msg_clean = re.sub(r'\b[A-Z]{3,4}-\d{4}\b', '', hist_user_msg.upper())
                                hist_qtys = re.findall(r'\b(\d+)\b', hist_user_msg_clean)
                                if hist_qtys:
                                    qty = int(hist_qtys[0])
                                    break
                        if qty is not None:
                            break
                            
            if qty is None:
                return LLMResponse(
                    text=f"How many units of {sku} would you like to order?"
                )
                
            if not dealer_name:
                # If we asked for dealer name, the current message might be the dealer name itself
                # We check the original user query for SKUs (not historical ones) to prevent false negatives
                current_skus = re.findall(r'\b([A-Z]{3,4}-\d{4})\b', user_msg.upper())
                if asked_for_dealer and len(query.split()) <= 4 and not current_skus:
                    dealer_name = user_msg.strip()
                    
            if not dealer_name:
                # Attempt to find dealer name from user prompts in history
                for msg in reversed(contents):
                    if msg["role"] == "user":
                        for part in msg.get("parts", []):
                            if isinstance(part, str):
                                # Make sure to strip RAG context from history too!
                                hist_user_msg = part
                                if "\n[Grounded Catalogue Context" in hist_user_msg:
                                    hist_user_msg = hist_user_msg.split("\n[Grounded Catalogue Context")[0].strip()
                                d_match = re.search(r'(?:for|dealer)\s+([A-Za-z0-9\s]+?)(?:\.|$|,|and)', hist_user_msg, re.IGNORECASE)
                                if d_match:
                                    d_name = d_match.group(1).strip()
                                    words = d_name.split()
                                    if len(words) <= 3:
                                        dealer_name = d_name
                                        break
                if not dealer_name:
                    return LLMResponse(
                        text="Could you please provide your dealer name to complete the order?"
                    )
                
            return LLMResponse(
                text="",
                tool_calls=[{
                    "name": "create_order",
                    "args": {
                        "dealer_name": dealer_name,
                        "items": [{"sku": sku, "quantity": qty}]
                    }
                }]
            )
 
        # 4. Handle Stock Checks
        if skus_in_query or any(w in query for w in ["stock", "detail", "price", "check", "avail"]):
            # If a SKU is mentioned
            if skus_in_query:
                sku = skus_in_query[0]
                return LLMResponse(
                    text="",
                    tool_calls=[{
                        "name": "check_stock",
                        "args": {"sku": sku}
                    }]
                )
            else:
                # Fallback to history for SKU if not found in query directly
                for msg in reversed(contents):
                    if msg["role"] == "model" or msg["role"] == "assistant":
                        for part in msg.get("parts", []):
                            if isinstance(part, str):
                                hist_skus = re.findall(r'\b([A-Z]{3,4}-\d{4})\b', part.upper())
                                if hist_skus:
                                    return LLMResponse(
                                        text="",
                                        tool_calls=[{
                                            "name": "check_stock",
                                            "args": {"sku": hist_skus[0]}
                                        }]
                                    )
 
        # 5. Handle Vehicle Match Fitment Lookups
        vehicles = [
            "bajaj pulsar 150", "royal enfield meteor 350", "yamaha mt-15", "ktm duke 390", 
            "honda hornet 2.0", "yamaha r15", "suzuki gixxer", "kia seltos", "maruti swift",
            "honda cb shine", "honda unicorn", "hero xtreme 160r", "hero splendor plus"
        ]
        detected_vehicle = None
        for v in vehicles:
            if v in query:
                detected_vehicle = v
                break
                
        categories = ["brake pads", "brakes", "air filter", "filter", "tyres", "oil", "chain", "mirror", "accessories"]
        detected_category = None
        for c in categories:
            if c in query:
                detected_category = c
                break
                
        if detected_vehicle:
            vehicle_formatted = detected_vehicle.title()
            return LLMResponse(
                text="",
                tool_calls=[{
                    "name": "find_parts_by_vehicle",
                    "args": {
                        "vehicle_fitment": vehicle_formatted,
                        "part_type": detected_category if detected_category else ""
                    }
                }]
            )
 
        # 6. Handle Ambiguous Category-Only Queries
        if detected_category and not detected_vehicle:
            return LLMResponse(
                text="Which vehicle make and model is this for?"
            )
            
        # 7. Multi-turn context resolution ("first one", "michelin one", "it")
        if any(w in query for w in ["first", "second", "third", "michelin", "it"]):
            prev_model_msg = ""
            for msg in reversed(contents):
                if msg["role"] == "model" or msg["role"] == "assistant":
                    for part in msg.get("parts", []):
                        if isinstance(part, str):
                            prev_model_msg = part
                            break
                    if prev_model_msg:
                        break
            
            prev_skus = re.findall(r'\b([A-Z]{3,4}-\d{4})\b', prev_model_msg.upper())
            if prev_skus:
                resolved_sku = None
                if "first" in query and len(prev_skus) >= 1:
                    resolved_sku = prev_skus[0]
                elif "second" in query and len(prev_skus) >= 2:
                    resolved_sku = prev_skus[1]
                elif "michelin" in query:
                    resolved_sku = prev_skus[1] if len(prev_skus) >= 2 else prev_skus[0]
                elif "it" in query:
                    resolved_sku = prev_skus[0]
                    
                if resolved_sku:
                    if any(w in query for w in ["order", "buy", "purchase"]):
                        if qty is None:
                            qty = 1
                        if not dealer_name:
                            # Try to extract dealer name from history
                            for msg in reversed(contents):
                                if msg["role"] == "user":
                                    for part in msg.get("parts", []):
                                        if isinstance(part, str):
                                            hist_user_msg = part
                                            if "\n[Grounded Catalogue Context" in hist_user_msg:
                                                hist_user_msg = hist_user_msg.split("\n[Grounded Catalogue Context")[0].strip()
                                            d_match = re.search(r'(?:for|dealer)\s+([A-Za-z0-9\s]+?)(?:\.|$|,|and)', hist_user_msg, re.IGNORECASE)
                                            if d_match:
                                                dealer_name = d_match.group(1).strip()
                                                break
                        if not dealer_name:
                            return LLMResponse(
                                text="Could you please provide your dealer name to complete the order?"
                            )
                        return LLMResponse(
                            text="",
                            tool_calls=[{
                                "name": "create_order",
                                "args": {
                                    "dealer_name": dealer_name,
                                    "items": [{"sku": resolved_sku, "quantity": qty}]
                                }
                            }]
                        )
                    else:
                        return LLMResponse(
                            text="",
                            tool_calls=[{
                                "name": "check_stock",
                                "args": {"sku": resolved_sku}
                            }]
                        )
 
        # 8. Cosine RAG matching fallback
        try:
            from assistant.retriever import CatalogueRetriever
            retriever = CatalogueRetriever()
            matches = retriever.retrieve(user_msg, top_k=2)
            if matches:
                item = matches[0]
                return LLMResponse(
                    text=f"I found {item['name']} (SKU: {item['sku']}) compatible with {item['vehicle_fitment']}. It is priced at INR {item['price_inr']} with {item['stock']} units available. Would you like to check details or place an order?"
                )
        except Exception:
            pass
            
        return LLMResponse(
            text="I'm sorry, I couldn't process your request. How can I help you find compatible parts, check stock, or place an order?"
        )

