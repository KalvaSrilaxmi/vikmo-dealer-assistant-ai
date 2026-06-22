import os
import json
import uuid
from assistant.retriever import CatalogueRetriever

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "index_metadata.json")
STATE_PATH = os.path.join(BASE_DIR, "catalogue_state.json")

def load_catalogue_state():
    """
    Loads the current active inventory state of the catalogue.
    If the state file does not exist, copies it from index_metadata.json.
    """
    if not os.path.exists(STATE_PATH):
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Catalogue metadata file not found at {METADATA_PATH}")
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Initialize active catalogue state
        with open(STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return metadata
    
    with open(STATE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_catalogue_state(state):
    """
    Saves the updated active inventory state to catalogue_state.json.
    """
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def check_stock(sku: str) -> dict:
    """
    Look up stock availability and price for a given product SKU.

    Args:
        sku: The unique product code (e.g., 'BRK-1042', 'FIL-1001').

    Returns:
        A dictionary containing SKU, product name, brand, stock level, price in INR, and status.
    """
    sku = sku.strip().upper()
    state = load_catalogue_state()
    
    for item in state:
        if item['sku'] == sku:
            return {
                "sku": sku,
                "name": item['name'],
                "brand": item['brand'],
                "stock": item['stock'],
                "price_inr": item['price_inr'],
                "status": "AVAILABLE" if item['stock'] > 0 else "OUT_OF_STOCK"
            }
            
    return {
        "sku": sku,
        "status": "NOT_FOUND",
        "error": f"SKU {sku} does not exist in the catalogue."
    }

def find_parts_by_vehicle(vehicle_fitment: str, part_type: str = None) -> dict:
    """
    Find parts that fit a given vehicle make/model/year. Optionally filters by part type.

    Args:
        vehicle_fitment: The vehicle make/model (e.g. 'Bajaj Pulsar 150', 'Kia Seltos', 'KTM Duke 390').
        part_type: Optional description of the part type (e.g. 'brake pads', 'oil filter', 'tyres').

    Returns:
        A dictionary wrapping a list of matching products under the key 'parts'.
    """
    query_fitment = vehicle_fitment.strip().lower()
    state = load_catalogue_state()
    
    # 1. Direct match on vehicle fitment (exact or substring) or "Universal"
    # Find all items compatible with this vehicle
    compatible_items = []
    for item in state:
        fit = item['vehicle_fitment'].lower()
        if query_fitment in fit or fit == "universal" or "universal" in query_fitment:
            compatible_items.append(item)
            
    # 2. If a part type filter is provided, perform a semantic filter on the compatible items
    if part_type and part_type.strip() != "":
        # We initialize retriever to score the compatible items semantically
        try:
            retriever = CatalogueRetriever()
            # Perform semantic search with a higher K to capture all potential matches
            semantic_matches = retriever.retrieve(f"{part_type} for {vehicle_fitment}", top_k=20)
            semantic_skus = {m['sku']: m['similarity_score'] for m in semantic_matches}
            
            # Filter and sort compatible items based on semantic retrieval scores
            scored_items = []
            for item in compatible_items:
                if item['sku'] in semantic_skus:
                    scored_item = item.copy()
                    scored_item['similarity_score'] = semantic_skus[item['sku']]
                    scored_items.append(scored_item)
            
            # Sort by similarity score descending
            scored_items.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            # Remove score from final outputs to keep it clean for agent
            for item in scored_items:
                item.pop('similarity_score', None)
            return {"parts": scored_items[:10]}
        except Exception as e:
            # Fallback to simple substring match in description/name if RAG fails
            part_term = part_type.lower()
            filtered = []
            for item in compatible_items:
                if part_term in item['name'].lower() or part_term in item['description'].lower() or part_term in item['category'].lower():
                    filtered.append(item)
            return {"parts": filtered[:10]}
            
    return {"parts": compatible_items[:10]}

def create_order(dealer_name: str, items: list) -> dict:
    """
    Place an order for a dealer with line items and quantities. Validates stock and returns structured receipt.

    Args:
        dealer_name: Name of the dealer/business placing the order (e.g. 'ABC Motors').
        items: A list of dicts, each with keys 'sku' (string) and 'quantity' (integer). 
               Example: [{"sku": "BRK-1002", "quantity": 2}, {"sku": "FIL-1001", "quantity": 5}]

    Returns:
        A dictionary representing the structured receipt with order status, details, total price, and item confirmations.
    """
    if not dealer_name or dealer_name.strip() == "":
        return {
            "status": "FAILED",
            "error": "Dealer name is required to place an order."
        }
        
    if not items or not isinstance(items, list) or len(items) == 0:
        return {
            "status": "FAILED",
            "error": "Order must contain at least one item."
        }
        
    state = load_catalogue_state()
    state_dict = {item['sku']: item for item in state}
    
    order_items_receipt = []
    order_total = 0
    stock_sufficient = True
    errors = []
    
    # First pass: Validate all items and quantities
    for order_item in items:
        sku = order_item.get('sku', '').strip().upper()
        try:
            qty = int(order_item.get('quantity', 0))
        except (ValueError, TypeError):
            qty = 0
            
        if qty <= 0:
            errors.append(f"Invalid quantity {qty} for SKU {sku}. Must be greater than 0.")
            stock_sufficient = False
            continue
            
        if sku not in state_dict:
            errors.append(f"SKU {sku} not found in catalogue.")
            stock_sufficient = False
            order_items_receipt.append({
                "sku": sku,
                "quantity": qty,
                "status": "INVALID_SKU"
            })
            continue
            
        item = state_dict[sku]
        available_stock = item['stock']
        price = item['price_inr']
        
        if available_stock < qty:
            errors.append(f"Insufficient stock for SKU {sku}. Requested: {qty}, Available: {available_stock}.")
            stock_sufficient = False
            order_items_receipt.append({
                "sku": sku,
                "quantity": qty,
                "available_stock": available_stock,
                "status": "OUT_OF_STOCK"
            })
        else:
            subtotal = price * qty
            order_total += subtotal
            order_items_receipt.append({
                "sku": sku,
                "name": item['name'],
                "quantity": qty,
                "unit_price": price,
                "subtotal": subtotal,
                "status": "CONFIRMED"
            })
            
    # Second pass: If everything is in stock and valid, commit changes and save state
    if stock_sufficient:
        for order_item in items:
            sku = order_item['sku'].strip().upper()
            qty = int(order_item['quantity'])
            state_dict[sku]['stock'] -= qty
            
        save_catalogue_state(state)
        
        return {
            "order_id": f"ORD-{uuid.uuid4().hex[:8].upper()}",
            "dealer_name": dealer_name,
            "status": "SUCCESS",
            "items": order_items_receipt,
            "total_price_inr": order_total
        }
    else:
        return {
            "dealer_name": dealer_name,
            "status": "FAILED",
            "errors": errors,
            "items": order_items_receipt
        }
