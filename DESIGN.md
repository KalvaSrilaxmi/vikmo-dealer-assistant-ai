# VIKMO Auto-Parts Dealer Assistant & Forecasting — Technical Design Document

This document explains the technical architecture, design decisions, and methodologies selected for both the agentic conversational assistant (Part A & B) and the demand forecasting system (Part C) for the VIKMO Intern Assignment.

---

## Part A: Dealer Assistant (Core Architecture)

### 1. Retrieval (RAG) Approach & Indexing Choices
* **Why FAISS?**:
  FAISS (Facebook AI Similarity Search) provides dense vector search capabilities with sub-millisecond retrieval times. Since the catalogue has 600 SKUs, keeping index execution entirely local on CPU reduces external latency to nearly zero, requires zero infrastructure overhead, and facilitates clean, self-contained deployment.
* **Why Embeddings?**:
  We use the `sentence-transformers/all-MiniLM-L6-v2` embedding model. It encodes text blocks into 384-dimensional dense vectors. Using dense embeddings enables semantic lookup (e.g. matching queries for "tyre" to catalogue items categorised under "Tyres & Tubes" or matching "spark plugs" to specific brand parts), solving vocabulary mismatch problems.
* **Retrieval Unit (Document Construction)**:
  Each catalogue item is compiled into a single text document detailing all attributes:
  ```
  Product Name: [name] | SKU: [sku] | Category: [category] | Brand: [brand] | Vehicle Fitment: [vehicle_fitment] | Description: [description] | Price: [price] INR | Stock Available: [stock] units
  ```
  Adding explicit attribute keys ("Product Name:", "Vehicle Fitment:", etc.) helps the model recognize field boundaries during encoding and ensures semantic similarity captures metadata terms.
* **Similarity Metric**:
  We normalize embedding vectors using L2 norm and construct a FAISS Inner Product index (`IndexFlatIP`). The inner product of L2-normalized vectors is mathematically equivalent to Cosine Similarity.

### 2. Agent & Tool Calling Design
* **LLM Model**:
  Google **Gemini 1.5 Flash** (or Gemini 2.5 Flash) is chosen for its fast response speeds, large context window, zero cost under the Google AI Studio free tier, and robust native tool-calling capabilities.
* **Prompt Design & System Instruction**:
  The agent is initialized with strict operational instructions defining its role, tone, grounding rules, and guardrails:
  1. *Persona*: Auto-parts assistant for dealers.
  2. *Grounding*: State product, price, and stock info strictly based on database retrieval or tool outputs. Do not make up SKUs.
  3. *Guardrails*: Decline general out-of-domain requests (weather, news, cooking recipes, general chit-chat).
  4. *Clarification*: Prompt user for specifications when query parameters are ambiguous (e.g. "I need brake pads" -> ask "For which vehicle make and model?").
* **Tool Invocation Routing**:
  We provide Gemini with three native Python function signatures:
  1. `check_stock(sku: str)`: Looks up stock level, price, and details.
  2. `find_parts_by_vehicle(vehicle_fitment: str, part_type: str)`: Filters compatibility list. If a part type is supplied, filters search semantically.
  3. `create_order(dealer_name: str, items: list)`: Accepts a structured list of SKU-quantity dicts, validates stock availability, deducts inventory, calculates the grand total, and outputs a success confirmation.
* **Inventory Stock Persistence**:
  When orders are successfully created, the assistant deducts stock and commits changes to a state file (`assistant/catalogue_state.json`). This ensures that stock modifications persist across turns and different queries in a chat session.

---

## Part B: Evaluation Framework

* **Test Scenarios**:
  We compiled a suite of 12 test queries in `eval/eval_set.json` categorized across:
  - Happy Path (standard product searches)
  - Ambiguous (queries lacking vehicle/type details)
  - Stock Checks (queries targeting specific SKU availability)
  - Vehicle Fitment (compatibility searches)
  - Order Creation (simple and multi-item orders)
  - Out-of-Domain (guardrail tests)
* **Metrics**:
  1. *Tool Selection Accuracy*: Evaluates if the agent invoked the expected function (or called none when appropriate).
  2. *Grounding and Hallucination Checks*: A regex analyzer inspects final responses for SKU patterns. Any SKU returned that does not exist in the catalogue triggers a grounding failure.
* **Failure Analysis Insights**:
  Integrating semantic retrieval with native tool calling creates a robust hybrid system:
  - If a user performs a search, RAG fetches the top 4 candidates, which are injected directly into the LLM context.
  - If a specific action is requested, Gemini calls the appropriate python tool.
  - State management ensures that stock levels remain synchronized.

---

## Part C: Demand Forecasting (Part B Bonus)

### 1. Feature Engineering & Preprocessing
* **Lag Features**: Since we want to forecast up to 4 weeks ahead, the minimum lag is set to 4 weeks (e.g., Lags 4, 5, 6, 7, 8, 12, 52). This guarantees that predicting week $t$ uses information only up to week $t-4$, ensuring **zero future data leakage**.
* **Rolling Features**: 4-week and 8-week rolling averages are computed over the 4-week shifted demand values to smooth short-term noise.
* **Calendar Features**: Week of year and month are extracted to capture seasonal fluctuations.
* **SKU Feature Representation**: We use One-Hot Encoding on the `sku` column to represent the 30 distinct models. This allows the Random Forest model to learn specific base demand levels per SKU.
* **Imputation**: To prevent discarding the first 52 rows due to the 52-week lag, missing values are filled with the historical median of that SKU calculated *strictly* from training data.

### 2. Validation Scheme & Leakage Prevention
* **Train/Test Split**: We use a temporal split:
  - **Training Set**: First 74 weeks (`2024-12-16` to `2026-05-11`).
  - **Validation Set (Test Window)**: Last 4 weeks (`2026-05-18` to `2026-06-08`).
* **Zero Leakage**:
  - Imputation statistics (SKU medians) are computed using training data *only* and mapped to test records.
  - Features do not use lags 1, 2, or 3, making the forecasting model fully capable of predicting 4 weeks ahead recursively or directly without needing future sales.

### 3. Model Performance Results

| Model / Baseline | Overall MAE (lower is better) | Overall MAPE (lower is better) |
|---|---|---|
| Naive (Last Value) | 9.3583 | 38.38% |
| Moving Average (4-week) | 8.1250 | 41.01% |
| Seasonal Naive (52-week) | 9.9833 | 65.08% |
| **Random Forest Regressor (Ours)** | **5.6429** | **38.31%** |

* **Analysis**:
  Our Random Forest Regressor achieves a **Mean MAE of 5.6429 units**, outperforming all baselines. This represents a **39.7% reduction in error** relative to the Naive baseline and a **30.5% reduction** relative to the 4-week Moving Average baseline. It successfully captures the calendar trends, promo flag benefits, and shared patterns across product series.
