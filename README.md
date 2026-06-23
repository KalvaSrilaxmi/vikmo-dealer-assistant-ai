# VIKMO Auto-Parts Dealer Assistant & Demand Forecasting

This repository contains the complete implementation for the VIKMO AI/ML Intern Assignment, structured into two core sections:
1. **Part A: Conversational Dealer Assistant & Evaluation Framework**: An agentic chat bot grounded in a 600 SKU product catalogue using FAISS RAG, SentenceTransformers, and Gemini API Tool Calling, backed by an automated evaluation test runner.
2. **Part B: Demand Forecasting System**: A supervised machine learning regressor (Random Forest) predicting 4-week-ahead sales demand across 30 SKUs, outperforming naive baselines.

---

## 1. Project Directory Structure

```
├── README.md                 # Project introduction, setup instructions, and quickstart guide
├── DESIGN.md                 # Detailed design decisions, rationale, and evaluation reports
├── requirements.txt          # Python dependencies (google-generativeai, faiss-cpu, sentence-transformers, etc.)
├── assistant/
│   ├── index.py              # SentenceTransformer embedding & FAISS index creation logic
│   ├── retriever.py          # Semantic search utility querying FAISS index
│   ├── tools.py              # Implementation of check_stock, find_parts_by_vehicle, create_order
│   ├── memory.py             # Class to manage multi-turn chat history context
│   ├── agent.py              # Agent loop orchestrating Gemini API, tool routing, and memory
│   └── cli.py                # Command Line Interface (CLI) to interact with the dealer assistant
├── eval/
│   ├── eval_set.json         # Structured evaluation set (queries, expected tools, parameters)
│   ├── evaluator.py          # Evaluator runner executing the test suite and computing metrics
│   └── eval_results.md       # Generated report summarizing metrics and logs
└── forecasting/
    ├── data_prep.py          # Feature engineering (lags, calendar, rollings, imputation)
    ├── baseline.py           # Computes Naive and Moving Average baseline metrics
    ├── model.py              # Fits the Random Forest model and performs predictions
    └── forecast.py           # Main forecasting executor outputting MAE & MAPE reports
```

---

## 2. Tech Stack

- **Language**: Python 3.13+
- **LLM Abstraction Layer**: Custom resilient multi-provider abstraction supporting:
  - **Google Gemini** (via native `google-generativeai` SDK)
  - **Groq Cloud API** (via HTTP endpoints using Llama 3)
  - **Ollama** (via local model endpoints like `llama3`)
  - **Local Demo Mode** (keyless, offline rule-based agent emulator)
- **Vector Database**: FAISS (local `faiss-cpu`)
- **Embedding Model**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Forecasting Regressor**: Scikit-Learn (`RandomForestRegressor`)
- **Data Engineering**: Pandas & NumPy

---

## 3. Setup and Installation

### Step 1: Install Dependencies
Create a virtual environment, activate it, and install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
The assistant features a resilient abstraction layer. You can configure multiple providers and fallbacks.

Key configuration variables:
- `LLM_PROVIDER`: Set to `gemini`, `groq`, `ollama`, or `demo` (default: `gemini`).
- `FALLBACK_PROVIDERS`: Comma-separated order of fallback providers if the primary fails (default: `groq,demo`).
- `DEMO_MODE`: Set to `true` to run offline/keyless local Demo Mode immediately.
- `GEMINI_API_KEY`: API Key for Google Gemini.
- `GROQ_API_KEY`: API Key for Groq Cloud API.
- `GROQ_MODEL`: Groq model name (default: `llama3-8b-8192`).
- `OLLAMA_MODEL`: Ollama model name (default: `llama3`).
- `OLLAMA_HOST`: Local Ollama service host (default: `http://localhost:11434`).

You can set these in your environment:
- **Windows (PowerShell)**:
  ```powershell
  $env:LLM_PROVIDER="gemini"
  $env:GEMINI_API_KEY="your_api_key"
  ```
- **Linux/macOS**:
  ```bash
  export LLM_PROVIDER="gemini"
  export GEMINI_API_KEY="your_api_key"
  ```

To run without any API keys or external connections (ideal for review and validation), configure the assistant to run in **Demo Mode**:
- **Windows (PowerShell)**:
  ```powershell
  $env:DEMO_MODE="true"
  ```
- **Linux/macOS**:
  ```bash
  export DEMO_MODE="true"
  ```

Alternatively, create a file named `.env` in the root directory of the workspace:
```env
DEMO_MODE=true
# Or configure keys
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_key
# FALLBACK_PROVIDERS=groq,demo
```
The agent loads `.env` variables automatically.

### Step 3: Build the FAISS Vector Index
Before running the assistant, run the indexer script to load `catalogue.json` and build the local FAISS index:
```bash
python assistant/index.py
```
This builds `assistant/faiss_index.bin` and `assistant/index_metadata.json`.

---

## 4. How to Run the Applications

### A. Run the Interactive Chat Assistant (CLI)
Start the CLI chat interface to talk directly to the agent:
```bash
python assistant/cli.py
```
- Type **`reset`** in the prompt to clear history and reset inventory stock levels.
- Type **`exit`** or **`quit`** to exit the program.

### B. Run the Automated Agent Evaluation
Run the test suite against the evaluation cases:
```bash
python -m eval.evaluator
```
This executes the 12 scenarios from `eval/eval_set.json` and updates the `eval/eval_results.md` file with a performance report.

### C. Run the Demand Forecasting Pipeline
Run the baseline comparisons and the Random Forest forecasting model:
```bash
python -m forecasting.forecast
```
This computes forecasting metrics (MAE and MAPE) per SKU, prints the overall performance comparison table, and verifies if the Random Forest model successfully beats all baselines.

---

## 5. Example Assistant Interactions

* **RAG / Vehicle Fitment lookup**:
  - *User*: "Do you have brake pads for a Bajaj Pulsar 150?"
  - *Assistant*: Calls `find_parts_by_vehicle` and displays the matching products, pricing, and stock.
* **Ambiguous Query Resolving**:
  - *User*: "I need tyres."
  - *Assistant*: "I'd be happy to help with tyres. For which vehicle make/model do you need them? Also, let me know if you are looking for front or rear tyres."
* **Direct Stock Checking**:
  - *User*: "Check stock of BRK-1036."
  - *Assistant*: Calls `check_stock(sku="BRK-1036")` and reports that 3 units are available for ₹2,130.
* **Order Creation**:
  - *User*: "Place an order for 10 units of BRK-1002 for ABC Motors."
  - *Assistant*: Calls `create_order` and prints a structured receipt:
    ```markdown
    Order ID: ORD-XXXXXX
    Dealer Name: ABC Motors
    Status: SUCCESS
    Items:
      - SKU: BRK-1002 | Quantity: 10 | Unit Price: ₹X | Subtotal: ₹X (CONFIRMED)
    Total Price: ₹X
    ```
* **Guardrail Refusal**:
  - *User*: "Can you give me a recipe for chocolate chip cookies?"
  - *Assistant*: "I apologize, but I can only assist with auto-parts inquiries, stock checks, and order processing."

---

## 6. Project Assumptions & Design Justifications

For a deep dive into our design choices, please refer to [DESIGN.md](file:///c:/Users/SHIVANI/OneDrive/Desktop/Custom%20elements%20Project/DESIGN.md).
Key assumptions made:
1. **State Persistence**: Successful orders decrease stock count. The state is written to `assistant/catalogue_state.json` and persists for the chat CLI.
2. **Forecasting Window**: We hold out the last 4 weeks of weekly records (`2026-05-18` to `2026-06-08`) as the test window.
3. **Data Leakage Prevention**: We restrict forecasting lag features to $\ge 4$ weeks, meaning the model makes predictions without knowing future target values.
