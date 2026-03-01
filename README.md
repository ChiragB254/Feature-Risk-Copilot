# Wealthsimple Feature Risk Copilot

An AI-powered system that lets a Wealthsimple PM paste a feature idea and instantly get financial impact simulation + fairness risk flags, enabling informed launch decisions without requiring technical expertise.

## Architecture

This system uses a **Sequential 4-Agent Pipeline** to enforce deterministic outputs, combined with a Pandas simulation engine.

```text
User Input (Feature Description)
        ↓
1. Spec Parser Agent
        ↓
Financial Simulation Engine (Pandas - data/customers.csv)
        ↓
2. Simulation Validator Agent
        ↓
3. Risk Analysis Agent
        ↓
4. Compliance Checklist Agent
        ↓
Streamlit Dashboard (Tabs & Results)
        ↓
Human Decision Engine (APPROVE / REVIEW / BLOCK)
```

**Note on AI Providers:** 
This application is designed to run locally via **Ollama** or extremely fast via **Groq**. 
*We use the `openai` Python package purely as a universal client because both Groq and Ollama provide OpenAI-compatible API endpoints. No actual OpenAI API keys or OpenAI models are used in this project.*

## Quick Start Guide

### 1. Setup Environment

Ensure you have Python 3.10+ installed.

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

Before running the app, generate the synthetic customer database for the simulation engine:

```bash
python src/data_generator.py
```
This will create `data/customers.csv` containing 5,000 synthetic records with statistical distributions representing realistic balances and activity.

### 3. Add API Keys (If using Groq)

If you intend to use the Groq provider for blazing-fast inference, create a `.env` file in the root of the project:

```env
GROQ_API_KEY="gsk_..."
```

*(If you are using Ollama, ensure your local Ollama engine is running. No API key is required).*

### 4. Launch the App

Run the Streamlit frontend:

```bash
streamlit run app.py
```

The application will be accessible in your browser at `http://localhost:8501`. 

## Built Components
- `src/models.py`: Strongly-typed Pydantic schemas validating all data flows and LLM outputs.
- `src/data_generator.py`: Reproducible synthetic user generator.
- `src/simulator.py`: Deterministic financial simulation engine calculating bounds and concentration risks. 
- `src/agents/`: Contains the 4 sequential AI logic agents (Spec Parser, Validator, Risk Analyzer, Compliance Checker).
- `app.py`: Full interactive UI dashboard pipeline for parsing, simulating, risk-scoring, and making decisions.
