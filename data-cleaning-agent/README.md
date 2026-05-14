# Data Cleaning Agent

An AI-powered data cleaning agent that automatically cleans messy datasets using LangChain and LangGraph. The agent uses an LLM to generate and execute Python code for common data cleaning tasks like handling missing values, removing duplicates, and dropping low-quality columns.

## How It Works

The agent follows a simple workflow:
1. **Analyze**: Examines your dataset structure and identifies data quality issues
2. **Generate**: Uses an LLM to create custom Python cleaning code based on the data
3. **Execute**: Runs the generated code to clean your data
4. **Retry**: Automatically fixes errors if the generated code fails (up to 3 attempts)

This approach combines the flexibility of LLMs with the reliability of pandas operations.

## Setup

Dependencies are managed at the repo root with [uv](https://docs.astral.sh/uv/). Python 3.14 or higher is required.

### Prerequisites

- **Python 3.14 or higher** (uv will install one automatically if needed)
- **uv** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install uv** (if not already installed):

   **Windows (PowerShell)**:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   **macOS/Linux**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   After installation, restart your terminal.

2. **Install dependencies** (from the repo root):
   ```bash
   uv sync
   ```

   This will resolve and install all dependencies from `uv.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:

   **Windows**:
   ```powershell
   copy data-cleaning-agent\.env.example data-cleaning-agent\.env
   ```

   **macOS/Linux**:
   ```bash
   cp data-cleaning-agent/.env.example data-cleaning-agent/.env
   ```

   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

## Usage

### Streamlit Web Interface

The easiest way to use the agent is through the web interface (run from the repo root):

```bash
uv run streamlit run data-cleaning-agent/app.py
```

Then:
1. Upload your CSV file
2. Click "Clean Data"
3. Download the cleaned dataset

### Python API

For programmatic use or integration into data pipelines:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Initialize the agent with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LightweightDataCleaningAgent(model=llm)

# Load your messy data
df = pd.read_csv("your_data.csv")

# Run the cleaning agent
agent.invoke_agent(source_df=df)

# Get the cleaned dataset
cleaned_df = agent.get_data_cleaned()

# Save or use the cleaned data
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

**Optional: Provide custom instructions**

```python
# End-user cleaning instructions (columns named here are protected)
agent.invoke_agent(
    source_df=df,
    user_instructions="Remove columns with more than 30% missing values and standardize date formats",
)

# Synthetic row id rules for alignment are embedded in ``data_cleaning.md``; no separate host prompt slot.
```

## Project Structure

```
data-cleaning-agent/
├── data_cleaning_agent/
│   ├── __init__.py
│   ├── data_cleaning_agent.py  # Main agent class
│   ├── prompts/
│   │   └── data_cleaning.md    # LLM prompt + default cleaning pipeline
│   └── utils.py                # Utility functions
├── app.py                      # Streamlit interface
└── README.md
```

The default 14-step pipeline that runs when no `user_instructions` are
provided is defined in
[`data_cleaning_agent/prompts/data_cleaning.md`](data_cleaning_agent/prompts/data_cleaning.md).

Dependencies and the lockfile live at the repo root (`pyproject.toml` and `uv.lock`).
