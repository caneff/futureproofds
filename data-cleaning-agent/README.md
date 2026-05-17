# Data Cleaning Agent

An AI-powered data cleaning agent that uses LangChain and LangGraph to produce a JSON **CleaningPlan**, then runs a fixed hybrid pandas pipeline (no generated Python or `exec`).

## How It Works

The agent follows this workflow:

1. **Analyze**: Summarizes your dataset (dtypes, missingness, detection flags).
2. **Plan**: The LLM returns a validated `CleaningPlan` (step skips, protected columns, coerce/impute lists).
3. **Execute**: Python runs steps in fixed order via `run_cleaning_pipeline`.
4. **Retry**: On validation or runtime errors, the LLM corrects the plan (up to 3 attempts).

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

Run from the repo root:

```bash
uv run streamlit run data-cleaning-agent/app.py
```

Then:

1. Upload your CSV file
2. Click **Generate cleaning plan**, review the plan JSON and step list
3. Click **Apply Cleaning**
4. Download the cleaned dataset

### Python API

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LightweightDataCleaningAgent(model=llm)

df = pd.read_csv("your_data.csv")
agent.invoke_agent(source_df=df)
cleaned_df = agent.get_data_cleaned()
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

**Optional: custom instructions**

```python
agent.invoke_agent(
    source_df=df,
    user_instructions="Protect country; drop columns with more than 30% missing values",
)
```

**Generate plan only, apply later**

```python
agent.generate_cleaning_plan(source_df=df, user_instructions="Protect country")
plan = agent.get_cleaning_plan()
result = agent.execute_stored_cleaning(source_df=df)
```

## Project Structure

```
data-cleaning-agent/
├── data_cleaning_agent/
│   ├── cleaning_plan.py          # CleaningPlan schema
│   ├── cleaning_pipeline.py      # Fixed-order pipeline
│   ├── plan_generation.py        # LLM plan JSON parse/generate/fix
│   ├── data_cleaning_agent.py    # LangGraph agent
│   ├── prompts/
│   │   ├── data_cleaning_plan.md
│   │   └── data_cleaning_plan_fix.md
│   └── utils.py                  # DataFrame summary for prompts
├── app.py                        # Streamlit interface
└── README.md
```

Prompts live under `data_cleaning_agent/prompts/`. The hybrid step order is defined in `pipeline_steps.py` and implemented in `cleaners.py`.

Dependencies and the lockfile live at the repo root (`pyproject.toml` and `uv.lock`).
