# LangGraph Workflow Example

A simple example demonstrating how to build workflows with LangGraph for data processing.

## How It Works

The workflow follows these steps:

1. **Load Data** - Reads the CSV file into a pandas DataFrame
2. **Summarize Data** - Generates comprehensive summary including:
   - Statistical description (`.describe()`)
   - Dataset info (`.info()`)
   - Explicit missing value counts
3. **LLM Reasoning** - Uses GPT-4o-mini to analyze the summary and decide which action to take
4. **Conditional Routing** - Routes to appropriate cleaning node based on LLM decision
5. **Data Cleaning** - Can execute:
   - **Handle Missing Values** - Fills numeric missing values with column means
   - **Remove Outliers** - Removes outliers using IQR (Interquartile Range) method
6. **Describe Data** - Generates statistical summary of cleaned data
7. **Output Results** - Prints the action taken and final summary

## Setup

Dependencies are managed at the repo root with [uv](https://docs.astral.sh/uv/). Python 3.14 or higher is required.

### Windows (PowerShell)

1. **Install uv** (if not already installed):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   After installation, restart your terminal.

2. **Install dependencies** (from the repo root):
   ```powershell
   uv sync
   ```
   This will also install a matching Python interpreter if one isn't found.

3. **Set up your OpenAI API key**:
   ```powershell
   copy .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key: `OPENAI_API_KEY=sk-your-key-here`

### macOS/Linux

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies** (from the repo root):
   ```bash
   uv sync
   ```

3. **Set up your OpenAI API key**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key: `OPENAI_API_KEY=sk-your-key-here`

## Running the Example

From the repo root:
```bash
uv run python simple-data-cleaning-langgraph-workflow/workflows/simple_clean_data_workflow.py
```

The workflow will:
1. Save a workflow graph visualization to `outputs/workflow_graph.png`
2. Load data from `data/missing_and_outliers.csv` (you can change this in the script)
3. Use an LLM to analyze the data and decide which cleaning action to take
4. Execute the appropriate cleaning steps
5. Display the results

## Project Structure

```
.
├── README.md
├── .env.example
├── .gitignore
│
├── data/
│   ├── missing.csv              # Data with only missing values
│   ├── outliers.csv             # Data with only outliers
│   └── missing_and_outliers.csv # Data with both issues
│
├── workflows/
│   └── simple_clean_data_workflow.py   # Main workflow implementation
│
└── outputs/
    └── .gitkeep                 # Generated files (graphs, reports)
```

**Folder Organization:**
- `data/` - Sample CSV files with different data quality issues
- `workflows/` - LangGraph workflow scripts
- `outputs/` - Generated outputs (visualizations, reports)
