# AGENTS.md

## Project Guidelines
- Always propose a plan in numbered steps before editing code.
- Keep changes minimal and avoid overengineering.
- Remove temporary or large data files, never commit datasets.

## Conventions
- Python: PEP8 style, Ruff linting, NumPy-style docstrings.
- Use pytest for testing, prefer pandas/NumPy/scikit-learn for analysis, plotly and dash for visualization.
- Jupyter notebooks must remain reproducible: no hard-coded paths, use relative paths.

## Agent Instructions
- Run `pytest -q` after changes and share results.
- Confirm before installing new dependencies.
- Never write secrets, always use environment variables.
- Always follow the git-workflow skill when committing code or managing branches.
- Use the PostgreSQL MCP Server tools when querying over creating new code yourself.
- When asked for visualizations, prefer making jupyter notebooks over scripts that generate raw html.
