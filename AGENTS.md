# AGENTS.md

## Project Guidelines
- For **small** edits (localized fix, a handful of lines in one file, obvious typo), a short numbered plan in chat is enough before editing.
- For **larger** work—roughly **multiple files**, a **new feature**, **refactors**, or **on the order of tens of lines or more** across the change—**switch to Plan mode first**, write the plan there, and **wait for agreement** before implementing. Do **not** apply large or wide-ranging diffs without the user having seen and accepted the plan.
- Keep changes minimal and avoid overengineering.
- Remove temporary or large data files, never commit datasets.

## Conventions
- Python: PEP8 style, Ruff linting, NumPy-style docstrings.
- Use pytest for testing, prefer pandas/NumPy/scikit-learn for analysis, plotly and dash for visualization.
- Jupyter notebooks must remain reproducible: no hard-coded paths, use relative paths.

## Agent Instructions
- When in doubt whether a change is "small," treat it as large: plan first, then implement after confirmation.
- Run `pytest -q` after changes and share results.
- Confirm before installing new dependencies.
- Never write secrets, always use environment variables.
- Always follow the git-workflow skill when committing code or managing branches.
- Use the PostgreSQL MCP Server tools when querying over creating new code yourself.
- When asked for visualizations, prefer making jupyter notebooks over scripts that generate raw html.

## Testing
- Use pytest for unit tests.
- Keep unit tests short, easy to read, and small in scope.
- Use descriptive names for each test.