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
- **Specs and design docs:** **Never commit** design specs, implementation plans, or brainstorming write-ups. Keep them only under gitignored paths (e.g. `docs/superpowers/`, `docs/data-cleaning-agent/`). Do not add tracked `docs/**` spec trees or copy specs into the codebase for version control. If the user wants a spec shared, they paste or attach it; agents do not commit specs.
- When in doubt whether a change is "small," treat it as large: plan first, then implement after confirmation.
- Run `pytest -q` after changes and share results.
- Confirm before installing new dependencies.
- Never write secrets, always use environment variables.
- Always follow the git-workflow-and-versioning skill when committing code or managing branches.
- Use the PostgreSQL MCP Server tools when querying over creating new code yourself.
- When asked for visualizations, prefer making jupyter notebooks over scripts that generate raw html.

## Testing
- Use pytest for unit tests.
- Keep unit tests short, easy to read, and small in scope.
- Use descriptive names for each test.