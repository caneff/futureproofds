# Data cleaning prompts (index)

The agent loads **two** prompt files for generation and one for error correction.
**Do not duplicate** pipeline steps, hard constraints, or plan JSON rules in this
file. Edit the linked files so what the model sees and what humans read stay the
same source of truth.

## Runtime prompts

| Step | File | Role |
|------|------|------|
| 1 | [data_cleaning_code_only.md](./data_cleaning_code_only.md) | Full default **pipeline** (steps 1–12), hard constraints, synthetic row id (`__agent_row_id__`) rules for Streamlit alignment, step 9 imputation rules, **Python-only** output. |
| 2 | [data_cleaning_plan_from_code.md](./data_cleaning_plan_from_code.md) | **JSON cleaning plan** schema and rules; the app appends the finalized Python from step 1 after this template is formatted (so braces in code never pass through `str.format`). |
| Fix | [data_cleaning_fix.md](./data_cleaning_fix.md) | Correct failing cleaner code; the app **re-runs** step 2 on the latest Python after a successful fix. |

This file is **not** a LangChain template and contains no `{placeholder}` fields.
