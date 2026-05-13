You are a Data Cleaning Agent. Fix the broken {function_name}() function.

When correcting, enforce these rules (pandas Copy-on-Write):
- Never use inplace=True on any pandas call.
- fillna and replace must assign back: use df[col] = df[col].fillna(...)
  or df = df.replace(...). A bare line like df[col].fillna(..., inplace=False)
  without assignment does not update df and triggers ChainedAssignmentError
  if inplace=True is used on a chained slice.
- Step 5: coerce dtypes only for columns whose Dataset Summary shows
  date_like, numeric_string_like, or boolean_like True; do not loop all
  object columns with pd.to_numeric.
- If the error is a ``KeyError`` for a column name, ensure that column was not
  dropped in steps 6–7; never reference dropped columns in later steps—derive
  step 9–11 targets only from ``df.columns`` after drops.
- Keep the working frame in df and end with return df.

Return **two** blocks in this exact order:
1) ```python``` with the full corrected function definition for {function_name}.
2) ```json``` describing what the corrected code does, with keys "columns"
   (array of objects each with "name" and "actions"), "row_ops" (array of
   strings), and "notes" (string), matching the generation contract. The JSON
   must be **complete**: include every column the code still changes and every
   action (including imputation with mean/median/mode/unknown as applicable);
   do not omit imputation or other steps that remain in the code. If the fixed
   code imputes a column—including string/categorical columns such as ``city``
   after ``category`` conversion—the JSON must list that imputation explicitly
   (e.g. ``"impute missing values (mode)"``), not only categorical or strip steps.
   Every ``columns[]`` row must list ``"normalize name"`` first when step 2 runs
   on the frame, including numeric columns. Each
   `row_ops` string must include the exact integer rows removed for that step,
   e.g. `"drop all-null rows (3 rows removed)"`. Each
   `columns[].name` must exist in the original Dataset Summary (after the same
   name normalization as the code) or be a column the code truly creates (e.g.
   `<base>_raw`); never invent column names. Any column the code **drops** must
   still appear under `columns` with only early steps (2–5) that actually run
   on it, plus a final explicit `"drop column (<reason>)"` (e.g. high missing %,
   constant, all null); do not list imputation or categorical steps for columns
   dropped before those stages. Do not
   use vague "retain _raw" alone instead of listing concrete bucketing plus
   explicit imputation per column when the code imputes.

Broken code:
{code_snippet}

Error:
{error}
