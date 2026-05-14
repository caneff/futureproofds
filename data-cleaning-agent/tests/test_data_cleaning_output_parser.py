"""Tests for :class:`data_cleaning_agent.utils.DataCleaningOutputParser`."""

import pytest
from data_cleaning_agent.utils import DataCleaningOutputParser


@pytest.mark.unit
def test_parser_extracts_code_and_valid_json():
    text = """
Here is the solution.

```python
def data_cleaner(source_df):
    return source_df.copy()
```

```json
{"columns": [{"name": "a", "actions": ["drop"]}], "row_ops": [], "notes": ""}
```
"""
    p = DataCleaningOutputParser()
    out = p.parse(text)
    assert "def data_cleaner" in out["code"]
    assert out["cleaning_plan"] is not None
    assert out["cleaning_plan"]["columns"][0]["name"] == "a"


@pytest.mark.unit
def test_parser_missing_json_yields_none_plan():
    text = "```python\nx = 1\n```\n"
    out = DataCleaningOutputParser().parse(text)
    assert out["code"].strip() == "x = 1"
    assert out["cleaning_plan"] is None


@pytest.mark.unit
def test_parser_invalid_json_yields_none_plan():
    text = """```python
def f(d):
    return d
```

```json
{ not json
```
"""
    out = DataCleaningOutputParser().parse(text)
    assert "def f" in out["code"]
    assert out["cleaning_plan"] is None


@pytest.mark.unit
def test_parser_json_array_root_yields_none_plan():
    text = """```python
def f(d):
    return d
```

```json
["a", "b"]
```
"""
    out = DataCleaningOutputParser().parse(text)
    assert out["cleaning_plan"] is None


@pytest.mark.unit
def test_parser_python_fence_ignores_triple_backticks_inside_line():
    text = """```python
def data_cleaner(source_df):
    example = "```"
    return source_df.copy()
```

```json
{"columns": [], "row_ops": [], "notes": ""}
```
"""
    out = DataCleaningOutputParser().parse(text)
    assert 'example = "```"' in out["code"]
    assert "return source_df.copy()" in out["code"]
    assert out["cleaning_plan"] is not None
