"""Unit tests for ``postgres-mcp-server`` ``main`` module.

These tests exercise the pure helpers (``_require_env``, ``_db_config``,
``_parse_table_identifier``) and the async MCP tools (``execute_sql``,
``list_tables``, ``get_schema``) by mocking ``_connect`` so that no real
PostgreSQL connection is attempted. Ideally it would be an actual in-memory
test database connection, but that is too complex to set up for the purposes of
this project.
"""

from __future__ import annotations

from typing import Tuple
from unittest.mock import MagicMock

import main
import pytest

# ---------------------------------------------------------------------------
# _require_env
# ---------------------------------------------------------------------------


def test_require_env_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUTUREPROOF_TEST_VAR", "hello")

    assert main._require_env("FUTUREPROOF_TEST_VAR") == "hello"


def test_require_env_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUTUREPROOF_TEST_VAR", raising=False)

    with pytest.raises(RuntimeError, match="FUTUREPROOF_TEST_VAR"):
        main._require_env("FUTUREPROOF_TEST_VAR")


def test_require_env_raises_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUTUREPROOF_TEST_VAR", "")

    with pytest.raises(RuntimeError, match="FUTUREPROOF_TEST_VAR"):
        main._require_env("FUTUREPROOF_TEST_VAR")


# ---------------------------------------------------------------------------
# _db_config
# ---------------------------------------------------------------------------


def _set_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_USER", "alice")
    monkeypatch.setenv("DB_NAME", "demo")


def test_db_config_returns_typed_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_db_env(monkeypatch)
    monkeypatch.setattr(main.keyring, "get_password", lambda *a, **k: "s3cret")

    cfg = main._db_config()

    assert cfg == {
        "host": "localhost",
        "port": 5432,
        "user": "alice",
        "password": "s3cret",
        "dbname": "demo",
    }


def test_db_config_raises_when_keyring_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_db_env(monkeypatch)
    monkeypatch.setattr(main.keyring, "get_password", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="keyring"):
        main._db_config()


def test_db_config_raises_when_keyring_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_db_env(monkeypatch)
    monkeypatch.setattr(main.keyring, "get_password", lambda *a, **k: "")

    with pytest.raises(RuntimeError, match="keyring"):
        main._db_config()


def test_db_config_raises_when_db_user_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "demo")

    with pytest.raises(RuntimeError, match="DB_USER"):
        main._db_config()


# ---------------------------------------------------------------------------
# _parse_table_identifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("users", ("public", "users")),
        ("public.users", ("public", "users")),
        ("marts.foo", ("marts", "foo")),
        ('"marts"."foo"', ("marts", "foo")),
        ("  users  ", ("public", "users")),
        ("  marts.foo  ", ("marts", "foo")),
        ('"users"', ("public", "users")),
    ],
)
def test_parse_table_identifier(raw: str, expected: Tuple[str, str]) -> None:
    assert main._parse_table_identifier(raw) == expected


# ---------------------------------------------------------------------------
# Async MCP tools (mocked _connect)
# ---------------------------------------------------------------------------


def _patch_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[MagicMock, MagicMock]:
    """Patch ``main._connect`` to yield mock conn/cursor context managers."""
    mock_cursor = MagicMock(name="cursor")
    mock_conn = MagicMock(name="conn")

    cursor_cm = MagicMock(name="cursor_cm")
    cursor_cm.__enter__.return_value = mock_cursor
    cursor_cm.__exit__.return_value = False
    mock_conn.cursor.return_value = cursor_cm

    conn_cm = MagicMock(name="conn_cm")
    conn_cm.__enter__.return_value = mock_conn
    conn_cm.__exit__.return_value = False

    monkeypatch.setattr(main, "_connect", lambda: conn_cm)
    return mock_conn, mock_cursor


@pytest.mark.asyncio
async def test_execute_sql_returns_dict_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_conn, mock_cursor = _patch_connect(monkeypatch)
    rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
    mock_cursor.fetchall.return_value = rows

    result = await main.execute_sql("SELECT id, name FROM users")

    assert result == rows
    mock_cursor.execute.assert_called_once_with("SELECT id, name FROM users")
    mock_conn.cursor.assert_called_once_with(row_factory=main.dict_row)


@pytest.mark.asyncio
async def test_list_tables_returns_string_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, mock_cursor = _patch_connect(monkeypatch)
    mock_cursor.fetchall.return_value = [("users",), ("marts.orders",)]

    result = await main.list_tables()

    assert result == ["users", "marts.orders"]
    mock_cursor.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_schema_public_table_binds_schema_public(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, mock_cursor = _patch_connect(monkeypatch)
    mock_cursor.fetchall.return_value = [("id", "integer"), ("name", "text")]

    result = await main.get_schema("users")

    assert result == [
        {"column": "id", "type": "integer"},
        {"column": "name", "type": "text"},
    ]
    args, _kwargs = mock_cursor.execute.call_args
    assert args[1] == ("users", "public")


@pytest.mark.asyncio
async def test_get_schema_qualified_table_binds_marts_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, mock_cursor = _patch_connect(monkeypatch)
    mock_cursor.fetchall.return_value = []

    result = await main.get_schema("marts.users")

    assert result == []
    args, _kwargs = mock_cursor.execute.call_args
    assert args[1] == ("users", "marts")
