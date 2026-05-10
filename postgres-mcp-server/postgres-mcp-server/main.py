import os
from typing import Dict, List, TypedDict

import keyring
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class DBConfig(TypedDict):
    host: str
    port: int
    user: str
    password: str
    dbname: str


def _db_config() -> DBConfig:
    """Get database configuration from environment variables, in a typed dict to prevent typechecking errors."""
    db_user = _require_env("DB_USER")
    password = keyring.get_password("futureproof_ds_db", db_user)
    if password is None or password == "":
        raise RuntimeError(
            "Database password not found in keyring "
            "(service 'futureproof_ds_db', username = DB_USER)."
        )
    return {
        "host": _require_env("DB_HOST"),
        "port": int(_require_env("DB_PORT")),
        "user": db_user,
        "password": password,
        "dbname": _require_env("DB_NAME"),
    }


# Initializes your MCP server instance. It's used to register your tools.
mcp = FastMCP("postgres-server")  # type: ignore

DB_CONFIG = _db_config()


def _connect() -> psycopg.Connection:
    """Helper function to connect to the database."""

    # Typechecking does not like globbing a dict into function arguments, so we destructure it manually.
    return psycopg.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        dbname=DB_CONFIG["dbname"],
    )


@mcp.tool()
async def execute_sql(query: str) -> List[Dict]:
    """Execute a SQL query and return rows as dicts (column name → value)."""
    with _connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query)  # type: ignore - This is usually inadvisable but will run as a read only user.
            rows = cur.fetchall()
    return rows


@mcp.tool()
async def list_tables() -> List[str]:
    """Return base table names: `public` tables unqualified, `marts` as `marts.table_name`."""
    sql = """
        SELECT CASE
            WHEN table_schema = 'public' THEN table_name
            ELSE table_schema || '.' || table_name
        END
        FROM information_schema.tables
        WHERE table_schema IN ('public', 'marts')
          AND table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = [r[0] for r in cur.fetchall()]
    return rows


def _parse_table_identifier(table: str) -> tuple[str, str]:
    """Resolve `table`, `schema.table`, or `"schema"."table"` to (schema, table_name)."""
    name = table.strip().strip('"')
    if "." in name:
        schema_part, table_part = name.split(".", 1)
        return schema_part.strip().strip('"'), table_part.strip().strip('"')
    return "public", name


@mcp.tool()
async def get_schema(table: str) -> List[Dict]:
    """Return column names and types for a given table (`name` or `schema.name`)."""
    schema, table_name = _parse_table_identifier(table)
    sql = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = %s
        ORDER BY ordinal_position
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (table_name, schema))
            rows = [{"column": r[0], "type": r[1]} for r in cur.fetchall()]
    return rows


def main():
    # Run MCP server using stdio transport for AI assistant integration
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
