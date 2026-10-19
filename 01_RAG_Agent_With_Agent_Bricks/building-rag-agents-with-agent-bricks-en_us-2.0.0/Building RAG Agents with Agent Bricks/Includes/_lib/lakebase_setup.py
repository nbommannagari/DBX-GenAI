# Includes/_lib/lakebase_setup.py
"""Lakebase Search setup — connect to pre-created project, load pre-chunked data, build indexes."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound


# ---------------------------------------------------------------------------
# Connection factory builder
# ---------------------------------------------------------------------------

def lb_connect_factory(
    w: WorkspaceClient,
    endpoint_name: str,
    host: str,
    database: str,
    user_email: str,
) -> Callable[[], psycopg.Connection]:
    """Return a zero-argument callable that opens an authenticated psycopg connection.

    Each call mints a fresh short-lived OAuth credential so the returned
    factory remains valid across the 1-hour token window.
    """
    def _connect() -> psycopg.Connection:
        credential = w.postgres.generate_database_credential(endpoint=endpoint_name)
        return psycopg.connect(
            host=host,
            dbname=database,
            user=user_email,
            password=credential.token,
            sslmode="require",
            autocommit=True,
        )
    return _connect


# ---------------------------------------------------------------------------
# Project provisioning
# ---------------------------------------------------------------------------

def _provision_project(
    w: WorkspaceClient,
    project_id: str,
    branch: str,
) -> tuple[str, str]:
    """Look up the student's pre-created Lakebase project and return (endpoint_name, host).

    Attempts a direct ``get_project`` by the computed project slug.  On
    failure, lists all visible projects so the error message shows what
    names actually exist (helpful for debugging naming mismatches).
    """
    try:
        w.postgres.get_project(name=f"projects/{project_id}")
        print(f"  ✓ Found Lakebase project: {project_id}")
    except NotFound:
        # Genuinely not created (404). Any OTHER error (auth, network, API
        # outage) propagates untouched, so a transient failure never
        # masquerades as "project not found — go create it".
        # List available projects so the student can see the mismatch
        try:
            projects = list(w.postgres.list_projects())
            available = sorted(p.name.rsplit('/', 1)[-1] for p in projects)
        except Exception:
            available = ["(unable to list projects)"]
        avail_str = (
            "\n".join(f"    • {n}" for n in available) if available else "    (none)"
        )
        raise RuntimeError(
            f"\n{'=' * 68}\n"
            f"  Lakebase project not found: {project_id}\n\n"
            f"  Available projects:\n{avail_str}\n\n"
            f"  Please create the project BEFORE running this setup:\n"
            f"    1. Click the waffle menu (top-right) → Lakebase Postgres\n"
            f"    2. Click ⊕ New Project\n"
            f"    3. Name it:  {project_id}\n"
            f"    4. Click Create\n"
            f"    5. Enable Lakebase Search in project Settings\n"
            f"    6. Re-run this cell\n"
            f"{'=' * 68}"
        )

    endpoints = list(w.postgres.list_endpoints(
        parent=f"projects/{project_id}/branches/{branch}"
    ))
    endpoint_name = endpoints[0].name
    host = endpoints[0].status.hosts.host
    print(f"  Endpoint : {endpoint_name}")
    print(f"  Host     : {host}")
    return endpoint_name, host


def _detect_lakebase_search(
    connect_fn: Callable[[], psycopg.Connection],
) -> bool:
    """Return True if the Lakebase Search extensions can be installed.

    Lakebase Search is a GA feature that must be enabled in the project
    settings UI before the extensions become available.  This function
    attempts to create them; if they fail with a shared_preload_libraries
    error it falls back to plain pgvector.
    """
    try:
        with connect_fn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS lakebase_text;")
        print("  Lakebase Search extensions installed (lakebase_vector, lakebase_text).")
        return True
    except Exception as exc:
        exc_msg = str(exc)
        if "shared_preload_libraries" in exc_msg:
            print("  Lakebase Search not enabled on this project — using pgvector + HNSW/GIN fallback.")
            print("  Tip: Enable Lakebase Search in project Settings to use lakebase_ann / lakebase_bm25.")
        else:
            print(f"  Warning: Could not install Lakebase Search extensions: {exc}")
            print("  Falling back to pgvector + HNSW/GIN.")
        # Ensure plain pgvector is available
        with connect_fn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        return False


# ---------------------------------------------------------------------------
# Schema + data loading
# ---------------------------------------------------------------------------

def _create_table_and_load(
    connect_fn: Callable[[], psycopg.Connection],
    json_path: str | Path,
    table_name: str,
    embedding_dimensions: int,
) -> int:
    """Create the doc_chunks table, load JSON records, and return the row count."""
    json_path = Path(json_path)
    with open(json_path) as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records from {json_path.name}")

    with connect_fn() as conn:
        with conn.cursor() as cur:
            # Create table
            cur.execute(f"""
                DROP TABLE IF EXISTS {table_name};
                CREATE TABLE {table_name} (
                    chunk_id           TEXT PRIMARY KEY,
                    chunk_pos_id       INTEGER NOT NULL,
                    filename           TEXT,
                    modification_time  TIMESTAMPTZ,
                    doc_uri            TEXT,
                    parser             TEXT,
                    splitter           TEXT,
                    chunk_size         INTEGER,
                    chunk_overlap      INTEGER,
                    chunk_text         TEXT,
                    doc_type           TEXT,
                    metadata           JSONB,
                    parsed_doc_summary TEXT,
                    embedding_model    TEXT,
                    embedding          vector({embedding_dimensions}),
                    chunk_tsvector     tsvector
                );
            """)
            print(f"  Table {table_name} created.")

            # Bulk insert
            insert_sql = f"""
                INSERT INTO {table_name} (
                    chunk_id, chunk_pos_id, filename, modification_time, doc_uri,
                    parser, splitter, chunk_size, chunk_overlap, chunk_text,
                    doc_type, metadata, parsed_doc_summary, embedding_model,
                    embedding, chunk_tsvector
                ) VALUES (
                    %(chunk_id)s, %(chunk_pos_id)s, %(filename)s, %(modification_time)s, %(doc_uri)s,
                    %(parser)s, %(splitter)s, %(chunk_size)s, %(chunk_overlap)s, %(chunk_text)s,
                    %(doc_type)s, %(metadata)s, %(parsed_doc_summary)s, %(embedding_model)s,
                    %(embedding)s::vector, %(chunk_tsvector)s::tsvector
                )
            """
            for r in records:
                params = {
                    "chunk_id":           r["chunk_id"],
                    "chunk_pos_id":       r["chunk_pos_id"],
                    "filename":           r["filename"],
                    "modification_time":  r["modificationTime"],
                    "doc_uri":            r["doc_uri"],
                    "parser":             r["parser"],
                    "splitter":           r["splitter"],
                    "chunk_size":         r["chunk_size"],
                    "chunk_overlap":      r["chunk_overlap"],
                    "chunk_text":         r["chunk_text"],
                    "doc_type":           r["doc_type"],
                    "metadata":           r["metadata"],
                    "parsed_doc_summary": r["parsed_doc_summary"],
                    "embedding_model":    r["embedding_model"],
                    "embedding":          r["__db_chunk_text_pgvector"],
                    "chunk_tsvector":     r["__db_chunk_text_tsvector"],
                }
                cur.execute(insert_sql, params)

            cur.execute(f"SELECT count(*) FROM {table_name};")
            row_count = cur.fetchone()[0]
            print(f"  Inserted {row_count} rows into {table_name}.")
            return row_count


def _build_indexes(
    connect_fn: Callable[[], psycopg.Connection],
    table_name: str,
    use_lakebase_search: bool = True,
) -> None:
    """Build vector and full-text indexes on the table.

    When *use_lakebase_search* is True, builds lakebase_ann (IVF + RaBitQ)
    and lakebase_bm25 (BM25 + Block-Max WAND) indexes.  Otherwise falls back
    to HNSW and GIN.
    """
    with connect_fn() as conn:
        with conn.cursor() as cur:
            if use_lakebase_search:
                vec_method, vec_label = "lakebase_ann", "lakebase_ann"
                ts_method,  ts_label  = "lakebase_bm25", "lakebase_bm25"
            else:
                vec_method, vec_label = "hnsw", "HNSW"
                ts_method,  ts_label  = "gin",  "GIN"

            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_emb_idx
                  ON {table_name} USING {vec_method} (embedding vector_cosine_ops);
            """)
            print(f"  {vec_label} index built on {table_name}.embedding (cosine).")

            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_ts_idx
                  ON {table_name} USING {ts_method} (chunk_tsvector);
            """)
            print(f"  {ts_label} index built on {table_name}.chunk_tsvector.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def setup_lakebase_search(
    lakebase_config: Dict[str, Any],
    user_email: str,
    includes_dir: str | Path,
) -> Dict[str, Any]:
    """Connect to a pre-created Lakebase project, load document chunks, and build search indexes.

    Parameters
    ----------
    lakebase_config : dict
        The ``lakebase_search`` section from config YAML.  Expected keys:
        ``project_suffix``, ``branch``, ``database``, ``table_name``,
        ``data_file``, ``embedding_dimensions``.
    user_email : str
        Current user's email (used for project naming and Postgres auth).
    includes_dir : str | Path
        Absolute path to the Includes/ directory (data_file is resolved relative to it).

    Returns
    -------
    dict
        ``lb_connect``       – callable returning an authenticated psycopg connection
        ``project_id``       – the Lakebase project ID
        ``endpoint_name``    – full endpoint resource name
        ``host``             – Postgres hostname
        ``table_name``       – name of the loaded table
        ``row_count``        – number of rows inserted
    """
    includes_dir = Path(includes_dir)
    w = WorkspaceClient()

    # Derive project slug from user email.  Lakebase normalises project
    # names to lowercase alphanumerics + hyphens, so we mirror that here.
    local_part = user_email.lower().split("@")[0]
    cleaned = local_part.replace(".", "-").replace("_", "-")
    suffix = lakebase_config.get("project_suffix", "lb-search")
    project_id = f"{cleaned}-{suffix}"

    branch   = lakebase_config.get("branch", "production")
    database = lakebase_config.get("database", "databricks_postgres")
    table    = lakebase_config.get("table_name", "doc_chunks")
    dims     = int(lakebase_config.get("embedding_dimensions", 1024))
    data_file = lakebase_config.get("data_file", "data/lakebase-data.json")

    # 1. Look up the student's pre-created Lakebase project
    endpoint_name, host = _provision_project(w, project_id, branch)

    # 2. Build connection factory
    connect_fn = lb_connect_factory(w, endpoint_name, host, database, user_email)

    # 3. Verify connectivity
    with connect_fn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print(f"  Connected: {cur.fetchone()[0][:60]} ...")

    # 4. Detect Lakebase Search availability (installs extensions if possible)
    use_lakebase_search = _detect_lakebase_search(connect_fn)

    # 5. Load data
    json_path = includes_dir / data_file
    row_count = _create_table_and_load(connect_fn, json_path, table, dims)

    # 6. Build indexes (lakebase_ann/bm25 when available, else hnsw/gin)
    _build_indexes(connect_fn, table, use_lakebase_search)

    return {
        "lb_connect":    connect_fn,
        "project_id":    project_id,
        "endpoint_name": endpoint_name,
        "host":          host,
        "table_name":    table,
        "row_count":     row_count,
    }
