from pathlib import Path
from datetime import datetime, timedelta, date
from decimal import Decimal
import uuid
import random

from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import concat, lit, col, regexp_replace, trim
from pyspark.sql.types import (
    ArrayType, BooleanType, DecimalType, DoubleType, IntegerType,
    LongType, MapType, StringType, StructField, StructType, TimestampType,
    DateType,
)

spark = SparkSession.builder.getOrCreate()


# Columns kept from the raw sf-airbnb listings, in order.
_LISTING_COLUMNS = ["id", "name", "neighbourhood", "neighbourhood_cleansed", "summary", "price", "room_type"]


def _finalize_listings(df, table_name: str) -> None:
    """Apply the shared listings transform and write to a Delta table.

    Takes a DataFrame already narrowed to ``_LISTING_COLUMNS``, normalizes
    ``price`` to a double, builds the ``listing_source_information`` text
    column, keeps 50 rows, and overwrites ``table_name``. Shared by both the
    Delta-share (:func:`process_csv_data`) and bundled-CSV
    (:func:`process_csv_file`) loaders so the transform stays identical.
    """
    df = (
        df
        # Make price numeric: remove $ and , then cast to double (empty strings become nulls when cast)
        .withColumn("price", regexp_replace(trim(col("price")), "[$,]", "").cast("double"))
        .withColumn(
            "listing_source_information",
            concat(
                lit("ID of the property: "), col("id"),
                lit("\nName of the property: "), col("name"),
                lit("\nSummary of the property: "), col("summary"),
            ),
        )
        .limit(50)
    )
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)


def process_csv_data(
    databricks_share_name: str,
    table_name: str
) -> None:
    """Load and process CSV data from a Delta share into a Delta table."""

    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .load(f"/Volumes/{databricks_share_name}/v01/sf-listings/sf-airbnb.csv")
        .select(*_LISTING_COLUMNS)
    )
    _finalize_listings(df, table_name)

    # Example for setting up Genie-specific table
    # df.select(["id", "price"]).write.mode("overwrite").saveAsTable(table_name+"_genie")


def process_csv_file(
    csv_path: str | Path,
    table_name: str
) -> None:
    """Load and process a bundled CSV file into a Delta table.

    The local-file counterpart to :func:`process_csv_data`. Reads a CSV that
    ships in the course's ``data/`` folder (a workspace file) rather than a
    Delta share, so a course can bundle its own dataset. Serverless Spark can't
    read a ``/Workspace`` file path directly, so we load it with pandas and hand
    it to Spark; the dataset is tiny (~50 rows kept). The transforms match the
    Delta-share path.

    Gated in the orchestrator by ``data.csv_file``; :func:`process_csv_data`
    remains for the Delta-share path (``data.databricks_share_name``).
    """
    import pandas as pd

    pdf = pd.read_csv(
        csv_path,
        dtype=str,          # keep everything as strings; Spark casts price below
        keep_default_na=False,
    )[_LISTING_COLUMNS]

    _finalize_listings(spark.createDataFrame(pdf), table_name)


# ---------------------------------------------------------------------------
# Synthetic AI Gateway system tables
# ---------------------------------------------------------------------------

def create_synthetic_gateway_tables(catalog_name: str) -> None:
    """Create synthetic versions of the AI Gateway system tables.

    Builds ``<catalog_name>.`system`.external_model_spend`` (150 rows) and
    ``<catalog_name>.`system`.usage`` (60 rows) with entirely fabricated data
    — fake account IDs, ``@example.com`` emails, private-range IPs, and
    placeholder endpoint names.  The ``usage`` table includes a realistic mix
    of HTTP status codes (200, 429, 500, 400, 401, 503).

    Gated in the orchestrator by the ``synthetic_gateway_tables`` config flag.

    Parameters
    ----------
    catalog_name : str
        The lab-user catalog (e.g. ``labuser_first_last``).
    """
    schema_fqn = f"{catalog_name}.`system`"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_fqn}")
    print(f"Schema ready: {catalog_name}.system")

    _create_external_model_spend(schema_fqn)
    _create_usage(schema_fqn)


# -- helpers (private) ------------------------------------------------------

def _create_external_model_spend(schema_fqn: str) -> None:
    """Create synthetic external_model_spend table (150 rows)."""
    random.seed(42)

    FAKE_ACCOUNT_ID   = "demo0000-0000-0000-0000-000000000001"
    FAKE_WORKSPACE_ID = "1000000000000001"
    FAKE_EP_IDS       = [str(uuid.UUID(int=i)) for i in range(1, 6)]
    EP_NAMES          = ["openai-prod", "claude-analytics", "gemini-search", "gpt4o-finance", "claude-hr"]
    PROV_MODELS       = [
        ("openai",    "gpt-4o"),
        ("openai",    "gpt-4o-mini"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("anthropic", "claude-3-haiku-20240307"),
        ("google",    "gemini-1.5-pro"),
    ]
    TEAMS    = ["data-engineering", "analytics", "finance", "hr", "product"]
    TIERS    = ["standard", "paygo", "enterprise"]
    SVC_ACCT = "svc-demo@example.com"

    spend_schema = StructType([
        StructField("record_id",        StringType()),
        StructField("account_id",       StringType()),
        StructField("workspace_id",     StringType()),
        StructField("usage_date",       DateType()),
        StructField("usage_start_time", TimestampType()),
        StructField("usage_end_time",   TimestampType()),
        StructField("ingestion_date",   DateType()),
        StructField("usage_metadata", StructType([
            StructField("provider",         StringType()),
            StructField("model",            StringType()),
            StructField("endpoint_id",      StringType()),
            StructField("endpoint_name",    StringType()),
            StructField("destination_id",   StringType()),
            StructField("destination_name", StringType()),
        ])),
        StructField("custom_tags", StructType([
            StructField("endpoint_tags", MapType(StringType(), StringType())),
            StructField("request_tags",  MapType(StringType(), StringType())),
        ])),
        StructField("usage_unit",     StringType()),
        StructField("usage_quantity", DecimalType(38, 18)),
        StructField("pricing_metadata", StructType([
            StructField("service_tier", StringType()),
            StructField("long_context", BooleanType()),
        ])),
        StructField("identity_metadata", StructType([
            StructField("run_by", StringType()),
            StructField("run_as", StringType()),
        ])),
    ])

    rows = []
    base_date = date(2026, 8, 1)
    for day_offset in range(30):
        d = base_date + timedelta(days=day_offset)
        for j, (provider, model_id) in enumerate(PROV_MODELS):
            hour     = (j * 4) % 24
            ts_start = datetime(d.year, d.month, d.day, hour, 0, 0)
            ts_end   = datetime(d.year, d.month, d.day, min(hour + 1, 23), 59, 59)
            ep_idx   = j % len(FAKE_EP_IDS)
            qty      = Decimal(str(round(random.uniform(0.001, 2.5), 6)))
            rows.append(Row(
                record_id        = str(uuid.uuid4()),
                account_id       = FAKE_ACCOUNT_ID,
                workspace_id     = FAKE_WORKSPACE_ID,
                usage_date       = d,
                usage_start_time = ts_start,
                usage_end_time   = ts_end,
                ingestion_date   = d + timedelta(days=1),
                usage_metadata   = Row(
                    provider         = provider,
                    model            = model_id,
                    endpoint_id      = FAKE_EP_IDS[ep_idx],
                    endpoint_name    = EP_NAMES[ep_idx],
                    destination_id   = f"dest-{provider}-001",
                    destination_name = f"{provider}-dest",
                ),
                custom_tags = Row(
                    endpoint_tags = {"team": TEAMS[j % len(TEAMS)]},
                    request_tags  = {"project": "demo-lab"},
                ),
                usage_unit       = "USD",
                usage_quantity   = qty,
                pricing_metadata = Row(
                    service_tier = TIERS[j % len(TIERS)],
                    long_context = False,
                ),
                identity_metadata = Row(
                    run_by = SVC_ACCT,
                    run_as = SVC_ACCT,
                ),
            ))

    df = spark.createDataFrame(rows, schema=spend_schema)
    df.createOrReplaceTempView("_spend_temp")
    spark.sql(f"""
      CREATE OR REPLACE TABLE {schema_fqn}.external_model_spend
      COMMENT 'Synthetic table — mirrors system.ai_gateway.external_model_spend schema. No real data.'
      AS SELECT * FROM _spend_temp
    """)
    print(f"  ✅ external_model_spend created: {df.count()} rows")


def _create_usage(schema_fqn: str) -> None:
    """Create synthetic usage table (60 rows) with realistic HTTP status codes."""
    random.seed(99)

    FAKE_ACCOUNT_ID   = "demo0000-0000-0000-0000-000000000001"
    FAKE_WORKSPACE_ID = "1000000000000001"
    FAKE_EP_IDS       = [str(uuid.UUID(int=i)) for i in range(1, 6)]
    EP_NAMES          = ["openai-prod", "claude-analytics", "gemini-search", "gpt4o-finance", "claude-hr"]
    PROV_MODELS       = [
        ("openai",    "gpt-4o"),
        ("openai",    "gpt-4o-mini"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("anthropic", "claude-3-haiku-20240307"),
        ("google",    "gemini-1.5-pro"),
    ]
    TEAMS  = ["data-engineering", "analytics", "finance", "hr", "product"]
    TIERS  = ["standard", "paygo", "enterprise"]
    USERS  = [
        "alice@example.com", "bob@example.com", "carol@example.com",
        "dave@example.com",  "eve@example.com",
    ]

    dest_schema = StructType([
        StructField("name",            StringType()),
        StructField("traffic_percent", DoubleType()),
        StructField("type",            StringType()),
    ])
    attempt_schema = StructType([
        StructField("priority",       StringType()),
        StructField("action",         StringType()),
        StructField("destination",    StringType()),
        StructField("destination_id", StringType()),
        StructField("status_code",    IntegerType()),
        StructField("error_code",     StringType()),
        StructField("latency_ms",     LongType()),
        StructField("start_time",     TimestampType()),
        StructField("end_time",       TimestampType()),
    ])

    usage_schema = StructType([
        StructField("account_id",        StringType()),
        StructField("workspace_id",      StringType()),
        StructField("request_id",        StringType()),
        StructField("schema_version",    IntegerType()),
        StructField("endpoint_id",       StringType()),
        StructField("endpoint_name",     StringType()),
        StructField("endpoint_tags",     MapType(StringType(), StringType())),
        StructField("endpoint_metadata", StructType([
            StructField("creator",           StringType()),
            StructField("creation_time",     TimestampType()),
            StructField("last_updated_time", TimestampType()),
            StructField("destinations",      ArrayType(dest_schema)),
            StructField("inference_table",   StringType()),
            StructField("fallbacks",         ArrayType(dest_schema)),
        ])),
        StructField("event_time",             TimestampType()),
        StructField("latency_ms",             LongType()),
        StructField("time_to_first_byte_ms",  LongType()),
        StructField("destination_type",       StringType()),
        StructField("destination_name",       StringType()),
        StructField("destination_id",         StringType()),
        StructField("destination_model",      StringType()),
        StructField("requester",              StringType()),
        StructField("requester_type",         StringType()),
        StructField("ip_address",             StringType()),
        StructField("url",                    StringType()),
        StructField("user_agent",             StringType()),
        StructField("api_type",               StringType()),
        StructField("request_tags",           MapType(StringType(), StringType())),
        StructField("input_tokens",           LongType()),
        StructField("output_tokens",          LongType()),
        StructField("total_tokens",           LongType()),
        StructField("token_details", StructType([
            StructField("cache_read_input_tokens",    LongType()),
            StructField("cache_creation_input_tokens", LongType()),
            StructField("output_reasoning_tokens",    LongType()),
        ])),
        StructField("response_content_type",  StringType()),
        StructField("status_code",            IntegerType()),
        StructField("routing_information", StructType([
            StructField("attempts", ArrayType(attempt_schema)),
        ])),
        StructField("invocation_id",          StringType()),
        StructField("invocation_metadata", StructType([
            StructField("source",       StringType()),
            StructField("service_tier", StringType()),
        ])),
        StructField("service_type",  StringType()),
        StructField("service_id",    StringType()),
        StructField("service_name",  StringType()),
        StructField("service_tags",  MapType(StringType(), StringType())),
        StructField("mcp_metadata",  StructType([
            StructField("tool_name",       StringType()),
            StructField("server_type",     StringType()),
            StructField("client_name",     StringType()),
            StructField("json_rpc_method", StringType()),
        ])),
    ])

    rows = []
    base_dt = datetime(2026, 8, 1, 0, 0, 0)
    for i in range(60):
        j                  = i % len(PROV_MODELS)
        provider, model_id = PROV_MODELS[j]
        ep_idx             = j % len(FAKE_EP_IDS)
        ep_name            = EP_NAMES[ep_idx]
        ep_id              = FAKE_EP_IDS[ep_idx]
        event_ts           = base_dt + timedelta(hours=i * 4 + random.randint(0, 3), minutes=random.randint(0, 59))
        latency            = random.randint(200, 3000)
        ttfb               = random.randint(80, latency)
        in_toks            = random.randint(50, 4000)
        out_toks           = random.randint(50, 2000)
        user               = USERS[i % len(USERS)]
        ip                 = f"10.0.{(i // 10) % 10}.{(i * 7 + 1) % 256}"

        # Realistic HTTP status distribution
        _roll = random.random()
        if _roll < 0.75:
            status, err_code = 200, None
        elif _roll < 0.85:
            status, err_code = 429, "rate_limit_exceeded"
        elif _roll < 0.90:
            status, err_code = 500, "internal_error"
        elif _roll < 0.94:
            status, err_code = 400, "invalid_request"
        elif _roll < 0.97:
            status, err_code = 401, "unauthorized"
        else:
            status, err_code = 503, "service_unavailable"

        end_ts = event_ts + timedelta(milliseconds=latency)

        rows.append(Row(
            account_id        = FAKE_ACCOUNT_ID,
            workspace_id      = FAKE_WORKSPACE_ID,
            request_id        = str(uuid.uuid4()),
            schema_version    = 1,
            endpoint_id       = ep_id,
            endpoint_name     = ep_name,
            endpoint_tags     = {"team": TEAMS[j % len(TEAMS)]},
            endpoint_metadata = Row(
                creator           = "demo-admin@example.com",
                creation_time     = datetime(2026, 1, 15, 9, 0, 0),
                last_updated_time = datetime(2026, 6, 1, 12, 0, 0),
                destinations      = [Row(name=model_id, traffic_percent=100.0, type=provider)],
                inference_table   = None,
                fallbacks         = [],
            ),
            event_time            = event_ts,
            latency_ms            = latency,
            time_to_first_byte_ms = ttfb,
            destination_type      = "EXTERNAL_MODEL",
            destination_name      = f"{provider}-dest",
            destination_id        = f"dest-{provider}-001",
            destination_model     = model_id,
            requester             = user,
            requester_type        = "USER" if i % 3 != 0 else "SERVICE_PRINCIPAL",
            ip_address            = ip,
            url                   = f"https://api.example.com/gateway/{ep_name}/chat/completions",
            user_agent            = "python-requests/2.31.0",
            api_type              = "chat/completions",
            request_tags          = {"project": "demo-lab", "env": "training"},
            input_tokens          = in_toks,
            output_tokens         = out_toks,
            total_tokens          = in_toks + out_toks,
            token_details         = Row(
                cache_read_input_tokens     = 0,
                cache_creation_input_tokens = 0,
                output_reasoning_tokens     = 0,
            ),
            response_content_type = "application/json",
            status_code           = status,
            routing_information   = Row(
                attempts = [Row(
                    priority       = "primary",
                    action         = "route",
                    destination    = f"{provider}-dest",
                    destination_id = f"dest-{provider}-001",
                    status_code    = status,
                    error_code     = err_code,
                    latency_ms     = latency,
                    start_time     = event_ts,
                    end_time       = end_ts,
                )]
            ),
            invocation_id       = str(uuid.uuid4()),
            invocation_metadata = Row(source="api", service_tier=TIERS[j % len(TIERS)]),
            service_type        = "EXTERNAL_MODEL",
            service_id          = ep_id,
            service_name        = ep_name,
            service_tags        = {"team": TEAMS[j % len(TEAMS)]},
            mcp_metadata        = Row(tool_name=None, server_type=None, client_name=None, json_rpc_method=None),
        ))

    df = spark.createDataFrame(rows, schema=usage_schema)
    df.createOrReplaceTempView("_usage_temp")
    spark.sql(f"""
      CREATE OR REPLACE TABLE {schema_fqn}.usage
      COMMENT 'Synthetic table — mirrors system.ai_gateway.usage schema. No real data.'
      AS SELECT * FROM _usage_temp
    """)
    print(f"  ✅ usage created: {df.count()} rows")


def create_sample_labels(table_name: str = "tko_sample_labels") -> None:

    schema = StructType([
        StructField("eval_id", StringType(), True),
        StructField("request", StringType(), True),
        StructField("guidelines", ArrayType(StringType()), True),
        StructField("items", StringType(), True),     # optional
        StructField("metadata", StringType(), True),  # optional (JSON string)
        StructField("tags", StringType(), True),
    ])

    data = [
        # KA-only count (attributes from PDF)
        (
            "1",
            "How many properties have the Victorian home style?",
            [
                "Use only the Properties PDF to determine membership; do not use the price table.",
                "Return a single integer: count of properties with home_style = 'Victorian'.",
                "If the style is missing or ambiguous for a record, exclude it and note exclusions.",
                "Provide citations to the exact PDF page(s) used."
            ],
            None,
            "{}", 
            "supervisor_ka_genie"
        ),

        # KA → Genie average price conditioned on attribute from PDF
        (
            "2",
            "What's the average price for all properties with 2 bedrooms?",
            [
                "Resolve the set of property IDs using only the Properties PDF (bedrooms == 2); provide PDF citations.",
                "If zero IDs are found, abstain: respond 'No matching properties found' and ask for a clarifying filter.",
                "Query Genie for prices only for the KA-supplied IDs.",
                "Exclude null or non-numeric prices; do not impute.",
                "Compute arithmetic mean and report sample size n.",
                "Preserve units exactly as stored; do not add currency symbols unless present.",
                "If duplicate price rows exist per ID, include valid prices and report duplicate count.",
                "Output format: 'Average price: <numeric>' on first line; 'n: <count>' on second; 'Notes: <brief summary>' on third; 'Citations: <PDF pages>' on fourth.",
                "If the ID list length ≤ 20, include IDs; otherwise state 'IDs omitted for brevity.'"
            ],
            None,
            "{}", 
            "supervisor_ka_genie"
        ),

        # Another KA → Genie example with a style attribute
        (
            "3",
            "What is the average price for all properties with Victorian home style?",
            [
                "Use only the Properties PDF to select IDs where home_style = 'Victorian'; provide PDF citations.",
                "If KA returns zero IDs, abstain and ask the user to refine the filter.",
                "Use Genie to fetch prices for those IDs and compute the arithmetic mean; report n.",
                "Exclude null/non-numeric prices; do not impute.",
                "If conflicting sources are detected, prioritize PDF for attributes and Genie for prices; flag conflict in Notes.",
                "Output format and ID listing rules same as above."
            ],
            None,
            "{}", 
            "supervisor_ka_genie"
        ),
    ]

    df = spark.createDataFrame(data, schema=schema)

    (df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(table_name))