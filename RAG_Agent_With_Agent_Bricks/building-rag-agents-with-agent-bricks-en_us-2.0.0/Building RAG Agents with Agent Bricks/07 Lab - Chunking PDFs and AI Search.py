# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Lab - Chunking PDFs and AI Search
# MAGIC #### Duration: 40 minutes
# MAGIC ## Overview
# MAGIC In this lab, you will take a PDF through the **Parse → Prep Search → AI Search** pipeline on your own. You will use `ai_parse_document` to extract text, `ai_prep_search` to chunk it, persist the chunks to a Delta table with Change Data Feed enabled, and create an AI Search index. Finally, you will query the index using three different search modes with `AISearchClient()` from the Databricks AI Search SDK.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lab, you will be able to:
# MAGIC 1. **Parse a PDF and chunk the text** using `ai_parse_document` and `ai_prep_search`
# MAGIC 2. **Write chunked data to a Delta table** with Change Data Feed enabled
# MAGIC 3. **Create an AI Search index** on the `vs_all_demo` endpoint
# MAGIC 4. **Test the index** with similarity, hybrid, and full-text keyword search queries
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Beta Features</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">This notebook uses Databricks Beta Features. While this notebook has been thoroughly tested, it's worth noting that full functionality is not guaranteed.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## REQUIRED - SELECT A COMPUTE ENVIRONMENT
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Select Compute</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Before starting this notebook, select the required compute environment listed below.</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Serverless Compute, Version 6</strong>: How to select an environment version (<a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V6</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Classroom Setup
# MAGIC
# MAGIC Run the following cell to configure your working environment. This sets your `catalog_name` and `schema_name` variables and makes the shared lab assets available to you.

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-07

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Review the Lab Variables
# MAGIC
# MAGIC As a part of the workspace setup, certain assets have been managed for you (volumes, PDF, etc.). You will read the PDF `airbnb_house_rules.pdf` from the shared catalog `dbacademy`, schema `rag_all`, and volume `vol_all`.
# MAGIC
# MAGIC 1. Run the next cell to define the source PDF path, target table names, embedding model, and AI Search endpoint/index names.
# MAGIC
# MAGIC 2. Confirm the printed output shows your own catalog and schema — you will write all lab objects into your personal schema.
# MAGIC
# MAGIC **NOTE:** These variables are referenced throughout the lab. Do not modify them.

# COMMAND ----------

HOUSE_RULES_PDF = "/Volumes/dbacademy/rag_all/vol_all/additional_listing_info/airbnb_house_rules.pdf"
CHUNKS_TABLE = f"{catalog_name}.{schema_name}.house_rules_chunks"

EMBEDDING_MODEL = "databricks-gte-large-en"
VS_ENDPOINT = "vs_all_demo"
VS_INDEX = f"{catalog_name}.{schema_name}.house_rules_index"

print(f"Source PDF:    {HOUSE_RULES_PDF}")
print(f"Chunks table:  {CHUNKS_TABLE}")
print(f"VS Index:      {VS_INDEX}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Parse the PDF

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Extract Elements with `ai_parse_document`
# MAGIC
# MAGIC The `ai_parse_document` function returns a `VARIANT` value describing the layout of a document: its pages, elements, and the text content of each element. To work with individual elements as rows, you explode the `elements` array out of the returned `VARIANT`.
# MAGIC
# MAGIC Complete the following in the TODO cell below:
# MAGIC
# MAGIC 1. Call `ai_parse_document()` on the `content` column, passing `MAP('version', '2.0')` as the options argument.
# MAGIC
# MAGIC 2. Read the source PDF using `READ_FILES()` with the `binaryFile` format.
# MAGIC
# MAGIC 3. Use `variant_explode()` on the `elements` array to produce one row per document element.
# MAGIC
# MAGIC 4. Save the resulting DataFrame as a table named `parsed_house_rules`.
# MAGIC
# MAGIC **NOTE:** This will parse ~300 PDF elements and may take a minute or two to complete.

# COMMAND ----------

Parse the house rules PDF and flatten its elements into rows.
parsed_df = spark.sql(f"""
  WITH parsed AS (
    SELECT
      path,
      <FILL_IN>(content, <FILL_IN>) AS parsed
    FROM READ_FILES('{HOUSE_RULES_PDF}', format => '<FILL_IN>')
  )
  SELECT
    path,
    element.value:type::STRING AS element_type,
    element.value:content::STRING AS content
  FROM parsed,
    LATERAL <FILL_IN>(parsed.parsed:document:elements) AS element
  WHERE element.value:content IS NOT NULL
""")

parsed_df.write.mode("overwrite").saveAsTable(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC parsed_df = spark.sql(f"""
# MAGIC   WITH parsed AS (
# MAGIC     SELECT
# MAGIC       path,
# MAGIC       ai_parse_document(content, MAP('version', '2.0')) AS parsed
# MAGIC     FROM READ_FILES('{HOUSE_RULES_PDF}', format => 'binaryFile')
# MAGIC   )
# MAGIC   SELECT
# MAGIC     path,
# MAGIC     element.value:type::STRING AS element_type,
# MAGIC     element.value:content::STRING AS content
# MAGIC   FROM parsed,
# MAGIC     LATERAL variant_explode(parsed.parsed:document:elements) AS element
# MAGIC   WHERE element.value:content IS NOT NULL
# MAGIC """)
# MAGIC
# MAGIC parsed_df.write.mode("overwrite").saveAsTable("parsed_house_rules")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Inspect the Parsed Elements
# MAGIC
# MAGIC 1. Complete the SQL query below to select all rows from the `parsed_house_rules` table.
# MAGIC
# MAGIC 2. Review the `element_type` column values. Notice the variety of element types (text, section headers, tables, etc.) that `ai_parse_document` identifies.

# COMMAND ----------

# MAGIC %sql
# MAGIC Select all rows from the parsed elements table
# MAGIC SELECT <FILL_IN> FROM <FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```sql
# MAGIC SELECT * FROM parsed_house_rules
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Chunk with `ai_prep_search`

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Generate Chunks
# MAGIC
# MAGIC The `ai_prep_search` function intelligently chunks parsed text. It uses AI models to split text at semantically meaningful boundaries, which produces higher-quality retrieval units than simple fixed-size splitting.
# MAGIC
# MAGIC **IMPORTANT:** `ai_prep_search()` needs the full `ai_parse_document` `VARIANT` per document — not the flattened elements stored in `parsed_house_rules`. This means you must run it against the original parsed output rather than the table you created in Section B.
# MAGIC
# MAGIC Complete the following in the TODO cell below:
# MAGIC
# MAGIC 1. Pass the full `parsed` `VARIANT` column into `ai_prep_search()`.
# MAGIC
# MAGIC 2. Explode `prepped:document:contents` to produce one row per chunk.
# MAGIC
# MAGIC 3. Extract the `chunk_id`, `chunk_position`, `chunk_to_retrieve`, and `chunk_to_embed` fields from each chunk.
# MAGIC
# MAGIC 4. Save the results to a table named `house_rules_chunks`, enabling `mergeSchema`.

# COMMAND ----------

Chunk the parsed document using ai_prep_search and save the results.
chunks_df = spark.sql(f"""
  WITH parsed AS (
    SELECT
      path,
      ai_parse_document(content, MAP('version', '2.0')) AS parsed
    FROM READ_FILES('{HOUSE_RULES_PDF}', format => 'binaryFile')
  ),
  prepped AS (
    SELECT
      path,
      <FILL_IN>(parsed) AS prepped
    FROM parsed
  )
  SELECT
    p.path,
    chunk.value:chunk_id::STRING        AS chunk_id,
    chunk.value:chunk_position::INT     AS chunk_position,
    chunk.value:<FILL_IN>::STRING       AS chunk_to_retrieve,
    chunk.value:<FILL_IN>::STRING       AS chunk_to_embed
  FROM prepped p,
       LATERAL variant_explode(p.prepped:document:<FILL_IN>) AS chunk
""")

chunks_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC chunks_df = spark.sql(f"""
# MAGIC   WITH parsed AS (
# MAGIC     SELECT
# MAGIC       path,
# MAGIC       ai_parse_document(content, MAP('version', '2.0')) AS parsed
# MAGIC     FROM READ_FILES('{HOUSE_RULES_PDF}', format => 'binaryFile')
# MAGIC   ),
# MAGIC   prepped AS (
# MAGIC     SELECT
# MAGIC       path,
# MAGIC       ai_prep_search(parsed) AS prepped
# MAGIC     FROM parsed
# MAGIC   )
# MAGIC   SELECT
# MAGIC     p.path,
# MAGIC     chunk.value:chunk_id::STRING        AS chunk_id,
# MAGIC     chunk.value:chunk_position::INT     AS chunk_position,
# MAGIC     chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
# MAGIC     chunk.value:chunk_to_embed::STRING  AS chunk_to_embed
# MAGIC   FROM prepped p,
# MAGIC        LATERAL variant_explode(p.prepped:document:contents) AS chunk
# MAGIC """)
# MAGIC
# MAGIC chunks_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("house_rules_chunks")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Understanding the Output
# MAGIC
# MAGIC The chunks table contains two columns of particular interest as you build an AI Search index:
# MAGIC
# MAGIC | Column | Purpose | Description |
# MAGIC |---|---|---|
# MAGIC | `chunk_to_retrieve` | Shown to the LLM after retrieval | The raw (or near-raw) chunk text, preserving full content and structure so the model has everything it needs to answer questions. |
# MAGIC | `chunk_to_embed` | Sent to the embedding model | A cleaned, context-enriched string that prepends document-level metadata (title, section headers, page number, captions, etc.) to the chunk text, optimized purely for retrieval quality. |
# MAGIC
# MAGIC Complete the following:
# MAGIC
# MAGIC 1. Fill in the query below to select all rows from `house_rules_chunks`.
# MAGIC
# MAGIC 2. In the first row of the produced table, click the dropdown to expand the `chunk_to_retrieve` and `chunk_to_embed` columns.
# MAGIC
# MAGIC 3. Inspect the results and confirm that the raw text is stored in `chunk_to_retrieve` while `chunk_to_embed` holds the context-enriched text prepared for embeddings. To see the format of `chunk_to_embed`, review the documentation ([AWS](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_prep_search#chunk_to_embed-format) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_prep_search#chunk_to_embed-format) | [GCP](https://docs.databricks.com/gcp/en/sql/language-manual/functions/ai_prep_search#chunk_to_embed-format)).

# COMMAND ----------

# MAGIC %sql
# MAGIC Select all rows from the chunks table
# MAGIC SELECT <FILL_IN> FROM <FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```sql
# MAGIC SELECT * FROM house_rules_chunks
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Write Chunks to Unity Catalog
# MAGIC
# MAGIC You need to persist the chunks in a Delta table with Change Data Feed (CDF) enabled — this is **required** for AI Search to sync automatically. CDF tracks and records all changes (inserts, updates, deletes) to the table, allowing downstream systems to efficiently detect and process data modifications.
# MAGIC
# MAGIC Complete the following in the TODO cell below:
# MAGIC
# MAGIC 1. Add a `chunk_id` column generated with the `uuid()` expression to serve as the primary key for the table and index.
# MAGIC
# MAGIC 2. Alias `chunk_to_embed` as `chunk_text` (the column AI Search will embed) and `chunk_to_retrieve` as `retrieval_text`.
# MAGIC
# MAGIC 3. Write the DataFrame to `CHUNKS_TABLE` in `delta` format using `overwrite` mode with `overwriteSchema` enabled.
# MAGIC
# MAGIC 4. Set the `delta.enableChangeDataFeed` table property to `true`.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Learn more about CDF</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC See the Databricks documentation on Change Data Feed (<a href="https://docs.databricks.com/aws/en/delta/delta-change-data-feed.html" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/tables/features/change-data-feed" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/delta/delta-change-data-feed.html" style="color: #1976d2; text-decoration: underline;">GCP</a>) for a deeper dive.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

Map ai_prep_search outputs into your chunks table schema and write with CDF enabled.
from pyspark.sql.functions import expr, lit, col

chunks_with_id = (
    chunks_df
    .withColumn("chunk_id", expr("<FILL_IN>"))            # PK for the table / index
    .withColumn("source_path", col("path"))
    .select(
        "chunk_id",
        col("<FILL_IN>").alias("chunk_text"),             # for embeddings
        col("<FILL_IN>").alias("retrieval_text"),
        "source_path"
    )
)

chunks_with_id.write.format("delta") \
    .mode("<FILL_IN>") \
    .option("overwriteSchema", "true") \
    .option("<FILL_IN>", "true") \
    .saveAsTable(<FILL_IN>)

spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (<FILL_IN> = true)")

print(f"Chunks table written: {CHUNKS_TABLE}")
display(spark.table(CHUNKS_TABLE))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.sql.functions import expr, lit, col
# MAGIC
# MAGIC # Map ai_prep_search outputs into your chunks table schema
# MAGIC chunks_with_id = (
# MAGIC     chunks_df
# MAGIC     .withColumn("chunk_id", expr("uuid()"))            # PK for the table / index
# MAGIC     .withColumn("source_path", col("path"))
# MAGIC     .select(
# MAGIC         "chunk_id",
# MAGIC         col("chunk_to_embed").alias("chunk_text"),     # for embeddings
# MAGIC         col("chunk_to_retrieve").alias("retrieval_text"),
# MAGIC         "source_path"
# MAGIC     )
# MAGIC )
# MAGIC
# MAGIC chunks_with_id.write.format("delta") \
# MAGIC     .mode("overwrite") \
# MAGIC     .option("overwriteSchema", "true") \
# MAGIC     .option("delta.enableChangeDataFeed", "true") \
# MAGIC     .saveAsTable(CHUNKS_TABLE)
# MAGIC
# MAGIC spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
# MAGIC
# MAGIC print(f"Chunks table written: {CHUNKS_TABLE}")
# MAGIC display(spark.table(CHUNKS_TABLE))
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Create AI Search Index

# COMMAND ----------

# MAGIC %md
# MAGIC ### E1. Verify the Endpoint Is Online
# MAGIC
# MAGIC You will attach an AI Search index to the shared `vs_all_demo` endpoint. Before creating the index, confirm the endpoint is in an `ONLINE` state.
# MAGIC
# MAGIC 1. Run the next cell to poll the endpoint status until it reports `ONLINE`.
# MAGIC
# MAGIC 2. You can also validate the endpoint manually by navigating to **Compute > AI Search** and viewing the **Status** column.
# MAGIC
# MAGIC **NOTE:** As part of the workspace setup, the `vs_all_demo` endpoint should already be ready by the time you reach this point in the lab. This cell is provided for you — no changes needed.

# COMMAND ----------

from databricks.ai_search.client import AISearchClient
import time

client = AISearchClient()

# Verify the endpoint is ready
for _ in range(150):
    try:
        ep = client.get_endpoint(VS_ENDPOINT)
        status = ep.get("endpoint_status", {}).get("state", "")
        if status == "ONLINE":
            print(f"Endpoint '{VS_ENDPOINT}' is ONLINE.")
            break
        print(f"  Endpoint status: {status} - waiting...")
        time.sleep(10)
    except Exception:
        print(f"  Endpoint '{VS_ENDPOINT}' not found yet - waiting...")
        time.sleep(10)
else:
    print("WARNING: Endpoint not ready. The index creation may fail.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Create the Delta Sync Index
# MAGIC
# MAGIC Create a **Delta Sync Index** with **managed embeddings**, meaning Databricks automatically computes embeddings from the source column you designate.
# MAGIC
# MAGIC Complete the following in the TODO cell below:
# MAGIC
# MAGIC 1. Call `create_delta_sync_index()` on the `AISearchClient` object `client`.
# MAGIC
# MAGIC 2. Set `source_table_name` to `CHUNKS_TABLE` and `index_name` to `VS_INDEX`.
# MAGIC
# MAGIC 3. Set `primary_key` to the column that uniquely identifies each chunk.
# MAGIC
# MAGIC 4. Set `embedding_source_column` to the column holding the context-enriched text and `embedding_model_endpoint_name` to `EMBEDDING_MODEL`.
# MAGIC
# MAGIC 5. Use `pipeline_type="TRIGGERED"` so the index syncs on demand rather than continuously.
# MAGIC
# MAGIC **NOTE:** Index creation and the initial sync can take several minutes. The `except` block re-syncs the index if it already exists, so re-running the cell is safe.

# COMMAND ----------

Create the Delta Sync index if it doesn't already exist
try:
    index = client.<FILL_IN>(
        endpoint_name=VS_ENDPOINT,
        source_table_name=<FILL_IN>,
        index_name=<FILL_IN>,
        pipeline_type="<FILL_IN>",
        primary_key="<FILL_IN>",
        embedding_source_column="<FILL_IN>",
        embedding_model_endpoint_name=<FILL_IN>,
        columns_to_sync=["chunk_id", "chunk_text", "source_path"],
    )
    print(f"Index '{VS_INDEX}' created and syncing.")
except Exception as e:
    if "already exists" in str(e):
        index = client.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
        print(f"Index '{VS_INDEX}' already exists. Syncing latest data.")
        index.sync()
    else:
        raise

print(index.describe())

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC # Create the index if it doesn't already exist
# MAGIC try:
# MAGIC     index = client.create_delta_sync_index(
# MAGIC         endpoint_name=VS_ENDPOINT,
# MAGIC         source_table_name=CHUNKS_TABLE,
# MAGIC         index_name=VS_INDEX,
# MAGIC         pipeline_type="TRIGGERED",
# MAGIC         primary_key="chunk_id",
# MAGIC         embedding_source_column="chunk_text",
# MAGIC         embedding_model_endpoint_name=EMBEDDING_MODEL,
# MAGIC         columns_to_sync=["chunk_id", "chunk_text", "source_path"],
# MAGIC     )
# MAGIC     print(f"Index '{VS_INDEX}' created and syncing.")
# MAGIC except Exception as e:
# MAGIC     if "already exists" in str(e):
# MAGIC         index = client.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
# MAGIC         print(f"Index '{VS_INDEX}' already exists. Syncing latest data.")
# MAGIC         index.sync()
# MAGIC     else:
# MAGIC         raise
# MAGIC
# MAGIC print(index.describe())
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Test the Index
# MAGIC
# MAGIC Now that the index is online, test it with three different search modes:
# MAGIC
# MAGIC | Subsection | Search Mode | `query_type` | How It Works |
# MAGIC |---|---|---|---|
# MAGIC | F1 | Similarity Search (Vector ANN) | *default* | Semantic vector similarity using Approximate Nearest Neighbor |
# MAGIC | F2 | Hybrid Search | `hybrid` | Combines vector and BM25 keyword matching via Reciprocal Rank Fusion (RRF) |
# MAGIC | F3 | Full-Text Keyword Search (Beta) | `FULL_TEXT` | BM25 keyword-only matching, no embeddings |

# COMMAND ----------

# MAGIC %md
# MAGIC ### F1. Similarity Search (Vector ANN)
# MAGIC
# MAGIC The default search mode uses Approximate Nearest Neighbor (ANN) vector similarity. The index converts your query text into an embedding and finds the most semantically similar chunks.
# MAGIC
# MAGIC Complete the following in the TODO cell below:
# MAGIC
# MAGIC 1. Retrieve the index with `client.get_index()` using `VS_ENDPOINT` and `VS_INDEX`, then print its detailed state.
# MAGIC
# MAGIC 2. Call `similarity_search()` with the query text *"What are the house rules about pets?"*.
# MAGIC
# MAGIC 3. Return the `chunk_id` and `chunk_text` columns, limited to 3 results.
# MAGIC
# MAGIC **NOTE:** If the index is still syncing, wait a few minutes and re-run the cell.

# COMMAND ----------

Retrieve the index and run a default vector similarity search
index = client.<FILL_IN>(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
status = index.describe().get("status", {}).get("detailed_state")
print(f"Index status: {status}")

results = index.<FILL_IN>(
    query_text="<FILL_IN>",
    columns=[<FILL_IN>],
    num_results=<FILL_IN>,
)
print(f"\n--- F1: Similarity Search (Vector ANN) ---")
for row in results.get("result", {}).get("data_array", []):
    print(f"\n--- Chunk ---")
    print(row[1])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC index = client.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
# MAGIC status = index.describe().get("status", {}).get("detailed_state")
# MAGIC print(f"Index status: {status}")
# MAGIC
# MAGIC results = index.similarity_search(
# MAGIC     query_text="What are the house rules about pets?",
# MAGIC     columns=["chunk_id", "chunk_text"],
# MAGIC     num_results=3,
# MAGIC )
# MAGIC print(f"\n--- F1: Similarity Search (Vector ANN) ---")
# MAGIC for row in results.get("result", {}).get("data_array", []):
# MAGIC     print(f"\n--- Chunk ---")
# MAGIC     print(row[1])
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### F2. Hybrid Search (ANN + Keyword)
# MAGIC
# MAGIC Hybrid search combines vector similarity (ANN) with BM25 full-text keyword matching, merging results via Reciprocal Rank Fusion (RRF). This mode is useful when queries contain exact terms that must match — such as specific names, error codes, or technical terminology — where pure semantic search might miss keyword-specific results.
# MAGIC
# MAGIC 1. Complete the TODO cell below by setting the `query_type` argument to perform a hybrid search.
# MAGIC
# MAGIC 2. Use the same query text and result count as F1 so you can compare the returned chunks directly.

# COMMAND ----------

Run a hybrid (ANN + BM25) search using the same query
results_hybrid = index.similarity_search(
    query_text="What are the house rules about pets?",
    columns=["chunk_id", "chunk_text"],
    num_results=3,
    query_type="<FILL_IN>",
)
print(f"\n--- F2: Hybrid Search (ANN + Keyword, RRF) ---")
for row in results_hybrid.get("result", {}).get("data_array", []):
    print(f"\n--- Chunk ---")
    print(row[1])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC results_hybrid = index.similarity_search(
# MAGIC     query_text="What are the house rules about pets?",
# MAGIC     columns=["chunk_id", "chunk_text"],
# MAGIC     num_results=3,
# MAGIC     query_type="hybrid",
# MAGIC )
# MAGIC print(f"\n--- F2: Hybrid Search (ANN + Keyword, RRF) ---")
# MAGIC for row in results_hybrid.get("result", {}).get("data_array", []):
# MAGIC     print(f"\n--- Chunk ---")
# MAGIC     print(row[1])
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### F3. Full-Text Keyword Search (Beta)
# MAGIC
# MAGIC Full-text keyword search uses only BM25 scoring — no vector embeddings are involved. This is useful when you need exact keyword matches without semantic interpretation.
# MAGIC
# MAGIC 1. Complete the TODO cell below by setting the `query_type` argument to perform a keyword-only search.
# MAGIC
# MAGIC 2. Compare the returned chunks against the F1 and F2 results. Which mode surfaced the most relevant chunk for this query?
# MAGIC
# MAGIC **NOTE:** This search mode is currently in Beta.

# COMMAND ----------

Run a BM25 keyword-only search using the same query
results_keyword = index.similarity_search(
    query_text="What are the house rules about pets?",
    columns=["chunk_id", "chunk_text"],
    num_results=3,
    query_type="<FILL_IN>",
)
print(f"\n--- F3: Full-Text Keyword Search ---")
for row in results_keyword.get("result", {}).get("data_array", []):
    print(f"\n--- Chunk ---")
    print(row[1])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC results_keyword = index.similarity_search(
# MAGIC     query_text="What are the house rules about pets?",
# MAGIC     columns=["chunk_id", "chunk_text"],
# MAGIC     num_results=3,
# MAGIC     query_type="FULL_TEXT",
# MAGIC )
# MAGIC print(f"\n--- F3: Full-Text Keyword Search ---")
# MAGIC for row in results_keyword.get("result", {}).get("data_array", []):
# MAGIC     print(f"\n--- Chunk ---")
# MAGIC     print(row[1])
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## G. Conclusion
# MAGIC
# MAGIC In this lab, you:
# MAGIC
# MAGIC 1. **Parsed** a PDF with `ai_parse_document` and flattened its elements into rows
# MAGIC 2. **Chunked** the text with `ai_prep_search` for semantically meaningful splits, distinguishing `chunk_to_retrieve` from `chunk_to_embed`
# MAGIC 3. **Stored** chunks in a Delta table with Change Data Feed enabled
# MAGIC 4. **Created** an AI Search Delta Sync index with managed embeddings on the shared `vs_all_demo` endpoint
# MAGIC 5. **Tested** the index using similarity, hybrid, and full-text keyword search
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC With a working retrieval index in place, you are ready to wire it into a retrieval-augmented generation (RAG) application. Consider how the choice of search mode and the distinction between embedded and retrieved text will affect the quality of the context you pass to the LLM.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>