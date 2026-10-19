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

# MAGIC %md
# MAGIC # Demo - Transforming PDFs to Structured Data
# MAGIC #### Duration: 35 minutes
# MAGIC ## Overview
# MAGIC In this demo, we'll use Databricks AI Functions to transform raw PDFs into structured tabular data. We'll parse documents, classify their content, and extract specific fields, turning unstructured files into queryable tables.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC 1. Parse a PDF using `ai_parse_document`
# MAGIC 2. Classify parsed content using `ai_classify`
# MAGIC 3. Extract structured fields using `ai_extract`
# MAGIC 4. Store the results in a Delta table

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

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-05

# COMMAND ----------

# MAGIC %md
# MAGIC As a part of the workspace setup, certain assets have been managed for you (volumes, PDF, etc.). Here, we read in from the shared catalog `dbacademy`, schema `rag_all`, and volume `vol_all` the PDF `airbnb_listings.pdf`.

# COMMAND ----------

# Variables from the classroom setup
# These reference the shared volume with Airbnb PDFs
SHARED_VOLUME = "/Volumes/dbacademy/rag_all/vol_all"
LISTINGS_PDF = f"{SHARED_VOLUME}/listings/airbnb_listings.pdf"

print(f"Listings PDF:    {LISTINGS_PDF}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Parsing Documents with `ai_parse_document`
# MAGIC
# MAGIC Let's start by parsing the Airbnb listings PDF. The `ai_parse_document` function reads the binary content and returns a machine-friendly format: a structured JSON representation of the document.
# MAGIC - We'll use the pre-defined variable from above `LISTINGS_PDF`
# MAGIC - We will also use `MAP('version', '2.0)`. This pins the parse output to the v2.0 schema (elements array, bbox, descriptions, etc.) instead of relying on whatever the current default version is.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. After running the next cell, expand the **parsed** column to view the parsed PDF contents.

# COMMAND ----------

parsed_df = spark.sql(f"""
  SELECT
    path,
    ai_parse_document(content, MAP('version', '2.0')) AS parsed
  FROM READ_FILES(
    '{LISTINGS_PDF}',
    format => 'binaryFile'
  )
""")

display(parsed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Exploring the Parsed Output
# MAGIC
# MAGIC The parsed output is a single `VARIANT` column containing a nested JSON structure. To work with individual content blocks, we need to flatten it. The query below uses two key SQL patterns:
# MAGIC
# MAGIC - **`LATERAL variant_explode(parsed:document:elements)`**: The colon (`:`) operator navigates into the VARIANT to reach the `elements` array. `variant_explode` ([AWS](https://docs.databricks.com/aws/en/sql/language-manual/functions/variant_explode) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/variant_explode) | [GCP](https://docs.databricks.com/gcp/en/sql/language-manual/functions/variant_explode)) then converts each array element into its own row, giving us one row per content block. The `LATERAL` keyword lets it reference the `parsed` column from the same `FROM` clause.
# MAGIC - **`element.value:type::STRING`**: Each exploded row has a `value` column (the array element as VARIANT). The `:` operator drills into fields like `type` and `content`, and `::STRING` casts the result to a string.

# COMMAND ----------

elements_df = spark.sql(f"""
  WITH parsed_docs AS (
    SELECT
      path,
      ai_parse_document(content, MAP('version', '2.0')) AS parsed
    FROM READ_FILES(
      '{LISTINGS_PDF}',
      format => 'binaryFile'
    )
  )
  SELECT
    path,
    element.value:type::STRING AS element_type,
    element.value:content::STRING AS content
  FROM
    parsed_docs,
    LATERAL variant_explode(parsed_docs.parsed:document:elements) AS element
  WHERE element.value:content IS NOT NULL
""")

display(elements_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Classifying Content with `ai_classify`
# MAGIC
# MAGIC After parsing, we have a flat list of text blocks — but we don't yet know *what kind* of content each block contains. Is it a property description? Pricing info? House rules? **Classification** answers that question: it reads each text block and assigns it a category label from a list you define.
# MAGIC
# MAGIC This is a separate step from **extraction**, which pulls out specific field values like property name or neighborhood. Think of it this way:
# MAGIC - **Classify** → *"What type of content is this?"* (e.g., `property_description`)
# MAGIC - **Extract** → *"What specific facts does this content contain?"* (e.g., `property_name: "Sunny Victorian"`)
# MAGIC
# MAGIC Classification is useful on its own — for example, you could filter your table to only `pricing_information` rows — and it also helps downstream extraction by letting you target the right labels to the right content.
# MAGIC
# MAGIC A few things to note in the query below:
# MAGIC
# MAGIC - **`LENGTH(...) > 50`**: Filters out short elements (headers, page numbers, etc.) that don't carry enough text for meaningful classification.
# MAGIC - **`ai_classify(content, '["property_description", ...]')`**: The second argument is a JSON array of candidate labels. The AI model reads the text and picks the best match.
# MAGIC - **`cls:response[0]`**: `ai_classify` returns a VARIANT with a `response` array. The `[0]` index retrieves the top predicted label.

# COMMAND ----------

classified_df = spark.sql(f"""
WITH elements AS (
  SELECT
    element.value:content::STRING AS content
  FROM (
    SELECT ai_parse_document(content, MAP('version', '2.0')) AS parsed
    FROM READ_FILES('{LISTINGS_PDF}', format => 'binaryFile')
  ) t,
  LATERAL variant_explode(parsed:document:elements) AS element
  WHERE element.value:content IS NOT NULL
    AND LENGTH(element.value:content::STRING) > 50
),
classified AS (
  SELECT
    content,
    ai_classify(
      content,
      '["property_description","pricing_information","location_details","amenities","host_information","other"]'
    ) AS cls
  FROM elements
)
SELECT
  content,
  cls:response[0] AS category
FROM classified
LIMIT 20
""")
display(classified_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Extracting Structured Fields with `ai_extract`
# MAGIC
# MAGIC In order to turn unstructured text into structured, queryable data, we need to pull out specific pieces of information — names, locations, categories, numbers — from free-form prose. This process is called **entity extraction**: the AI reads a block of text and identifies the concrete facts buried inside it.
# MAGIC
# MAGIC For example, a paragraph like *"Sunny Victorian in Hayes Valley with WiFi and a garden"* contains several distinct facts: a property name (`Sunny Victorian`), a neighborhood (`Hayes Valley`), and amenities (`WiFi`, `garden`). Extraction locates and returns each of these as named fields, so you end up with clean columns you can filter, join, and aggregate — just like any other table.
# MAGIC
# MAGIC `ai_extract` handles this for you: you tell it which fields to look for, and it returns a structured JSON object with those fields populated. Let's extract property name, neighborhood, property type, and key amenities from the parsed listing text.
# MAGIC
# MAGIC A few things to note in the query below:
# MAGIC
# MAGIC - **`LENGTH(...) > 100`**: Uses a higher threshold than `ai_classify` because extraction needs enough surrounding text to locate meaningful field values.
# MAGIC - **`ai_extract(content, '["property_name", "neighborhood", ...]')`**: The second argument lists the field names you want. The AI model scans the text and returns a JSON object with those keys populated.
# MAGIC - **`ext:response.property_name`**: The result VARIANT has a `response` object. Use dot notation to access each extracted field by name.

# COMMAND ----------

extracted_df = spark.sql(f"""
WITH elements AS (
  SELECT
    element.value:content::STRING AS content
  FROM (
    SELECT ai_parse_document(content, MAP('version', '2.0')) AS parsed
    FROM READ_FILES('{LISTINGS_PDF}', format => 'binaryFile')
  ) t,
  LATERAL variant_explode(parsed:document:elements) AS element
  WHERE element.value:content IS NOT NULL
    AND LENGTH(element.value:content::STRING) > 100
),
extracted AS (
  SELECT
    content,
    ai_extract(
      content,
      '["property_name", "neighborhood", "property_type", "key_amenities"]'
    ) AS ext
  FROM elements
)
SELECT
  content,
  ext:response.property_name   AS property_name,
  ext:response.neighborhood    AS neighborhood,
  ext:response.property_type   AS property_type,
  ext:response.key_amenities   AS key_amenities
FROM extracted
LIMIT 15
""")

display(extracted_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also explore the raw `ai_extract` VARIANT output directly, without unpacking individual fields. This lets you see the full JSON structure returned by the function, useful for understanding the schema before deciding which fields to extract.

# COMMAND ----------

# Explore the raw ai_extract output
explore_df = spark.sql(f"""
WITH elements AS (
  SELECT
    element.value:content::STRING AS content
  FROM (
    SELECT ai_parse_document(content, MAP('version', '2.0')) AS parsed
    FROM READ_FILES('{LISTINGS_PDF}', format => 'binaryFile')
  ) t,
  LATERAL variant_explode(parsed:document:elements) AS element
  WHERE element.value:content IS NOT NULL
    AND LENGTH(element.value:content::STRING) > 100
)
SELECT
  content,
  ai_extract(
    content,
    '["property_name", "neighborhood", "property_type", "key_amenities"]'
  ) AS ext
FROM elements
LIMIT 5
""")

display(explore_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Storing Results in a Delta Table
# MAGIC
# MAGIC Let's combine parsing, classification, and extraction into a single pipeline and save the results. The query below chains everything into one `CREATE OR REPLACE TABLE` statement:
# MAGIC
# MAGIC 1. The `elements` CTE parses the PDF and flattens it (same as section B1)
# MAGIC 2. The `classified_extracted` CTE runs both `ai_classify` and `ai_extract` on each element. Since they're in the same `SELECT`, Databricks can process them in parallel per row
# MAGIC 3. The final `SELECT` unpacks both results and writes them as named columns

# COMMAND ----------

# MAGIC %md
# MAGIC Run the cell below to create a structured table from the parsed PDF content.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.airbnb_structured_listings AS
WITH elements AS (
  SELECT
    element.value:content::STRING AS content
  FROM (
    SELECT ai_parse_document(content, MAP('version', '2.0')) AS parsed
    FROM READ_FILES('{LISTINGS_PDF}', format => 'binaryFile')
  ) t,
  LATERAL variant_explode(parsed:document:elements) AS element
  WHERE element.value:content IS NOT NULL
    AND LENGTH(element.value:content::STRING) > 100
),
classified_extracted AS (
  SELECT
    content,
    ai_classify(
      content,
      '["property_description","pricing_information","location_details","amenities","host_information","other"]'
    ) AS cls,
    ai_extract(
      content,
      '["property_name","neighborhood","property_type","key_amenities"]'
    ) AS ext
  FROM elements
)
SELECT
  content,
  cls:response[0]                 AS category,
  ext:response.property_name      AS property_name,
  ext:response.neighborhood       AS neighborhood,
  ext:response.property_type      AS property_type,
  ext:response.key_amenities      AS key_amenities
FROM classified_extracted
""")

print("Table created: airbnb_structured_listings")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM airbnb_structured_listings

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Conclusion
# MAGIC In this demo, you:
# MAGIC
# MAGIC 1. **Parsed** PDFs using `ai_parse_document` to extract structured content
# MAGIC 2. **Classified** parsed text into categories using `ai_classify`
# MAGIC 3. **Extracted** structured fields using `ai_extract`
# MAGIC 4. **Stored** the results in a Delta table for downstream analysis
# MAGIC
# MAGIC This **Parse → Classify → Extract** pattern is ideal for turning unstructured documents into structured, queryable data.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC