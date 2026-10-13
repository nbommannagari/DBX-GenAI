# Databricks notebook source
# MAGIC %md
# MAGIC # Workspace Setup (Admin-only)
# MAGIC
# MAGIC This notebook provisions shared resources for the course. Run it once per workspace before students begin.
# MAGIC
# MAGIC **What it creates:**
# MAGIC 1. Shared Unity Catalog namespace (`dbacademy.airbnb_agent`) with the Airbnb listings table
# MAGIC 2. User permissions on all shared resources

# COMMAND ----------

import sys, os
includes_dir = "."
config_version = "workspace-setup"
sys.path.insert(0, os.path.abspath('.'))
config_path = f"{includes_dir}/config/config-{config_version}.yaml"

from _lib.library_installer import install_libraries
install_libraries(config_path)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

ws = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define Variables and Create Catalog / Schema
# MAGIC
# MAGIC Set up the shared UC namespace and define all variable names used throughout this notebook.

# COMMAND ----------

# --- Shared UC namespace ---
SHARED_CATALOG = "dbacademy"
SHARED_SCHEMA  = "airbnb_agent"

# --- External (Marketplace) dependencies ---
# Loaded outside this notebook from the Marketplace share; the grant in
# Section 3 assumes the table is already present.
DATABRICKS_SHARE_NAME = "dbacademy_airbnb"
TABLE_NAME            = "sf_airbnb_listings"

# Create catalog / schema
spark.sql(f"USE CATALOG {SHARED_CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SHARED_SCHEMA}")
spark.sql(f"USE SCHEMA {SHARED_SCHEMA}")

print(f"Shared catalog : {SHARED_CATALOG}")
print(f"Shared schema  : {SHARED_SCHEMA}")

# COMMAND ----------

from pyspark.sql.functions import concat, lit, col, regexp_replace, trim

df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .load(f"/Volumes/{DATABRICKS_SHARE_NAME}/v01/sf-listings/sf-airbnb.csv")
    .select("id", "name", "neighbourhood", "neighbourhood_cleansed", "summary", "price", "room_type")
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

df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_NAME)
print(f"Loaded {df.count()} rows into {SHARED_CATALOG}.{SHARED_SCHEMA}.{TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate listings PDF into shared volume

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Grant read access to shared resources

# COMMAND ----------

# Grant read-only access on shared catalog resources
spark.sql(f"GRANT USE CATALOG ON CATALOG {SHARED_CATALOG} TO `account users`")
spark.sql(f"GRANT USE SCHEMA ON SCHEMA {SHARED_CATALOG}.{SHARED_SCHEMA} TO `account users`")
spark.sql(f"GRANT SELECT ON TABLE {SHARED_CATALOG}.{SHARED_SCHEMA}.{TABLE_NAME} TO `account users`")

print(f"Granted read-only access on {SHARED_CATALOG}.{SHARED_SCHEMA} to all users")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("Workspace Setup Complete")
print("=" * 60)
print(f"  Shared catalog   : {SHARED_CATALOG}")
print(f"  Shared schema    : {SHARED_SCHEMA}")
print(f"  Table            : {SHARED_CATALOG}.{SHARED_SCHEMA}.{TABLE_NAME}")
print(f"  User permissions : READ-ONLY on shared resources")