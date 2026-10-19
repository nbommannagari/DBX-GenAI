# Databricks notebook source
# MAGIC %md
# MAGIC # Workspace Setup (Admin-only)
# MAGIC
# MAGIC This notebook provisions shared resources for the course. Run it once per workspace before students begin.
# MAGIC
# MAGIC **What it creates:**
# MAGIC 1. Shared Unity Catalog namespace (`dbacademy.rag_all`) with volumes and PDFs
# MAGIC 2. Vector Search endpoint (`vs_all_demo`) with user permissions
# MAGIC 3. User permissions on all shared resources
# MAGIC 4. A scheduled job that re-runs this notebook every 10 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Libraries

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

import os
import shutil
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import iam
from databricks.ai_search.client import AISearchClient

ws = WorkspaceClient()
vsc = AISearchClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define Variables and Create Catalog / Schema / Volume
# MAGIC
# MAGIC Set up the shared UC namespace and define all variable names used throughout this notebook.

# COMMAND ----------

# --- Shared UC namespace ---
SHARED_CATALOG = "dbacademy"
SHARED_SCHEMA  = "rag_all"
SHARED_VOLUME  = "vol_all"
suffix = ""

# --- Vector Search ---
VS_ENDPOINT_DEMO = f"vs_all_demo{suffix}"

# --- Volume paths ---
VOLUME_PATH          = f"/Volumes/{SHARED_CATALOG}/{SHARED_SCHEMA}/{SHARED_VOLUME}"
LISTINGS_SUBFOLDER   = f"{VOLUME_PATH}/listings"
ADDITIONAL_SUBFOLDER = f"{VOLUME_PATH}/additional_listing_info"
INVOICES_SUBFOLDER   = f"{VOLUME_PATH}/invoices"
LISTINGS_PDF_PATH    = f"{LISTINGS_SUBFOLDER}/airbnb_listings.pdf"
HOUSE_RULES_PDF_PATH = f"{ADDITIONAL_SUBFOLDER}/airbnb_house_rules.pdf"

# --- External (Marketplace) dependencies ---
# Loaded outside this notebook from the Marketplace share; the grant in
# Section 4 assumes the table is already present.
DATABRICKS_SHARE_NAME = "dbacademy_airbnb"
TABLE_NAME            = "sf_airbnb_listings"

# Create catalog / schema / volume
spark.sql(f"USE CATALOG {SHARED_CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SHARED_SCHEMA}")
spark.sql(f"USE SCHEMA {SHARED_SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {SHARED_SCHEMA}.{SHARED_VOLUME}")

print(f"Shared catalog : {SHARED_CATALOG}")
print(f"Shared schema  : {SHARED_SCHEMA}")
print(f"Shared volume  : {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Copy PDFs to Volume
# MAGIC
# MAGIC Copy the source PDFs (listings, house rules, invoices) into the shared UC volume so they are accessible to all students.

# COMMAND ----------

# Source paths (relative to Includes/ when run as %run)
data_dir = os.path.join(os.getcwd(), "data")


def copy_file_to_volume(source_file, dest_path):
    """Copy a local file to a UC volume.

    Uses a direct Python copy instead of `dbutils.fs.cp` because on
    serverless compute the latter routes through Spark Connect, which
    can't read `file:/Workspace/...` source paths and raises a noisy
    `WorkspaceLocalFileSystemInternalError` even though the file is
    locally readable.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copyfile(source_file, dest_path)
    print(f"  Copied: {os.path.basename(source_file)} -> {dest_path}")


# Create subfolders
os.makedirs(LISTINGS_SUBFOLDER, exist_ok=True)
os.makedirs(ADDITIONAL_SUBFOLDER, exist_ok=True)

# airbnb_listings.pdf -> vol_all/listings/
copy_file_to_volume(
    os.path.join(data_dir, "airbnb_listings.pdf"),
    f"{LISTINGS_SUBFOLDER}/airbnb_listings.pdf",
)

# airbnb_house_rules.pdf -> vol_all/additional_listing_info/
copy_file_to_volume(
    os.path.join(data_dir, "airbnb_house_rules.pdf"),
    f"{ADDITIONAL_SUBFOLDER}/airbnb_house_rules.pdf",
)

# invoices -> vol_all/invoices/
invoices_src_dir = os.path.join(data_dir, "invoices")
if os.path.exists(invoices_src_dir):
    os.makedirs(INVOICES_SUBFOLDER, exist_ok=True)
    for fname in sorted(os.listdir(invoices_src_dir)):
        if fname.endswith(".pdf"):
            copy_file_to_volume(
                os.path.join(invoices_src_dir, fname),
                f"{INVOICES_SUBFOLDER}/{fname}",
            )

print("\nAll PDFs copied to volume.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. Create Vector Search Endpoints
# MAGIC
# MAGIC One endpoint is created:
# MAGIC - **`vs_all_demo`** — used by demo and lab notebooks

# COMMAND ----------

def _ensure_vs_endpoint(endpoint_name):
    """Create a VS endpoint if it doesn't exist and wait for ONLINE."""
    try:
        vsc.get_endpoint(endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists.")
    except Exception:
        vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        print(f"Creating endpoint '{endpoint_name}'...")

    for _ in range(60):
        ep = vsc.get_endpoint(endpoint_name)
        status = ep.get("endpoint_status", {}).get("state", "")
        if status == "ONLINE":
            print(f"Endpoint '{endpoint_name}' is ONLINE.")
            return
        print(f"  {endpoint_name} status: {status} - waiting...")
        time.sleep(10)
    print(f"WARNING: Endpoint '{endpoint_name}' did not reach ONLINE status in time.")

endpoint_list = [VS_ENDPOINT_DEMO]

for ep_name in endpoint_list:
    _ensure_vs_endpoint(ep_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Grant `CAN_USE` on Endpoints
# MAGIC
# MAGIC VS endpoints are workspace objects, so permissions are managed via the Permissions API. Without `CAN_USE`, users cannot query indexes hosted on these endpoints.

# COMMAND ----------

for ep_name in endpoint_list:
    ep = vsc.get_endpoint(ep_name)
    ep_id = str(ep["id"])
    ws.permissions.set(
        request_object_type="vector-search-endpoints",
        request_object_id=ep_id,
        access_control_list=[
            iam.AccessControlRequest(
                group_name="users",
                permission_level=iam.PermissionLevel.CAN_USE,
            )
        ],
    )
    print(f"Granted CAN_USE on endpoint '{ep_name}' (id={ep_id}) to users.")

# COMMAND ----------

# Wait for system.ai schema to become available before granting permissions
max_attempts = 5
backoff_seconds = 60

for attempt in range(1, max_attempts + 1):
    try:
        spark.sql("SELECT routine_name, routine_type FROM system.information_schema.routines WHERE routine_schema = 'ai'")
        print(f"system.ai schema is available (attempt {attempt}/{max_attempts}).")
        break
    except Exception as e:
        if attempt < max_attempts:
            print(f"  Attempt {attempt}/{max_attempts}: system.ai not yet available. Waiting {backoff_seconds}s...")
            time.sleep(backoff_seconds)
        else:
            print(f"  system.ai not available after {max_attempts} attempts. Continuing anyway...")

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT USE CATALOG ON CATALOG system TO `account users`;
# MAGIC GRANT USE SCHEMA ON SCHEMA system.ai TO `account users`;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW GRANTS ON SCHEMA system.ai;
# MAGIC SHOW GRANTS ON CATALOG system;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Grant Permissions
# MAGIC
# MAGIC Grant access to all shared catalog resources.

# COMMAND ----------

# Read-only access on shared catalog resources
spark.sql(f"GRANT USE CATALOG ON CATALOG {SHARED_CATALOG} TO `account users`")
spark.sql(f"GRANT USE SCHEMA  ON SCHEMA  {SHARED_CATALOG}.{SHARED_SCHEMA} TO `account users`")
spark.sql(f"GRANT READ VOLUME ON VOLUME  {SHARED_CATALOG}.{SHARED_SCHEMA}.{SHARED_VOLUME} TO `account users`")

# Marketplace-loaded Airbnb table — grant only if it exists in this workspace
try:
    spark.sql(
        f"GRANT SELECT ON TABLE "
        f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{TABLE_NAME} TO `account users`"
    )
    print(f"Granted SELECT on {SHARED_CATALOG}.{SHARED_SCHEMA}.{TABLE_NAME}")
except Exception as e:
    print(
        f"Skipped GRANT on {TABLE_NAME}: {e}\n"
        f"  Load the table from the '{DATABRICKS_SHARE_NAME}' Marketplace "
        f"share, then re-run this cell."
    )

print(f"Granted read-only access on {SHARED_CATALOG}.{SHARED_SCHEMA} to all users")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy Scheduled Job
# MAGIC
# MAGIC Create (or update) a Databricks Job that re-runs this notebook every 10 minutes to keep shared resources synced.

# COMMAND ----------

from databricks.sdk.service.jobs import (
    JobSettings,
    Task,
    NotebookTask,
    CronSchedule,
    PauseStatus,
)

JOB_NAME = f"Workspace-Setup Refresh{suffix}"
NOTEBOOK_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# Check if the job already exists
existing_jobs = [j for j in ws.jobs.list(name=JOB_NAME)]

job_settings = JobSettings(
    name=JOB_NAME,
    tasks=[
        Task(
            task_key="run_workspace_setup",
            notebook_task=NotebookTask(
                notebook_path=NOTEBOOK_PATH,
            ),
        )
    ],
    schedule=CronSchedule(
        quartz_cron_expression="0 */10 * * * ?",
        timezone_id="UTC",
        pause_status=PauseStatus.UNPAUSED,
    ),
)

if existing_jobs:
    job_id = existing_jobs[0].job_id
    ws.jobs.reset(job_id=job_id, new_settings=job_settings)
    print(f"Updated existing job '{JOB_NAME}' (job_id={job_id})")
else:
    created_job = ws.jobs.create(
        name=job_settings.name,
        tasks=job_settings.tasks,
        schedule=job_settings.schedule,
    )
    job_id = created_job.job_id
    print(f"Created job '{JOB_NAME}' (job_id={job_id})")

print(f"  Schedule: Every 10 minutes (UTC)")
print(f"  Notebook: {NOTEBOOK_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("Workspace Setup Complete")
print("=" * 60)
print(f"  Shared catalog   : {SHARED_CATALOG}")
print(f"  Shared schema    : {SHARED_SCHEMA}")
print(f"  Shared volume    : {VOLUME_PATH}")
print(f"  Listings PDF     : {LISTINGS_PDF_PATH}")
print(f"  VS endpoint demo : {VS_ENDPOINT_DEMO}")
print(f"  User permissions : READ-ONLY on shared resources")