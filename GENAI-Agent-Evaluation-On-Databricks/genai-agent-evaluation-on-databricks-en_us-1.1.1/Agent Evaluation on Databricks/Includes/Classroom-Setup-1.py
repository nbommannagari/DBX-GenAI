# Databricks notebook source
# MAGIC %uv pip install --upgrade "backoff==2.2.1"
# MAGIC %uv pip install "databricks-openai==0.15.0"
# MAGIC %uv pip install "uv==0.11.16"
# MAGIC %uv pip install "databricks-agents==1.10.2"
# MAGIC %uv pip install "mlflow-skinny[databricks]==3.12.0"
# MAGIC %uv pip install "databricks-langchain==0.19.0"
# MAGIC %uv pip install "langchain-community==0.4.1"
# MAGIC %uv pip install "langchain==1.2.10"
# MAGIC %uv pip install "unitycatalog-langchain[databricks]==0.4.0"
# MAGIC %uv pip install "unitycatalog-ai[databricks]==0.4.0"
# MAGIC %uv pip install "databricks-sdk==0.94.0"
# MAGIC %uv pip install "databricks-vectorsearch==0.68"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
_course_root = os.path.abspath('..')
sys.path.insert(0, _course_root)

from Includes import setup_demo_environment

config = setup_demo_environment(config_path=os.path.join(_course_root, "Includes", "config", "config_1.yaml"))

# COMMAND ----------

catalog_name = config["catalog_name"]
schema_name = config["schema_name"]
volume_path = config["volume_path"]
username = config["username"]