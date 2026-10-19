# Databricks notebook source
!pip install "pyyaml==6.0.3"
!pip install "backoff==2.2.1"
!pip install "databricks-openai==0.15.0"
!pip install "databricks-agents==1.10.2"
!pip install "mlflow==3.13.0"
!pip install "databricks-sdk==0.110.0"
!pip install "openai==2.30.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys, os

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

includes_dir = "./Includes"
course_dir = os.path.dirname(includes_dir)
sys.path.insert(0, course_dir)

config_path = f"{includes_dir}/config/config_5.yaml"

# COMMAND ----------

from Includes import setup_demo_environment

env = setup_demo_environment(
    config_path=config_path,
)
catalog_name = env["catalog_name"]
schema_name  = env["schema_name"]
username     = env["username"]
mlflow_experiment_path = env["mlflow_experiment_path"]