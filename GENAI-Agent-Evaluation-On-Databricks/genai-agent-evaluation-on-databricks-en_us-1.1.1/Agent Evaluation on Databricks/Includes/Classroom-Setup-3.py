# Databricks notebook source
# MAGIC %uv pip install --upgrade "mlflow[databricks]==3.8.1"
# MAGIC %uv pip install "backoff==2.2.1"
# MAGIC %uv pip install "databricks-openai==0.15.0"
# MAGIC %uv pip install "databricks-vectorsearch==0.68"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
_course_root = os.path.abspath('..')
sys.path.insert(0, _course_root)

from Includes import setup_demo_environment

config = setup_demo_environment(config_path=os.path.join(_course_root, "Includes", "config", "config_3.yaml"))

catalog_name = config["catalog_name"]
schema_name = config["schema_name"]
username = config["username"]

# COMMAND ----------

import mlflow

mlflow.set_tracking_uri("databricks")

EXPERIMENT_NAME = f"/Users/{username}/airbnb_experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Your username is: {username}")
print(f"The location of your experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

alias = "champion"
agent_name = "airbnb"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{agent_name}"

print(UC_MODEL_NAME)

agent = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@{alias}")

# COMMAND ----------

development_config = os.path.join(_course_root, "Includes", "artifacts", "configs", "agent_eval_config.yaml")

config_obj = mlflow.models.ModelConfig(development_config=development_config)

guidelines_endpoint = config_obj.get("GUIDELINES_ENDPOINT")

print(f"Guidelines Endpoint: {guidelines_endpoint}")