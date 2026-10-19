# Databricks notebook source
!pip install "pyyaml==6.0.3"
!pip install "backoff==2.2.1"
!pip install "databricks-openai==0.15.0"
!pip install "databricks-agents==1.10.2"
!pip install "mlflow==3.13.0"
!pip install "databricks-sdk==0.110.0"
!pip install "pydantic==2.11.10"

# COMMAND ----------

import sys, os

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

# Clear stale typing_extensions modules loaded by the runtime
for module_name in list(sys.modules):
    if module_name == "typing_extensions" or module_name.startswith("typing_extensions."):
        del sys.modules[module_name]

# COMMAND ----------


includes_dir = "./Includes"
course_dir = os.path.dirname(includes_dir)
sys.path.insert(0, course_dir)


config_path = f"{includes_dir}/config/config_2.yaml" # Notebook number needs ot match config file name

# COMMAND ----------

from Includes import setup_demo_environment

env = setup_demo_environment(
    config_path=config_path,
)
catalog_name = env["catalog_name"]
schema_name = env["schema_name"]

# COMMAND ----------

# MAGIC %pip install -U "typing_extensions>=4.16.0"
# MAGIC
# MAGIC dbutils.library.restartPython()