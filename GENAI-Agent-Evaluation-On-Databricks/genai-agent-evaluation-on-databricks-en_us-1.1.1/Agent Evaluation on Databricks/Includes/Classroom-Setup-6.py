# Databricks notebook source
# MAGIC %uv pip install --upgrade "mlflow[databricks]==3.8.1"
# MAGIC %uv pip install "backoff==2.2.1"
# MAGIC %uv pip install "databricks-openai==0.15.0"
# MAGIC %uv pip install "databricks-agents==1.10.2"
# MAGIC %uv pip install "databricks-vectorsearch==0.68"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
_course_root = os.path.abspath('..')
sys.path.insert(0, _course_root)

from Includes import setup_demo_environment

config = setup_demo_environment(config_path=os.path.join(_course_root, "Includes", "config", "config_6.yaml"))

catalog_name = config["catalog_name"]
schema_name = config["schema_name"]
username = config["username"]

# COMMAND ----------

import mlflow
from mlflow.entities import SpanType
from mlflow.tracking import MlflowClient

agent_name = "airbnb"
session_id = "feedback-session-001"
experiment_name = f"/Workspace/Users/{username}/airbnb_experiment"

model_username_string = username.split('@')[0].replace(".", "_") + "_agent"

# COMMAND ----------

# Load the agent from UC
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri("databricks-uc")

uc_model_name = f"{catalog_name}.{schema_name}.{agent_name}"
agent = mlflow.pyfunc.load_model(f"models:/{uc_model_name}@champion")

# COMMAND ----------

# Generate traces with session metadata for the feedback lab
queries = [
    "How many Entire home/apt listings are in the Mission neighborhood?",
    "Count the number of Private room listings in Nob Hill.",
    "What is the average listing price in Haight Ashbury?"
]

for query in queries:
    print(f"Running query: {query}")
    with mlflow.start_span(name="populate_agent_trace", span_type=SpanType.AGENT) as span:
        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.session": session_id,
                "mlflow.trace.user": username
            },
            tags={
                "training_type": "human_feedback_training",
                "agent_type": "TOOL-CALLING"
            }
        )
        query_payload = [{"input": [{"role": "user", "content": query}]}]
        response = agent.predict(query_payload)
        span.set_inputs({"query": query})
        span.set_outputs({"response": response})

print(f"\nGenerated 3 traces in session '{session_id}'")