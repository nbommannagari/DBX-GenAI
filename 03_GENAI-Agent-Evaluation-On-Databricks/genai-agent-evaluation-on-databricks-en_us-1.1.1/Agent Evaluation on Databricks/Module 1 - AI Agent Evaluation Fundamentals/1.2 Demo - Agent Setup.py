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
# MAGIC # Demo - Agent Setup
# MAGIC ## Overview
# MAGIC In this demo, we'll set up a tool-calling agent registered to Unity Catalog and verify it works by generating MLflow traces. This agent will be used throughout the remaining demos and labs for evaluation exercises.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC 1. Load a Unity Catalog-registered agent by name and alias
# MAGIC 2. Generate traces by interacting with the agent
# MAGIC 3. Locate those traces in Unity Catalog
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Select Compute</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Before starting this notebook, select the required compute environment listed below.</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Serverless Compute, Version 5</strong>: How to select an environment version (<a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V5</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Classroom Setup
# MAGIC Run the following cell to configure your environment. This provisions the catalog, schema, data, tools, agents, and evaluation datasets.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-1

# COMMAND ----------

print(f"Username:          {username}")
print(f"Catalog Name:      {catalog_name}")
print(f"Schema Name:       {schema_name}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Load the Agent
# MAGIC
# MAGIC The setup script registered a tool-calling agent to Unity Catalog. We'll load it using `mlflow.pyfunc.load_model()` with the model's UC path and `@champion` alias.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">About This Agent</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">This is a simple <strong>tool-calling agent</strong> backed by SQL UDFs registered to Unity Catalog. The agent itself is not the focus - it exists so we can practice evaluation techniques.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.airbnb"
agent = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@champion")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## C. Query the Agent and Inspect Traces
# MAGIC
# MAGIC Let's send some questions to the agent using its `predict()` method. Each call automatically generates an **MLflow trace**: a structured record of the agent's full execution flow, including:
# MAGIC
# MAGIC - **Spans** for each step (LLM call, tool invocation, retrieval)
# MAGIC - **Input/output payloads** at every span
# MAGIC - **Latency** for each operation
# MAGIC
# MAGIC Traces are the foundation for evaluation. They let us verify *what* the agent did, not just what it returned.
# MAGIC
# MAGIC The helper below formats our question into the **ChatCompletionRequest** schema the agent expects: a list of message objects with `role` and `content` fields, matching the OpenAI-compatible message format used by MLflow's pyfunc serving interface.

# COMMAND ----------

def agent_payload(question: str):
    return [{"input": [{"role": "user", "content": question}]}]

# COMMAND ----------

# MAGIC %md
# MAGIC This first question is a **metadata query**: the agent can answer it directly from its system prompt without invoking any tools. Notice that the resulting trace will show only an LLM span with no tool-call spans.

# COMMAND ----------

agent.predict(agent_payload("What tools do you have available?"))

# COMMAND ----------

# MAGIC %md
# MAGIC The next question will trigger both `avg_neigh_price` and `cnt_by_room_type` tool calls. Inspect the trace output to see the tool invocations.

# COMMAND ----------

agent.predict(agent_payload("Can you tell me what the average price is in Mission? Also, what's the number of listings for that neighborhood that have private rooms?"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## D. View Traces in Unity Catalog
# MAGIC
# MAGIC Navigate to **Catalog Explorer** and search for **airbnb**. You'll find the registered model with the alias **@champion**. Click the latest version (you will create multiple versions of the agent if you rerun the setup script but the **@champion** alias will always point to the latest registered version) and open the **Traces** tab to see the two requests we just sent.
# MAGIC
# MAGIC **What to look for:**
# MAGIC - **Span hierarchy** - expand a trace to see nested spans (LLM → tool calls → tool responses)
# MAGIC - **Inputs & outputs** - click any span to inspect the exact payload sent and received
# MAGIC - **Latency breakdown** - identify which step (LLM reasoning vs. tool execution) took the most time
# MAGIC - **Contrast the two traces** - the first has no tool spans; the second shows two parallel tool invocations

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Conclusion
# MAGIC In this demo, you:
# MAGIC
# MAGIC 1. Loaded a Unity Catalog-registered agent using `mlflow.pyfunc.load_model()`
# MAGIC 2. Queried the agent and observed MLflow traces capturing tool calls
# MAGIC 3. Located traces in Unity Catalog for monitoring and debugging

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC