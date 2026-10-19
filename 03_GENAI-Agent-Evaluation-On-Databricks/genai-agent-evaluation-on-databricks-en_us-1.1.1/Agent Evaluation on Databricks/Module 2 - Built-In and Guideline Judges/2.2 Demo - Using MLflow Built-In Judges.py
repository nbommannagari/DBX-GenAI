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
# MAGIC # Demo - Using MLflow Built-In Judges
# MAGIC ## Overview
# MAGIC In this demo, we'll use MLflow's built-in `Correctness` and `Safety` judges to evaluate the Airbnb agent. You'll create judge instances, load evaluation datasets, run `mlflow.genai.evaluate()`, and inspect the results.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC 1. Configure built-in judges for correctness and safety assessment
# MAGIC 2. Execute evaluations using `mlflow.genai.evaluate()` with multiple scorers
# MAGIC 3. Interpret evaluation results using MLflow's UI and result dataframes
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Prerequisites</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">This demo uses the agent created in <strong>Demo - Agent Setup</strong>. Please ensure you have completed that notebook before proceeding.</p>
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
# MAGIC <li><strong>Serverless Compute, Version 5</strong>: How to select an environment version (<a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V5</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Classroom Setup
# MAGIC Run the following cell to configure your environment.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-2

# COMMAND ----------

print(f"Username:          {username}")
print(f"Catalog Name:      {catalog_name}")
print(f"Schema Name:       {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Correctness Evaluation
# MAGIC
# MAGIC The `Correctness` judge assesses whether the agent's response is factually correct by comparing it against ground truth defined as `expected_facts` or `expected_response`.
# MAGIC
# MAGIC We'll specify a model endpoint for the judge using the `model` parameter with the format `databricks:/<endpoint-name>`.

# COMMAND ----------

from mlflow.genai.scorers import Correctness

correctness_eval = Correctness(
    model=correctness_eval_endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Load Evaluation Dataset
# MAGIC
# MAGIC The correctness evaluation dataset is stored in the UC volume. Each entry contains `inputs` (the query) and `expectations` with `expected_facts` that the response should contain.

# COMMAND ----------

import json
from pprint import pprint
from pathlib import Path

path = Path(f"/Volumes/{catalog_name}/{schema_name}/agent_vol/correctness_eval.json")

with path.open("r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

pprint(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Run Correctness Evaluation
# MAGIC
# MAGIC `mlflow.genai.evaluate()` takes three key parameters:
# MAGIC - **data**: The evaluation dataset
# MAGIC - **predict_fn**: A function that calls the agent for each input
# MAGIC - **scorers**: The list of judges to apply
# MAGIC
# MAGIC After running the cell, click **View evaluation results in MLflow** to inspect the traces and feedback.

# COMMAND ----------

correctness_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: agent.predict({"input": input}),
    scorers=[correctness_eval],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### B3. Inspect Results
# MAGIC
# MAGIC The `EvaluationResult` object contains `metrics` (aggregated scores) and `result_df` (per-row details with values and rationales). Click on any row to view the full MLflow trace.

# COMMAND ----------

print(f"The run ID is: {correctness_results.run_id}")
print(f"The aggregated metrics are: {correctness_results.metrics}")
print("\nThe results from the previous batch of inputs:")
display(getattr(correctness_results, "result_df", None))

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Safety Evaluation
# MAGIC
# MAGIC The `Safety` judge checks whether content is free from harmful, offensive, or toxic material. Unlike Correctness, it does not require ground truth: it evaluates the agent's response directly.

# COMMAND ----------

from mlflow.genai.scorers import Safety

safety_eval = Safety(
    model=safety_eval_endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### C1. Load Safety Evaluation Dataset
# MAGIC
# MAGIC Unlike the correctness dataset, the safety dataset does not require `expected_facts` or ground truth. The judge evaluates the response in isolation, checking for harmful, offensive, or toxic content regardless of factual accuracy.

# COMMAND ----------

path = Path(f"/Volumes/{catalog_name}/{schema_name}/agent_vol/safety_eval.json")

with path.open("r", encoding="utf-8") as f:
    safety_eval_dataset = json.load(f)

pprint(safety_eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### C2. Run Safety Evaluation
# MAGIC
# MAGIC The same `mlflow.genai.evaluate()` pattern applies here. Since `Safety` doesn't need ground truth, the dataset only needs `inputs`: making it easy to generate adversarial test cases that probe the agent's guardrails.

# COMMAND ----------

safety_results = mlflow.genai.evaluate(
    data=safety_eval_dataset,
    predict_fn=lambda input: agent.predict({"input": input}),
    scorers=[safety_eval],
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### C3. Inspect Results
# MAGIC
# MAGIC Safety scores are binary: `yes` (safe) or `no` (unsafe). The rationale explains which safety category was flagged (if any). Look for cases where the agent might have been tricked into producing harmful content despite its system prompt.

# COMMAND ----------

print(f"The run ID is: {safety_results.run_id}")
print(f"The aggregated metrics are: {safety_results.metrics}")
print("\nThe results from the safety evaluation:")
display(getattr(safety_results, "result_df", None))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## D. Evaluating Multiple Scorers at Once
# MAGIC
# MAGIC You can pass multiple scorers in a single `mlflow.genai.evaluate()` call. All metrics are logged to the same evaluation run, which means:
# MAGIC - A single set of traces is generated (reducing LLM calls to the agent)
# MAGIC - All judge scores appear side-by-side in the results dataframe
# MAGIC - The MLflow UI displays a unified view for comparing across dimensions
# MAGIC
# MAGIC This is the recommended pattern for production evaluation pipelines where you want to assess multiple quality dimensions simultaneously.

# COMMAND ----------

scorers = [safety_eval, correctness_eval]
mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: agent.predict({"input": input}),
    scorers=scorers
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Conclusion
# MAGIC In this demo, you:
# MAGIC
# MAGIC 1. Configured built-in `Correctness` and `Safety` judges with custom model endpoints
# MAGIC 2. Loaded evaluation datasets from a UC volume and ran evaluations with `mlflow.genai.evaluate()`
# MAGIC 3. Inspected per-row results including scores, rationales, and traces
# MAGIC 4. Combined multiple scorers in a single evaluation run

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC