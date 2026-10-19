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
# MAGIC # Demo - Guideline Judges with MLflow
# MAGIC ## Overview
# MAGIC In this demo, we'll implement both global and per-row guideline judges. We'll define natural-language evaluation rules, run them against evaluation datasets, and inspect the results, including evaluating pre-generated outputs without calling the agent.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC 1. Implement the `Guidelines` judge for uniform evaluation criteria across all rows
# MAGIC 2. Evaluate pre-generated inputs/outputs without a predict function
# MAGIC 3. Apply `ExpectationsGuidelines` for per-row, scenario-specific evaluation
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

# MAGIC %run ../Includes/Classroom-Setup-3

# COMMAND ----------

print(f"Username:          {username}")
print(f"Catalog Name:      {catalog_name}")
print(f"Schema Name:       {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Load Evaluation Datasets
# MAGIC
# MAGIC We'll use three datasets stored in the UC volume, each demonstrating a different guideline evaluation pattern:
# MAGIC - **`guidelines_eval.json`** - Inputs only (agent generates outputs at evaluation time)
# MAGIC - **`guidelines_eval_pre_gen.json`** - Pre-generated inputs and outputs (no agent call needed)
# MAGIC - **`guidelines_eval_row_level.json`** - Includes per-row guidelines in the `expectations` field

# COMMAND ----------

import json
from pprint import pprint
from pathlib import Path

def read_eval_from_vol(file_name: str):
    path = Path(f"/Volumes/{catalog_name}/{schema_name}/agent_vol/{file_name}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

guidelines_dataset = read_eval_from_vol("guidelines_eval.json")
guidelines_dataset_pre_gen = read_eval_from_vol("guidelines_eval_pre_gen.json")
guidelines_dataset_row_level = read_eval_from_vol("guidelines_eval_row_level.json")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### B1. Inspect Dataset Structure
# MAGIC
# MAGIC Let's examine the first dataset to understand the expected schema. Each entry contains an `inputs` field with the user's request: this is what gets passed to the agent during evaluation.

# COMMAND ----------

pprint(guidelines_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Global Guidelines Evaluation
# MAGIC
# MAGIC The `Guidelines` judge applies the same rules to every row. Here we create a guideline requiring responses in Spanish. Both inputs will fail since the agent responds in English.

# COMMAND ----------

from mlflow.genai.scorers import Guidelines

language_guideline = Guidelines(
    name="spanish",
    guidelines=["The response should be in Spanish"],
    model_name=guidelines_endpoint
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Run Evaluation with Agent

# COMMAND ----------

guidelines_dataset_results = mlflow.genai.evaluate(
    data=guidelines_dataset,
    predict_fn=lambda input: agent.predict({"input": input}),
    scorers=[language_guideline]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Inspect Results

# COMMAND ----------

print(f"The run ID is: {guidelines_dataset_results.run_id}")
print(f"The aggregated metrics are: {guidelines_dataset_results.metrics}")
print("\nThe results from the previous batch of inputs:")
display(guidelines_dataset_results.result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Evaluate Pre-Generated Outputs
# MAGIC
# MAGIC When you already have inputs and outputs stored in a dataset, you can skip the `predict_fn` parameter entirely. The dataset `guidelines_eval_pre_gen.json` contains one correct Spanish response and one English response. Expect one pass and one fail.

# COMMAND ----------

guidelines_dataset_pre_gen_results = mlflow.genai.evaluate(
    data=guidelines_dataset_pre_gen,
    scorers=[language_guideline]
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### C4. Inspect Pre-Generated Results
# MAGIC
# MAGIC Since the dataset already contained outputs, the evaluation skipped agent invocation entirely. We expect one row to pass (Spanish response) and one to fail (English response).

# COMMAND ----------

print(f"The run ID is: {guidelines_dataset_pre_gen_results.run_id}")
print(f"The aggregated metrics are: {guidelines_dataset_pre_gen_results.metrics}")
print("\nThe results from the previous batch of inputs:")
display(guidelines_dataset_pre_gen_results.result_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Per-Row Guidelines with ExpectationsGuidelines
# MAGIC
# MAGIC `ExpectationsGuidelines` applies different guidelines to each row using the `expectations` field in the dataset. This is useful when different interactions require different evaluation criteria.
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;"><code>ExpectationsGuidelines</code> requires an <code>outputs</code> field.</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">They can be passed directly in the dataset or as a trace containing them. We will pass them directly.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### D1. Define and Run Per-Row Evaluation
# MAGIC
# MAGIC Unlike global `Guidelines`, `ExpectationsGuidelines` reads the `expectations` field from each row in the dataset. Each row can specify its own list of guidelines, enabling scenario-specific evaluation without separate evaluation runs.

# COMMAND ----------

from mlflow.genai.scorers import ExpectationsGuidelines

expected_guidelines = ExpectationsGuidelines(
    name="expected_guidelines",
    model_name=guidelines_endpoint
)

guidelines_dataset_row_level_results = mlflow.genai.evaluate(
    data=guidelines_dataset_row_level,
    scorers=[expected_guidelines]
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### D2. Inspect Per-Row Results
# MAGIC
# MAGIC Each row is scored against its own `expectations` guidelines. Rows meeting their specific criteria pass; those that don't will show which guideline was violated.

# COMMAND ----------

print(f"The run ID is: {guidelines_dataset_row_level_results.run_id}")
print(f"The aggregated metrics are: {guidelines_dataset_row_level_results.metrics}")
print("\nThe results from the previous batch of inputs:")
display(guidelines_dataset_row_level_results.result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Conclusion
# MAGIC In this demo, you:
# MAGIC
# MAGIC 1. Applied global `Guidelines` to evaluate all rows against uniform criteria (Spanish language requirement)
# MAGIC 2. Evaluated pre-generated outputs without calling the agent by omitting `predict_fn`
# MAGIC 3. Used `ExpectationsGuidelines` for per-row, scenario-specific evaluation criteria

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC
# MAGIC