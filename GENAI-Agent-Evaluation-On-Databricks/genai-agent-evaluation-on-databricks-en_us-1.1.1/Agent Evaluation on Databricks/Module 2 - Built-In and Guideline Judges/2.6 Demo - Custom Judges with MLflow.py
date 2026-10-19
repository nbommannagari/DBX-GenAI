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
# MAGIC # Demo - Custom Judges with MLflow
# MAGIC ## Overview
# MAGIC In this demo, we'll create custom LLM judges using `make_judge()` for evaluating response completeness and tool usage. We'll build both a standard custom judge that evaluates inputs/outputs and a trace-based judge that analyzes the agent's execution flow.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC 1. Create custom judges with `make_judge()` using natural language instructions and template variables
# MAGIC 2. Configure feedback value types for categorical and boolean scoring
# MAGIC 3. Build trace-based judges that analyze agent execution workflows
# MAGIC 4. Run batch evaluations over traced sessions
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

# MAGIC %run ../Includes/Classroom-Setup-5

# COMMAND ----------

print(f"Username:          {username}")
print(f"Catalog Name:      {catalog_name}")
print(f"Schema Name:       {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Load Evaluation Dataset
# MAGIC
# MAGIC The `custom_eval.json` dataset contains queries without expectations. We'll use it to evaluate the agent's response completeness.

# COMMAND ----------

import json
from pathlib import Path
from pprint import pprint

path = Path(f"/Volumes/{catalog_name}/{schema_name}/agent_vol/custom_eval.json")

with path.open("r", encoding="utf-8") as f:
    custom_eval = json.load(f)

pprint(custom_eval)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Standard Custom Judge - Response Completeness
# MAGIC
# MAGIC We'll use `make_judge()` to create a judge that evaluates whether responses are complete. The `instructions` parameter uses the `coherent_instructions` loaded from the agent eval config, which references `{{inputs}}` and `{{outputs}}` template variables.
# MAGIC
# MAGIC The `feedback_value_type` is set to a `Literal` type so the judge returns one of `"complete"`, `"partial"`, or `"incomplete"`.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <strong style="display:block; color:#0d47a1; margin-bottom:6px; font-size: 1.1em;">Why metrics will be empty</strong>
# MAGIC   <div style="color:#333;">Because the feedback type is a <code>Literal</code> (categorical), MLflow cannot compute a numeric mean — so <code>metrics</code> will be <code>{}</code>. Boolean feedback types (like in section D) <em>do</em> produce a mean metric.</div>
# MAGIC </div>

# COMMAND ----------

from typing import Literal
from mlflow.genai import make_judge

response_completeness_feedback_value_type = Literal["complete", "partial", "incomplete"]

completeness_judge = make_judge(
    name="response_completeness",
    instructions=coherent_instructions,
    feedback_value_type=response_completeness_feedback_value_type
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Run Completeness Evaluation

# COMMAND ----------

completeness_results = mlflow.genai.evaluate(
    data=custom_eval,
    predict_fn=lambda input: agent.predict({"input": input}),
    scorers=[completeness_judge]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Inspect Results

# COMMAND ----------

print(f"The run ID is: {completeness_results.run_id}")
print(f"The aggregated metrics are: {completeness_results.metrics}")
print("\nThe results from the completeness evaluation:")
display(completeness_results.result_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Trace-Based Custom Judge - Tool Usage
# MAGIC
# MAGIC Trace-based judges analyze the full execution trace using `{{trace}}` in the instructions. This judge validates whether the agent used tools correctly.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <strong style="display:block; color:#0d47a1; margin-bottom:6px; font-size: 1.1em;">How trace-based judges work</strong>
# MAGIC   <div style="color:#333;">When instructions contain <code>{{trace}}</code>, the judge receives the full trace object and uses tool calling to inspect span data (inputs, outputs, tool calls, timing). The <code>model</code> parameter is optional.</div>
# MAGIC </div>

# COMMAND ----------

tool_usage_judge = make_judge(
    name="tool_usage_validator",
    instructions=tool_usage_instructions,
    feedback_value_type=bool,
    model=custom_eval_endpoint
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Generate Traces
# MAGIC
# MAGIC To evaluate tool usage, we first need traces. The helper function below queries the agent and wraps each call in an MLflow span with session metadata, then we search for those traces.

# COMMAND ----------

from typing import Tuple, Dict, Any
from mlflow.entities import SpanType

def gen_trace_data(query: str, model: str = custom_eval_endpoint, user_id: str = username, session_id: str = session_id) -> Tuple[Dict[str, Any]]:
    with mlflow.start_span(name="populate_agent_trace", span_type=SpanType.AGENT) as span:
        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.session": session_id,
                "mlflow.trace.user": user_id
            },
            tags={
                "training_type": "agent_eval_training",
                "model": model,
                "agent_type": "TOOL-CALLING"
            }
        )

        query_payload = [
            {"input": [{"role": "user", "content": query}]}]

        response = agent.predict(query_payload)

        span.set_inputs({"query": query})
        span.set_outputs({"response": response})
        trace_id = span.trace_id

    return query, trace_id

# COMMAND ----------

queries = [
    "How many Entire home/apt listings are in the Mission neighborhood?",
    "Count the number of Private room listings in Nob Hill.",
    "What is the average listing price in Haight Ashbury?"
]

for query in queries:
    gen_trace_data(query=query)

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Evaluate a Single Trace
# MAGIC
# MAGIC Search for the traces we just generated using the session ID, then evaluate the first one individually.

# COMMAND ----------

session_traces = mlflow.search_traces(
    locations=[experiment_id],
    filter_string=f"metadata.`mlflow.trace.session` = '{session_id}'",
    return_type="list")
session_traces[0]

# COMMAND ----------

trace = session_traces[0]
feedback = tool_usage_judge(trace=trace)

# COMMAND ----------

# MAGIC %md
# MAGIC The judge returns a `Feedback` object with a boolean `value` (pass/fail) and a `rationale` explaining its reasoning.

# COMMAND ----------

print(f"Assessment: {feedback.value}")
print(f"Rationale: {feedback.rationale}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Batch Trace Evaluation
# MAGIC
# MAGIC Evaluate all session traces at once using `mlflow.genai.evaluate()`.

# COMMAND ----------

trace_judge_results = mlflow.genai.evaluate(
    data=session_traces,
    scorers=[tool_usage_judge]
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### D4. Inspect Batch Results
# MAGIC
# MAGIC Review the aggregated metrics and per-trace results from the batch evaluation. Because `feedback_value_type=bool`, MLflow can compute a mean — `1.0` means all traces passed.

# COMMAND ----------

print(f"The run ID is: {trace_judge_results.run_id}")
print(f"The aggregated metrics are: {trace_judge_results.metrics}")
print("\nThe results from trace-based evaluation:")
display(trace_judge_results.result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Conclusion
# MAGIC In this demo, you:
# MAGIC
# MAGIC 1. Created a standard custom judge with `make_judge()` using `Literal` feedback types
# MAGIC 2. Built a trace-based judge that analyzes agent tool usage via `{{trace}}`
# MAGIC 3. Generated traces with session metadata and evaluated them individually and in batch

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC
# MAGIC