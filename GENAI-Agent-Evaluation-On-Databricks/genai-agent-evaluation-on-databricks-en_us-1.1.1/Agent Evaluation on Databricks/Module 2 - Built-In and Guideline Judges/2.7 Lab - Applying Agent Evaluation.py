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
# MAGIC
# MAGIC # Lab - Applying Agent Evaluation
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This lab provides hands-on experience with MLflow's evaluation framework for generative AI agents. You will work with the NYC Taxi dataset to build evaluation workflows using both built-in judges (correctness, safety) and custom guideline judges.
# MAGIC
# MAGIC In this lab, you will create evaluation datasets, configure multiple types of judges, and analyze evaluation results to understand agent performance. The lab emphasizes practical implementation of automated evaluation workflows that can be applied to real-world AI applications.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC
# MAGIC - **Configure and execute** MLflow's built-in judges for correctness and safety assessment
# MAGIC - **Implement guideline judges** for custom business rule evaluation using natural language criteria
# MAGIC - **Create evaluation datasets** with proper structure for different judge types
# MAGIC - **Analyze evaluation results** using MLflow's tracing capabilities and result dataframes
# MAGIC - **Apply best practices** for separating evaluation models and managing configuration files
# MAGIC
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Prerequisites</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"> This demo uses the agent created in <strong>Demo - Agent Setup</strong>. Please ensure you have completed that notebook before proceeding.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">That you might need to rerun the cells in the notebook if you receive an <strong>ValueError</strong> or <strong>MLflowException</strong> error regarding a datatype that was generated as a part of the agent's trace.</p>
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

# MAGIC %run ../Includes/Classroom-Setup-4

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Test Agent Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Test Agent Functions
# MAGIC
# MAGIC Let's test the agent's capabilities by calling it with different types of queries to understand how it uses the available functions.

# COMMAND ----------

def agent_payload(question: str):
    return [{"input": [{"role": "user", "content": question}]}]

# COMMAND ----------

## Test the agent with a query about trip information
## Use a query that would trigger the get_trip_info function
test_query_1 = <FILL_IN>
result_1 = agent.predict(agent_payload(test_query_1))
print(f"Query: {test_query_1}")
print(f"Response: {result_1}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task B1: Test Trip Info Query
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC test_query_1 = "What's the average trip distance for trips from zip code 10001 to 10002?"
# MAGIC result_1 = agent.predict(agent_payload(test_query_1))
# MAGIC print(f"Query: {test_query_1}")
# MAGIC print(f"Response: {result_1}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

## Test the agent with a fare calculation query
## Use a query that would trigger the calculate_fare_estimate function
test_query_2 = <FILL_IN>
result_2 = agent.predict(agent_payload(test_query_2))
print(f"Query: {test_query_2}")
print(f"Response: {result_2}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task B1: Test Fare Calculation Query
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC test_query_2 = "What would be the estimated fare for a 5.2 mile trip with a base fare of $3.50?"
# MAGIC result_2 = agent.predict(agent_payload(test_query_2))
# MAGIC print(f"Query: {test_query_2}")
# MAGIC print(f"Response: {result_2}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Built-In Judges Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Create Evaluation Dataset for Built-In Judges
# MAGIC
# MAGIC Create an evaluation dataset with two inputs, expected facts, and expected responses for testing correctness and safety judges. Part of the evaluation dataset has been filled out for you.
# MAGIC
# MAGIC **Hint:** If you get stuck, you can view a sample evaluation dataset in your volume at `/Volumes/{catalog_name}/{schema_name}/agent_vol/nyc_taxi_eval.json`.

# COMMAND ----------

## Create an evaluation dataset for built-in judges
## Include inputs and expectations with expected_facts for each query
eval_dataset_builtin = [
  {
    "inputs": {
      "input": [
        {
          "role": <FILL_IN>,
          "content": <FILL_IN>
        }
      ]
    },
    "expectations": {
      "expected_facts": [
        <FILL_IN>
      ]
    }
  },
  {
    "inputs": {
      "input": [
        {
          "role": <FILL_IN>,
          "content": <FILL_IN>
        }
      ]
    },
    "expectations": {
      "expected_facts": [
        <FILL_IN>
      ]
    }
  }
]

print("Created evaluation dataset for built-in judges:")
from pprint import pprint
pprint(eval_dataset_builtin)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task C1: Create Evaluation Dataset
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC eval_dataset_builtin = [
# MAGIC   {
# MAGIC     "inputs": {
# MAGIC       "input": [
# MAGIC         {
# MAGIC           "role": "user",
# MAGIC           "content": "What is the average fare for trips from zip code 10001 to 10002?"
# MAGIC         }
# MAGIC       ]
# MAGIC     },
# MAGIC     "expectations": {
# MAGIC       "expected_facts": [
# MAGIC         "Pickup zip code is 10001",
# MAGIC         "Dropoff zip code is 10002",
# MAGIC         "Response includes average fare information"
# MAGIC       ]
# MAGIC     }
# MAGIC   },
# MAGIC   {
# MAGIC     "inputs": {
# MAGIC       "input": [
# MAGIC         {
# MAGIC           "role": "user",
# MAGIC           "content": "How many trips were taken from zip code 10003 to 10004?"
# MAGIC         }
# MAGIC       ]
# MAGIC     },
# MAGIC     "expectations": {
# MAGIC       "expected_facts": [
# MAGIC         "Pickup zip code is 10003",
# MAGIC         "Dropoff zip code is 10004",
# MAGIC         "Response includes trip count",
# MAGIC         "Response includes 'Are there any additional questions I can help with?'"
# MAGIC       ]
# MAGIC     }
# MAGIC   }
# MAGIC ]
# MAGIC print("Created evaluation dataset for built-in judges:")
# MAGIC from pprint import pprint
# MAGIC pprint(eval_dataset_builtin)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Configure Built-In Judges
# MAGIC
# MAGIC Set up the Correctness and Safety judges using the configured endpoints.

# COMMAND ----------

## Import and configure the built-in judges
from mlflow.genai.scorers import <FILL_IN>, <FILL_IN>

## Create correctness judge instance
correctness_eval = <FILL_IN>(
    model=<FILL_IN>
)

## Create safety judge instance
safety_eval = <FILL_IN>(
    model=<FILL_IN>
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task C2: Configure Built-In Judges
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import Correctness, Safety
# MAGIC # Create correctness judge instance
# MAGIC correctness_eval = Correctness(
# MAGIC     model=correctness_eval_endpoint
# MAGIC )
# MAGIC # Create safety judge instance
# MAGIC safety_eval = Safety(
# MAGIC     model=safety_endpoint
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Execute Built-In Judge Evaluation
# MAGIC
# MAGIC Run the evaluation using both correctness and safety judges simultaneously. Click on the **View evaluation results with MLflow** to inspect the results.

# COMMAND ----------

## Execute evaluation with both built-in judges
builtin_results = mlflow.genai.evaluate(
    data=<FILL_IN>,
    predict_fn=<FILL_IN>,
    scorers=<FILL_IN>
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task C3: Execute Evaluation
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC builtin_results = mlflow.genai.evaluate(
# MAGIC     data=eval_dataset_builtin,
# MAGIC     predict_fn=lambda input: agent.predict({"input": input}),
# MAGIC     scorers=[correctness_eval, safety_eval]
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### C4. Analyze Built-In Judge Results
# MAGIC
# MAGIC Examine the evaluation results and understand the scoring rationale by accessing `builtin_results.metrics` and `builtin_results.result_df`.

# COMMAND ----------

## Display the evaluation results and metrics
print(f"Run ID: {<FILL_IN>}")
print(f"Aggregated Metrics: {<FILL_IN>}")
print("\nDetailed Results:")
display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task C4: Analyze Results
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC print(f"Run ID: {builtin_results.run_id}")
# MAGIC print(f"Aggregated Metrics: {builtin_results.metrics}")
# MAGIC print("\nDetailed Results:")
# MAGIC display(builtin_results.result_df)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### C5. Update and Iterate (Optional)
# MAGIC
# MAGIC Did your agent pass your expectations? If not, try updating the system prompt in `Includes/artifacts/configs/nyc_taxi_agent_config.yaml` so that the agent aligns with your evaluation expectations. Iterate until all evaluations **Pass**.
# MAGIC
# MAGIC For example, the default system prompt is
# MAGIC
# MAGIC ```
# MAGIC You are an expert in answering questions regarding NYC Taxi data.
# MAGIC ```
# MAGIC
# MAGIC Which isn't very descriptive if, for example, we pass the question _What is the average fare for trips from zip code 10001 to 10002?_ and expect the following in our output
# MAGIC - Pickup zip code is 10001,
# MAGIC - Dropoff zip code is 10002,
# MAGIC - Response includes average fare information,
# MAGIC - Response includes 'Are there any additional questions I can help with?'
# MAGIC
# MAGIC then we might update our system prompt with _You are an expert in answering questions regarding NYC Taxi data. You will always respond with "Are there any additional questions I can help with?"_ (see screenshot below).
# MAGIC
# MAGIC ![update_system_prompt.png](https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/applying agent eval/update_system_prompt.png "update_system_prompt.png")

# COMMAND ----------

# MAGIC %md
# MAGIC After updating the prompt, you can redeploy your agent with the following code snippet.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">CODE</span> Redeploy Agent with Updated Prompt
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC import yaml
# MAGIC from mlflow.models.resources import DatabricksServingEndpoint, DatabricksFunction
# MAGIC
# MAGIC # Point to the rendered config file created by the setup orchestrator
# MAGIC CONFIG_YAML = "../Includes/artifacts/configs/nyc_taxi_agent_config.yaml"
# MAGIC
# MAGIC # Load config from YAML
# MAGIC with open(CONFIG_YAML, "r") as f:
# MAGIC     cfg = yaml.safe_load(f)
# MAGIC
# MAGIC # Pull values from YAML
# MAGIC CATALOG = cfg["CATALOG_NAME"]
# MAGIC SCHEMA = cfg["SCHEMA_NAME"]
# MAGIC MODEL_BASE_NAME = cfg.get("MODEL_NAME", "nyc_taxi")
# MAGIC UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_BASE_NAME}"
# MAGIC LLM_ENDPOINT = cfg["LLM_ENDPOINT_NAME"]
# MAGIC
# MAGIC # Build tool list from config
# MAGIC tool1 = cfg.get("TOOL1")
# MAGIC tool2 = cfg.get("TOOL2")
# MAGIC tool_names = [f"{CATALOG}.{SCHEMA}.{t}" for t in [tool1, tool2] if t]
# MAGIC
# MAGIC # Build MLflow resources
# MAGIC resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)] + [
# MAGIC     DatabricksFunction(function_name=t) for t in tool_names
# MAGIC ]
# MAGIC
# MAGIC # Agent source file (created by the setup orchestrator)
# MAGIC AGENT_PY = "../Includes/artifacts/nyc_taxi_agent.py"
# MAGIC ALIAS = "challenger"
# MAGIC
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC
# MAGIC with mlflow.start_run(tags={
# MAGIC     "training_type": "agent_eval_training",
# MAGIC     "model": LLM_ENDPOINT,
# MAGIC     "agent_type": "TOOL-CALLING",
# MAGIC }):
# MAGIC     logged = mlflow.pyfunc.log_model(
# MAGIC         artifact_path="agent",
# MAGIC         python_model=AGENT_PY,
# MAGIC         model_config=CONFIG_YAML,
# MAGIC         artifacts={"agent_config": CONFIG_YAML},
# MAGIC         input_example={"input": [{"role": "user", "content": "hello"}]},
# MAGIC         pip_requirements=["databricks-openai", "backoff", "pyyaml"],
# MAGIC         resources=resources,
# MAGIC         registered_model_name=UC_MODEL_NAME,
# MAGIC     )
# MAGIC
# MAGIC new_version = logged.registered_model_version
# MAGIC print("New UC version:", new_version)
# MAGIC
# MAGIC from mlflow.tracking import MlflowClient
# MAGIC MlflowClient().set_registered_model_alias(
# MAGIC     name=UC_MODEL_NAME, alias=ALIAS, version=new_version
# MAGIC )
# MAGIC
# MAGIC agent = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@{ALIAS}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC You can now go back to sections **C3** and **C4** and re-evaluate.

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Guideline Judges Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Create Global Guidelines Dataset
# MAGIC
# MAGIC Create an evaluation dataset for testing global guidelines that apply uniformly to all responses. Since the `Guidelines` judge defines rules at the judge level (not per-row), the dataset only needs `inputs`.

# COMMAND ----------

## Create evaluation dataset for global guidelines
eval_dataset_guidelines = [
    {
        "inputs": <FILL_IN>
    }
]

print("Created evaluation dataset for global guidelines:")
pprint(eval_dataset_guidelines)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task D1: Create Guidelines Dataset
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC eval_dataset_guidelines = [
# MAGIC   {
# MAGIC     "inputs": {
# MAGIC       "input": [
# MAGIC         {
# MAGIC           "role": "user",
# MAGIC           "content": "What is the average fare for trips from zip code 10001 to 10002?"
# MAGIC         }
# MAGIC       ]
# MAGIC     }
# MAGIC   }
# MAGIC ]
# MAGIC print("Created evaluation dataset for global guidelines:")
# MAGIC pprint(eval_dataset_guidelines)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Implement Global Guidelines Judge
# MAGIC
# MAGIC Create a global guidelines judge that ensures responses are professional and include specific formatting requirements.

# COMMAND ----------

## Import and configure the Guidelines judge
from mlflow.genai.scorers import <FILL_IN>

## Create a global guideline for professional responses
professional_guideline = <FILL_IN>(
    name=<FILL_IN>,
    guidelines=[
        <FILL_IN>,
        <FILL_IN>
    ],
    model_name=<FILL_IN>
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task D2: Implement Guidelines Judge
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import Guidelines
# MAGIC # Create a global guideline for professional responses
# MAGIC professional_guideline = Guidelines(
# MAGIC     name="professional_response",
# MAGIC     guidelines=[
# MAGIC         "The response should be professional and courteous",
# MAGIC         "The response should include specific numerical values when providing calculations or statistics",
# MAGIC         "The response should clearly indicate when it's using NYC taxi data"
# MAGIC     ],
# MAGIC     model_name=guidelines_endpoint
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Execute Global Guidelines Evaluation
# MAGIC
# MAGIC Run the evaluation using the global guidelines judge. Click on the **View evaluation results with MLflow** to inspect the results.

# COMMAND ----------

## Execute evaluation with global guidelines
guidelines_results = mlflow.genai.evaluate(
    data=<FILL_IN>,
    predict_fn=<FILL_IN>,
    scorers=<FILL_IN>
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task D3: Execute Guidelines Evaluation
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC guidelines_results = mlflow.genai.evaluate(
# MAGIC     data=eval_dataset_guidelines,
# MAGIC     predict_fn=lambda input: agent.predict({"input": input}),
# MAGIC     scorers=[professional_guideline]
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Comprehensive Evaluation Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### E1. Compare Evaluation Results
# MAGIC
# MAGIC Analyze and compare results from different judge types to understand their strengths and use cases.

# COMMAND ----------

## Display results from all evaluation runs for comparison
print("=== BUILT-IN JUDGES RESULTS ===")
print(f"Run ID: {<FILL_IN>}")
print(f"Metrics: {<FILL_IN>}")
display(<FILL_IN>)

print("\n=== GLOBAL GUIDELINES RESULTS ===")
print(f"Run ID: {<FILL_IN>}")
print(f"Metrics: {<FILL_IN>}")
display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task E1: Compare Results
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC print("=== BUILT-IN JUDGES RESULTS ===")
# MAGIC print(f"Run ID: {builtin_results.run_id}")
# MAGIC print(f"Metrics: {builtin_results.metrics}")
# MAGIC display(builtin_results.result_df)
# MAGIC print("\n=== GLOBAL GUIDELINES RESULTS ===")
# MAGIC print(f"Run ID: {guidelines_results.run_id}")
# MAGIC print(f"Metrics: {guidelines_results.metrics}")
# MAGIC display(guidelines_results.result_df)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Create Comprehensive Evaluation
# MAGIC
# MAGIC Combine multiple judge types in a single evaluation run to get comprehensive quality assessment.

# COMMAND ----------

## Create a comprehensive evaluation using multiple judge types
## Use the built-in dataset and combine correctness with global guidelines
comprehensive_results = mlflow.genai.evaluate(
    data=<FILL_IN>,
    predict_fn=<FILL_IN>,
    scorers=<FILL_IN>
)

print("=== COMPREHENSIVE EVALUATION RESULTS ===")
print(f"Run ID: {<FILL_IN>}")
print(f"Metrics: {<FILL_IN>}")
display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task E2: Comprehensive Evaluation
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC comprehensive_results = mlflow.genai.evaluate(
# MAGIC     data=eval_dataset_builtin,
# MAGIC     predict_fn=lambda input: agent.predict({"input": input}),
# MAGIC     scorers=[correctness_eval, professional_guideline]
# MAGIC )
# MAGIC print("=== COMPREHENSIVE EVALUATION RESULTS ===")
# MAGIC print(f"Run ID: {comprehensive_results.run_id}")
# MAGIC print(f"Metrics: {comprehensive_results.metrics}")
# MAGIC display(comprehensive_results.result_df)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC In this lab, you have successfully implemented a comprehensive evaluation framework for AI agents using MLflow's built-in and guideline judges. You learned how to:
# MAGIC
# MAGIC - Configure and execute multiple types of judges for different evaluation needs
# MAGIC - Create properly structured evaluation datasets for various judge types
# MAGIC - Analyze evaluation results using MLflow's tracing and result analysis capabilities
# MAGIC - Apply best practices for evaluation configuration and model management
# MAGIC
# MAGIC These evaluation techniques provide the foundation for ensuring your AI agents meet quality standards and perform reliably in production environments. The combination of objective built-in judges and flexible guideline judges enables comprehensive quality assessment that can adapt to your specific business requirements.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC