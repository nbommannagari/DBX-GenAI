# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Introduction to OpenAI Agents SDK on Databricks
# MAGIC #### Duration: 20 minutes
# MAGIC ## Overview
# MAGIC In this demo, you will build a single Data Analyst agent from scratch using the OpenAI Agents SDK on Databricks. You will define a custom tool with `@function_tool`, create and test the agent, register tools to Unity Catalog for governance, and enable MLflow tracing to observe execution.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC 1. Configure the OpenAI Agents SDK to use Databricks model routing
# MAGIC 2. Define custom tools using `@function_tool`
# MAGIC 3. Build and test a single agent with tools
# MAGIC 4. Register tools to Unity Catalog for governance
# MAGIC 5. Enable MLflow autologging and inspect execution traces

# COMMAND ----------

# DBTITLE 1,REQUIRED - SELECT A COMPUTE ENVIRONMENT
# MAGIC %md-sandbox
# MAGIC ## REQUIRED - SELECT A COMPUTE ENVIRONMENT
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Select Compute</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Before starting this notebook, select the required compute environment listed below.</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Serverless Compute, Version 6</strong>: How to select an environment version (<a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V6</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Classroom Setup

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-5

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Import Libraries
# MAGIC
# MAGIC To get started with the OpenAI SDK, we have a few things that need to be setup:
# MAGIC 1. Configure the OpenAI Agents SDK to use Databricks model routing **using Unity Gateway Model Services** using `AsyncDatabricksOpenAI(use_ai_gateway=True)`.
# MAGIC 1. Import MCP components needed to connect to agent tools that have already been registered to UC.
# MAGIC 1. Initialize the Databricks `WorkspaceClient`.
# MAGIC 1. Set the default trace processor to MLflow.
# MAGIC
# MAGIC Additional documentation and reading:
# MAGIC - [`set_default_openai_client`](https://openai.github.io/openai-agents-python/ref/#agents.set_default_openai_client)
# MAGIC - [`set_default_openai_api`](https://openai.github.io/openai-agents-python/ref/#agents.set_default_openai_api)
# MAGIC - [`set_trace_processors`](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md)

# COMMAND ----------

from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
import logging

set_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))
set_default_openai_api("chat_completions")
set_trace_processors([])  # Disable OpenAI trace processor; use MLflow instead
logging.getLogger("openai").setLevel(logging.WARNING)

print("OpenAI Agents SDK configured to use Databricks model routing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Define the Data Analyst Agent

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Define the `classify_price_tier` Tool
# MAGIC
# MAGIC The `classify_price_tier` tool classifies an Airbnb listing's nightly price into one of four market tiers based on the SF pricing distribution:
# MAGIC
# MAGIC | Tier | Price Range | Percentile Band |
# MAGIC |---|---|---|
# MAGIC | **Budget** | Under $100 | Bottom 25% |
# MAGIC | **Mid-Range** | $100 -- $174 | 25--50% |
# MAGIC | **Premium** | $175 -- $299 | 50--75% |
# MAGIC | **Luxury** | $300+ | Top 25% |
# MAGIC
# MAGIC The `@function_tool` decorator registers this function as a tool that the agent can invoke. The SDK extracts the function signature and docstring to generate the tool schema that the LLM uses to decide when and how to call the tool.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC An example production workflow is: prototype with <code>@function_tool</code>, then register to Unity Catalog, then access via a Managed MCP server. This demo follows the first two steps of that progression. In Section C, we explicitly re-register the function to Unity Catalog for governance.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

import json

@function_tool
def classify_price_tier(price: float) -> str:
    """Classify an Airbnb listing price into a market tier based on SF pricing distribution.

    Uses hardcoded SF Airbnb market thresholds to label a nightly price as
    Budget, Mid-Range, Premium, or Luxury, and returns the corresponding
    percentile band for context.

    Useful for questions like: 'Is $180/night a good deal in SF?',
    'What tier does this listing fall into?'

    Returns a JSON object with tier, percentile_band, and a short interpretation.

    Args:
        price: Nightly listing price in USD.
    """
    import json

    tiers = [
        (100,  "Budget",    "bottom 25%",  "Well below the SF market average."),
        (175,  "Mid-Range", "25–50%",      "Around the SF market median."),
        (300,  "Premium",   "50–75%",      "Above average for SF listings."),
        (float("inf"), "Luxury", "top 25%", "Among the most expensive listings in SF."),
    ]

    if price < 0:
        return json.dumps({"error": "Price must be a positive value."})

    for threshold, tier, band, interpretation in tiers:
        if price < threshold:
            return json.dumps({
                "price": price,
                "tier": tier,
                "percentile_band": band,
                "interpretation": interpretation
            }, indent=2)

print(f"Defined tool: {classify_price_tier.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Create the Agent
# MAGIC
# MAGIC With the tool defined, we create a single agent that uses 
# MAGIC - `classify_price_tier` to answer questions about SF Airbnb pricing
# MAGIC - `instructions` parameter shapes the agent's behavior and tells it when to use the tool
# MAGIC - A Unity Catalog model service, referenced by its fully qualified name `{catalog_name}.{schema_name}.agent-training-model`
# MAGIC
# MAGIC This is accomplished with the `Agent` class. You can read more about `Agent` [here](https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### B2.1 Inspect and Load the Agent Configuration
# MAGIC We will load the agent configuration file. This is a recommended practice since any updates to the agents' system prompts, name, etc. live outside this notebook. These variables will be stored in the dictionary `agents`:
# MAGIC - `agents["X"]['name']`: The agent's name
# MAGIC - `agents["X"]['llm_endpoint']`: The name of the Unity Catalog **model service** the agent calls (`agent-training-model`). The next cell combines it with your catalog and schema into the service's fully qualified name.
# MAGIC - `agents["X"]['system_prompt']`: The agent's system prompt
# MAGIC
# MAGIC Navigate to the folder **agent_config** and inspect the agent contents for **data_analyst_agent.yaml**
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Why a model service?</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC In this course, agents call a <strong>model service</strong> registered in your catalog and schema instead of calling a foundation model endpoint such as <code>databricks-gpt-5-2</code> directly. The classroom setup created the service <code>agent-training-model</code>, which routes requests through Unity Gateway to <code>databricks-gpt-5-2</code>. Because the service is a Unity Catalog object, access to it is governed with Unity Catalog permissions, and because agents reference the service rather than the model, the model behind it can change without changing agent code.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

import yaml
from pathlib import Path

agent_files = {
    "da": "data_analyst_agent"
}

agents = {}
for prefix, filename in agent_files.items():
    cfg = yaml.safe_load(Path(f"./agent_config/{filename}.yaml").read_text())
    agents[prefix] = {
        "name": cfg["name"],
        "llm_endpoint": cfg["llm_endpoint"],
        "system_prompt": cfg["system_prompt"],
    }

# COMMAND ----------

name = agents["da"]["name"]
system_prompt = agents["da"]["system_prompt"]
llm_endpoint = f"{catalog_name}.{schema_name}.{agents["da"]["llm_endpoint"]}"

data_analyst_agent = Agent(
    name=name,
    instructions=system_prompt,
    model=llm_endpoint,
    tools=[classify_price_tier],
)

print(f"Created agent: {data_analyst_agent.name}")
print(f"  Model: {data_analyst_agent.model}")
print(f"  Tools: {[t.name for t in data_analyst_agent.tools]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B3. Test the Agent
# MAGIC
# MAGIC We send three progressively complex queries to validate that the agent correctly invokes the tool and interprets the results. Each call uses `await Runner.run()`, which is the execution interface. It executes the full agent loop: the LLM decides whether to call a tool, executes it if needed, and generates a final response.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. The next cell will run a straightforward pricing question that requires a single tool call.

# COMMAND ----------

query_1 = "Is $150/night a good deal in SF?"
result_1 = await Runner.run(data_analyst_agent, query_1)
print(f"User:     {query_1}")
print(f"Response: {result_1.final_output}")

# COMMAND ----------

# MAGIC %md
# MAGIC 2. The next cell will run a direct tier classification at the high end of the price spectrum.

# COMMAND ----------

query_2 = "What tier would a $500/night luxury listing fall into?"
result_2 = await Runner.run(data_analyst_agent, query_2)
print(f"User:     {query_2}")
print(f"Response: {result_2.final_output}")

# COMMAND ----------

# MAGIC %md
# MAGIC 3. The next cell will run multiple prices in a single request, testing whether the agent calls the tool multiple times.

# COMMAND ----------

query_3 = "Classify these prices: $80, $200, $350"
result_3 = await Runner.run(data_analyst_agent, query_3)
print(f"User:     {query_3}")
print(f"Response: {result_3.final_output}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Register Tools to Unity Catalog

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. Register `classify_price_tier` as a Python UC Function
# MAGIC
# MAGIC The inline `@function_tool` is local to this notebook session. To make the tool governed, discoverable, and reusable across agents and workspaces, we register it to Unity Catalog as a Python function using `DatabricksFunctionClient`.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC The function registered below is a <strong>plain Python function</strong> (no <code>@function_tool</code> decorator). The <code>DatabricksFunctionClient</code> inspects the function signature and docstring to create the UC function metadata. The logic is identical to the tool we prototyped above.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

def classify_price_uc(price: float) -> str:
    """Classify an Airbnb listing price into a market tier based on SF pricing distribution.

    Uses hardcoded SF Airbnb market thresholds to label a nightly price as
    Budget, Mid-Range, Premium, or Luxury, and returns the corresponding
    percentile band for context.

    Useful for questions like: 'Is $180/night a good deal in SF?',
    'What tier does this listing fall into?'

    Returns a JSON object with tier, percentile_band, and a short interpretation.

    Args:
        price: Nightly listing price in USD.
    """
    import json

    tiers = [
        (100,  "Budget",    "bottom 25%",  "Well below the SF market average."),
        (175,  "Mid-Range", "25–50%",      "Around the SF market median."),
        (300,  "Premium",   "50–75%",      "Above average for SF listings."),
        (float("inf"), "Luxury", "top 25%", "Among the most expensive listings in SF."),
    ]

    if price < 0:
        return json.dumps({"error": "Price must be a positive value."})

    for threshold, tier, band, interpretation in tiers:
        if price < threshold:
            return json.dumps({
                "price": price,
                "tier": tier,
                "percentile_band": band,
                "interpretation": interpretation
            }, indent=2)

client = DatabricksFunctionClient()
function_info = client.create_python_function(
    func=classify_price_uc,
    catalog=catalog_name,
    schema=schema_name,
    replace=True,
)

print(f"Created UC function: {function_info.full_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Verify Registered Functions
# MAGIC
# MAGIC Confirm that the function is visible in the Unity Catalog schema, then call it directly to verify it returns the expected output.
# MAGIC 1. Navigate to the **Catalog Explorer** on the left side menu and search for `classify_price_uc`
# MAGIC 1. Run the next cell to test the output from calling the function from UC

# COMMAND ----------

# Call classify_price_tier directly to verify
result = spark.sql(f"SELECT {catalog_name}.{schema_name}.classify_price_uc(150.0) AS result")
display(result)

# COMMAND ----------

# DBTITLE 1,C3. Access the UC Function via a Managed MCP Server
# MAGIC %md-sandbox
# MAGIC ### C3. Access the UC Function via a Managed MCP Server
# MAGIC
# MAGIC Databricks provides **Managed MCP servers** that automatically expose Unity Catalog functions as tools. Any agent can discover and call these functions through the MCP protocol without custom integration code.
# MAGIC
# MAGIC The MCP server endpoint for UC functions follows this pattern:
# MAGIC `https://workspace/api/2.0/mcp/functions/catalog/schema`
# MAGIC
# MAGIC With the OpenAI Agents SDK, you use `McpServer` from `databricks_openai.agents` to connect as shown in the next cell.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC The <code>McpServer</code> class from <code>databricks_openai.agents</code> is an async context manager designed for the Databricks Apps runtime. In a notebook, we use <code>DatabricksMCPClient</code> from <code>databricks_mcp</code> to interact with the same MCP endpoint.
# MAGIC See: Managed MCP servers (<a href="https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/managed-mcp" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/generative-ai/mcp/managed-mcp" style="color: #1976d2; text-decoration: underline;">GCP</a>)
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

ws = WorkspaceClient()
workspace_host = ws.config.host

# Point to the MCP server for our catalog/schema
mcp_url = f"{workspace_host}/api/2.0/mcp/functions/{catalog_name}/{schema_name}"
mcp_client = DatabricksMCPClient(server_url=mcp_url, workspace_client=ws)

# Discover available tools (nest_asyncio patches the event loop for DatabricksMCPClient)
tools = mcp_client.list_tools()
print(f"Discovered {len(tools)} UC function tool(s):")
for t in tools:
    print(f"  - {t.name}: {t.description}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C4. Call the UC Function Through MCP
# MAGIC
# MAGIC Now we call the registered UC function through the MCP protocol, just as an agent would. The default tool name is of the form `catalogName__schemaName__toolName` as demonstrated in the next cell.

# COMMAND ----------

# Call the UC function via MCP
tool_name = f"{catalog_name}__{schema_name}__classify_price_uc"
result = mcp_client.call_tool(tool_name, {"price": 120.00})
print(f"Result: {''.join([c.text for c in result.content])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Enable MLflow Tracing

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D1. Enable Autologging
# MAGIC
# MAGIC MLflow autologging for the OpenAI integration automatically captures execution traces for every agent run by patching the OpenAI SDK so that every call to various methods automatically generates MLflow traces. Each trace records the full sequence of LLM calls, tool invocations, inputs, outputs, and latencies without requiring any code changes to the agent itself.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Autologging captures: LLM call spans (model, prompt, completion tokens), tool call spans (function name, arguments, return value), and the overall agent run span with total latency. All traces are stored in the MLflow experiment associated with this notebook.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

import mlflow

mlflow.openai.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Registering Traces to Unity Catalog
# MAGIC At this point, our traces haven't been logged. This means we have no insight into things like token usage, execution time per tool call, etc. 
# MAGIC
# MAGIC Next, we'll show how to register traces in Unity Catalog. Databricks recommends this pattern for new and production workloads. 

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Instantiate the `UnityCatalog` class from `mlflow` and call it `uc`.

# COMMAND ----------

from mlflow.entities.trace_location import UnityCatalog

uc = UnityCatalog(
    catalog_name = catalog_name,
    schema_name = schema_name,
    table_prefix = "oai_sa" # oai_sa: "OpenAI Single Agent"
)

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Create and set the experiment location using `mlflow.set_experiment()`. In the code below, we will use `dbutils`, which is a Databricks utility library that allows users to do things like get the current notebook path within the workspace.

# COMMAND ----------

experiment_path = f"/Users/{username}/oai-sa-sdk-training"

mlflow.set_experiment(
    experiment_name=experiment_path,
    trace_location=uc,
)

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Navigate to Unity Catalog and see that we now have multiple tables and views with the prefix we set earlier, `oai_sa` ("OpenAI Single Agent"). These tables are empty since we haven't made any calls with our agent. Let's stress test the agent with and without MCP tools. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Tracing without MCP
# MAGIC
# MAGIC With autologging enabled, the next agent run will produce a full execution trace, which we will inspect in a moment. The trace captures the LLM reasoning step, the tool call to `classify_price_tier`, and the final response generation.

# COMMAND ----------

traced_query = "I found a listing for $225/night in the Mission district. Is that a fair price?"
result = await Runner.run(data_analyst_agent, traced_query)
print(f"User:     {traced_query}")
print(f"Response: {result.final_output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Navigate to the `oai_sa_otel_spans` table and click on the **Sample Data** tab. In the **name** column you will see `data_analyst_mcp_execution` and `AsyncDatabricksCompletions` This course does not go into details on spans and OTel concepts, but you can read more here ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/trace-unity-catalog) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/trace-unity-catalog)).
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Why Two Traces?</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC You may notice that a single agent query produces <strong>two separate traces</strong> in MLflow. This is expected behavior with <code>mlflow.openai.autolog()</code>. Because autologging patches at the <strong>OpenAI client level</strong>, each individual API call becomes its own top-level trace. A single <code>Runner.run()</code> call typically makes two underlying LLM calls:
# MAGIC </p>
# MAGIC <ol style="margin: 8px 0 0 16px; color: #333;">
# MAGIC <li><strong>First LLM call</strong>: The model receives the user query, reasons about it, and decides to invoke the <code>classify_price_tier</code> tool.</li>
# MAGIC <li><strong>Second LLM call</strong>: After the tool returns its result, the model generates the final natural-language response.</li>
# MAGIC </ol>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Without a parent span grouping them, each call appears as a separate trace. The next section solves this by wrapping the full agent run in <code>@mlflow.trace()</code>, which concatenates all spans into <strong>one unified trace</strong>.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# DBTITLE 1,D3. Tracing with MCP
# MAGIC %md-sandbox
# MAGIC ### D4. Tracing with MCP
# MAGIC
# MAGIC In a Databricks App deployment, the `McpServer` from `databricks_openai.agents` is used as an async context manager with the `Agent()` class.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="background:#F9F7F4;border:2px solid #1B5162;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;font-weight:600;">@function_tool<br/><span style="font-weight:400;font-size:12px;">Local only</span></div>
# MAGIC   <div style="color:#999;font-size:22px;padding:0 8px;">&#x2192;</div>
# MAGIC   <div style="background:#2574B5;color:#fff;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;font-weight:600;">UC Function<br/><span style="font-weight:400;font-size:12px;">Governed + reusable</span></div>
# MAGIC   <div style="color:#999;font-size:22px;padding:0 8px;">&#x2192;</div>
# MAGIC   <div style="background:#02A36F;color:#fff;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;font-weight:600;">MCP Server<br/><span style="font-weight:400;font-size:12px;">Agent-accessible + standardized</span></div>
# MAGIC   <div style="color:#999;font-size:22px;padding:0 8px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;font-weight:600;">Agent<br/><span style="font-weight:400;font-size:12px;">Ready for testing</span></div>
# MAGIC </div>
# MAGIC
# MAGIC This is the recommended progression: start with `@function_tool` for rapid prototyping, register to UC for governance, then expose through MCP for production agent access.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Run the next cell to expose only `classify_price_uc` to the agent. 

# COMMAND ----------

from databricks_openai.agents import McpServer

mcp_server = McpServer.from_uc_function(
    catalog=catalog_name,
    schema=schema_name,
    function_name="classify_price_uc",
    timeout=60.0,
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 2. **Redefine our agent** from before, but this time we pass `mcp_servers=[mcp_server]` instead of `tools`. The agent now discovers its tools at runtime from the UC function MCP server rather than from inline `@function_tool` definitions.
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC Why do we need @mlflow.trace, async def, and async?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC <ul style="margin: 0; padding-left: 20px;">
# MAGIC <li><strong><code>@mlflow.trace(...)</code></strong>: Without this, the MCP connection, LLM call, and tool call each produce separate traces. Wrapping them in a single decorated function concatenates all spans into one unified trace.</li>
# MAGIC <li><strong><code>async def</code></strong>: <code>Runner.run()</code> is async (it makes HTTP calls to the model endpoint), and <code>McpServer</code> is an async context manager. Both require <code>await</code>, which means the enclosing function must be <code>async def</code>.</li>
# MAGIC <li><strong><code>async with mcp_server:</code></strong>: Opens the HTTP connection to the Databricks Managed MCP server before the agent runs, and closes it after. The agent calls <code>tools/list</code> and <code>tools/call</code> over this connection during execution.</li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

data_analyst_mcp_agent_update = Agent(
    name="Data Analyst",
    instructions="You are a data analyst that helps users understand SF Airbnb listing prices. Use the classify_price_tier tool to categorize prices into market tiers. Provide clear, concise answers.",
    model=llm_endpoint,  # the same model service as the first agent
    mcp_servers=[mcp_server], # <---- used to be tools exposed via @function_tool
)

# Traced wrapper that concatenates all spans into a single trace
@mlflow.trace(span_type="AGENT", name="data_analyst_mcp_execution")
async def run_data_analyst_mcp(question: str) -> str:
    """Run the Data Analyst agent with MCP tools as a single traced operation."""
    async with mcp_server:
        result = await Runner.run(data_analyst_mcp_agent_update, question)
        return result.final_output

# COMMAND ----------

response = await run_data_analyst_mcp(query_1)
print(f"User:     {query_1}")
print(f"Response: {response}")

# COMMAND ----------

# DBTITLE 1,D5. View MLflow Traces
# MAGIC %md
# MAGIC ### D5. View MLflow Traces
# MAGIC Finally, now that we have some traces logged with MLflow, we can view the traces in either the **Experiments** UI or directly from the unified trace table stored in the schema `agent_training` with the table prefix `oai_sa`. 
# MAGIC
# MAGIC #### Experiment UI
# MAGIC 1. Navigate to **Experiments** in the left sidebar, right click, and select **Open Link in New Tab**. 
# MAGIC 2. Search for the experiment **oai-sa-sdk-training**. There you will see traces under the **Traces** tab. Click any trace to see a detailed breakdown of tools used and model calls.
# MAGIC #### MLflow Unified Tables from Unity Catalog
# MAGIC 1. Navigate to the **Catalog Explorer** and search for **oai_sa**. This surface 6 default tables that have been created. 
# MAGIC     - 4 tables: **oai_sa_otel_logs**, **oai_sa_otel_spans**, **oai_sa_otel_annotations**, and **oai_sa_otel_annotations**
# MAGIC     - 2 views: **oai_sa_trace_unified**, and **oai_sa_trace_metadata**
# MAGIC 2. Click on the **_otel_spans** table and click on **Sample Data** and ask a question in the provided query box like _Which trace took the longest to run?_ This will automatically generate a sql query (you might be asked to grant auto-run, click accept). The answer will be returned to you.
# MAGIC 3. Run the next cell to see logged traces here in the notebook. Notice the same information is captured in the column `spans`.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM oai_sa_trace_unified

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Conclusion
# MAGIC
# MAGIC In this demo, you:
# MAGIC
# MAGIC 1. Configured the OpenAI Agents SDK to call a Unity Catalog model service through Unity Gateway
# MAGIC 2. Defined a `classify_price_tier` tool using the `@function_tool` decorator to classify Airbnb prices into market tiers
# MAGIC 3. Built a single Data Analyst agent and tested it with progressively complex pricing queries
# MAGIC 4. Registered `classify_price_tier` as a Python UC function for governance and connected it to the agent via MCP
# MAGIC 5. Enabled MLflow autologging and inspected execution traces capturing LLM calls, tool invocations, and response generation

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>