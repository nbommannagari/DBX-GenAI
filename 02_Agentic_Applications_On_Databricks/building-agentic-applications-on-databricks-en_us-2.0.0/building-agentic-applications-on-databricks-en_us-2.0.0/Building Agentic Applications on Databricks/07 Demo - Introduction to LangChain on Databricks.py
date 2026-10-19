# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Demo - Introduction to LangChain on Databricks
# MAGIC #### Duration: 20 minutes
# MAGIC ## Overview
# MAGIC In this demo, you will build a single Data Analyst agent using LangChain on Databricks. You will define a custom tool with LangChain's `@tool` decorator, register it to Unity Catalog, discover it via a Managed MCP server, and connect it to a LangChain agent with MLflow tracing.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC 1. Define custom tools using LangChain's `@tool` decorator
# MAGIC 2. Build and test a LangChain agent with `create_agent`
# MAGIC 3. Register tools to Unity Catalog for governance
# MAGIC 4. Discover UC tools via Managed MCP servers
# MAGIC 5. Rebuild the agent with MCP-sourced tools
# MAGIC 6. Enable MLflow autologging and inspect execution traces
# MAGIC
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC This notebook mirrors notebook <strong>05 Demo - Introduction to OpenAI Agents SDK on Databricks</strong>, using LangChain instead of the OpenAI Agents SDK.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

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

# MAGIC %run ./Includes/Classroom-Setup-7

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Import Libraries
# MAGIC
# MAGIC We import the core components needed for this demo:
# MAGIC - `ChatDatabricks` for connecting to Databricks model serving endpoints
# MAGIC - `@tool` decorator for defining agent tools
# MAGIC - `create_agent` from `langchain.agents` for building the agent

# COMMAND ----------

from langchain.agents import create_agent
from langchain_core.tools import tool
from databricks_langchain import ChatDatabricks

print("LangChain components imported.")

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
# MAGIC LangChain's `@tool` decorator registers this function as a tool that the agent can invoke. The SDK extracts the function signature and docstring to generate the tool schema that the LLM uses to decide when and how to call the tool.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC An example production workflow is: prototype with <code>@tool</code>, then register to Unity Catalog, then access via a Managed MCP server. This demo follows that full progression. In Section C, we register the function to Unity Catalog. In Section D, we connect via MCP.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

import json

@tool
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
    tiers = [
        (100,  "Budget",    "bottom 25%",  "Well below the SF market average."),
        (175,  "Mid-Range", "25-50%",      "Around the SF market median."),
        (300,  "Premium",   "50-75%",      "Above average for SF listings."),
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
# MAGIC With the tool defined, we create an agent using LangChain's `create_agent`. This takes three inputs:
# MAGIC 1. **LLM**: `ChatDatabricks` connects to a Databricks model serving endpoint
# MAGIC 2. **Tools**: The list of tools the agent can call
# MAGIC 3. **System prompt**: Instructions that shape the agent's behavior, passed as `system_prompt`
# MAGIC
# MAGIC The agent runs a ReAct loop: the LLM reasons about which tool to call, executes it, observes the result, and generates a final response. `create_agent` replaces LangGraph's `create_react_agent`, which is deprecated as of LangGraph 1.0, and still runs on LangGraph under the hood.

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

system_prompt = agents["da"]["system_prompt"]
llm_endpoint = f"{catalog_name}.{schema_name}.{agents["da"]["llm_endpoint"]}"

llm = ChatDatabricks(
    model=llm_endpoint,
    use_ai_gateway=True
)

agent_executor = create_agent(llm, tools=[classify_price_tier], system_prompt=system_prompt)

print("Agent created with inline @tool.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B3. Test the Agent
# MAGIC
# MAGIC We send three progressively complex queries to validate that the agent correctly invokes the tool and interprets the results.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. A straightforward pricing question that requires a single tool call.

# COMMAND ----------

query_1 = "Is $150/night a good deal in SF?"
result_1 = agent_executor.invoke({"messages": [("human", query_1)]})
print(f"User:     {query_1}")
print(f"Response: {result_1['messages'][-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC 2. A direct tier classification at the high end of the price spectrum.

# COMMAND ----------

query_2 = "What tier would a $500/night luxury listing fall into?"
result_2 = agent_executor.invoke({"messages": [("human", query_2)]})
print(f"User:     {query_2}")
print(f"Response: {result_2['messages'][-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Multiple prices in a single request, testing whether the agent calls the tool multiple times.

# COMMAND ----------

query_3 = "Classify these prices: $80, $200, $350"
result_3 = agent_executor.invoke({"messages": [("human", query_3)]})
print(f"User:     {query_3}")
print(f"Response: {result_3['messages'][-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Register Tools to Unity Catalog

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. Register `classify_price_tier` as a Python UC Function
# MAGIC
# MAGIC The inline `@tool` is local to this notebook session. To make the tool governed, discoverable, and reusable across agents and workspaces, we register it to Unity Catalog as a Python function using `DatabricksFunctionClient`.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC The function registered below is a <strong>plain Python function</strong> (no <code>@tool</code> decorator). The <code>DatabricksFunctionClient</code> inspects the function signature and docstring to create the UC function metadata. The logic is identical to the tool we prototyped above.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

def classify_price_tier_uc(price: float) -> str:
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
        (175,  "Mid-Range", "25-50%",      "Around the SF market median."),
        (300,  "Premium",   "50-75%",      "Above average for SF listings."),
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
    func=classify_price_tier_uc,
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
# MAGIC 1. Navigate to the **Catalog Explorer** on the left side menu and search for `classify_price_tier_uc`
# MAGIC 1. Run the next cell to test the output from calling the function from UC

# COMMAND ----------

result = spark.sql(f"SELECT {catalog_name}.{schema_name}.classify_price_tier_uc(150.0) AS result")
display(result)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Discover Tools via MCP

# COMMAND ----------

# DBTITLE 1,D1. Connect to the Managed MCP Server
# MAGIC %md-sandbox
# MAGIC ### D1. Connect to the Managed MCP Server
# MAGIC
# MAGIC Databricks provides **Managed MCP servers** that automatically expose Unity Catalog functions as tools. Any agent can discover and call these functions through the MCP protocol without custom integration code.
# MAGIC
# MAGIC The MCP server endpoint for UC functions follows this pattern:
# MAGIC `https://workspace/api/2.0/mcp/functions/catalog/schema`
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC We use <code>DatabricksMCPClient</code> from <code>databricks_mcp</code> to interact with the MCP endpoint in a notebook context.
# MAGIC See: Managed MCP servers (<a href="https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/managed-mcp" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/generative-ai/mcp/managed-mcp" style="color: #1976d2; text-decoration: underline;">GCP</a>)
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

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

# Discover available tools
tools_discovered = mcp_client.list_tools()
print(f"Discovered {len(tools_discovered)} UC function tool(s) via MCP:")
for t in tools_discovered:
    print(f"  - {t.name}: {t.description}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Call the UC Function Through MCP
# MAGIC
# MAGIC Now we call the registered UC function through the MCP protocol, just as an agent would. The default tool name is of the form `catalogName__schemaName__toolName`.

# COMMAND ----------

tool_name = f"{catalog_name}__{schema_name}__classify_price_tier_uc"
result = mcp_client.call_tool(tool_name, {"price": 120.00})
print(f"Result: {''.join([c.text for c in result.content])}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D3. Load MCP Tools into LangChain
# MAGIC
# MAGIC We convert the MCP tools we discovered into LangChain-compatible `StructuredTool` objects. Each tool wraps a call to `mcp_client.call_tool()`, so tool invocations route through MCP at runtime.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;gap:0;margin:20px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="background:#F9F7F4;border:2px solid #1B5162;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;font-weight:600;">@tool<br/><span style="font-weight:400;font-size:12px;">Local only</span></div>
# MAGIC   <div style="color:#999;font-size:22px;padding:0 8px;">&#x2192;</div>
# MAGIC   <div style="background:#2574B5;color:#fff;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;font-weight:600;">UC Function<br/><span style="font-weight:400;font-size:12px;">Governed + reusable</span></div>
# MAGIC   <div style="color:#999;font-size:22px;padding:0 8px;">&#x2192;</div>
# MAGIC   <div style="background:#02A36F;color:#fff;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;font-weight:600;">MCP Server<br/><span style="font-weight:400;font-size:12px;">Agent-accessible + standardized</span></div>
# MAGIC   <div style="color:#999;font-size:22px;padding:0 8px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;font-weight:600;">LangChain Agent<br/><span style="font-weight:400;font-size:12px;">Production-ready</span></div>
# MAGIC </div>
# MAGIC
# MAGIC This is the recommended progression: start with `@tool` for rapid prototyping, register to UC for governance, then expose through MCP for production agent access.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC The next cell defines `_make_mcp_tool`, which converts each MCP tool into a LangChain `StructuredTool`. Before you run it, click each pill at the top to highlight that part of the code and read what it does.
# MAGIC
# MAGIC <div id="mcp-tool-explorer"></div>
# MAGIC <style>
# MAGIC   .sx-wrap-py {
# MAGIC     position: relative;
# MAGIC     font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
# MAGIC     border: 1px solid #DCE0E2;
# MAGIC     border-radius: 12px;
# MAGIC     overflow: hidden;
# MAGIC     background: #F9F7F4;
# MAGIC     margin: 16px auto;
# MAGIC     max-width: 100%;
# MAGIC     box-shadow: 0 2px 8px rgba(27,49,57,0.06);
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-title {
# MAGIC     display: flex;
# MAGIC     justify-content: space-between;
# MAGIC     align-items: center;
# MAGIC     background: #EEEDE9;
# MAGIC     border-bottom: 1px solid #DCE0E2;
# MAGIC     padding: 10px 16px;
# MAGIC     font-size: 14px;
# MAGIC     font-weight: 700;
# MAGIC     color: #5A6F77;
# MAGIC     letter-spacing: 0.02em;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-rail {
# MAGIC     display: flex;
# MAGIC     flex-wrap: wrap;
# MAGIC     gap: 8px;
# MAGIC     padding: 12px 16px;
# MAGIC     background: #F9F7F4;
# MAGIC     border-bottom: 1px solid #EEEDE9;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-chip {
# MAGIC     border: 1px solid #DCE0E2;
# MAGIC     background: #FFFFFF;
# MAGIC     border-radius: 999px;
# MAGIC     padding: 6px 12px;
# MAGIC     font-size: 14px;
# MAGIC     font-weight: 700;
# MAGIC     color: #0b2026;
# MAGIC     cursor: pointer;
# MAGIC     transition: background 0.12s, border-color 0.12s, color 0.12s;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-chip:hover,
# MAGIC   .sx-wrap-py .sx-chip.active {
# MAGIC     background: #E4F0FA;
# MAGIC     border-color: #2272B4;
# MAGIC     color: #1B5162;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-code {
# MAGIC     padding: 12px 0;
# MAGIC     overflow-x: auto;
# MAGIC     font-size: 14px;
# MAGIC     line-height: 1.6;
# MAGIC     background: #F9F7F4;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-code,
# MAGIC   .sx-wrap-py .sx-line-code {
# MAGIC     font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-line {
# MAGIC     display: grid;
# MAGIC     grid-template-columns: 50px 1fr;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-line-no {
# MAGIC     color: #618794;
# MAGIC     text-align: right;
# MAGIC     padding: 0 12px;
# MAGIC     user-select: none;
# MAGIC     border-right: 1px solid #EEEDE9;
# MAGIC     font-weight: 700;
# MAGIC     font-size: 14px;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-line-code {
# MAGIC     padding: 0 16px;
# MAGIC     white-space: pre;
# MAGIC     color: #0b2026;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-hotspot { cursor: pointer; }
# MAGIC   .sx-wrap-py .sx-line.sx-active-line { background: #E4F0FA; }
# MAGIC   .sx-wrap-py .sx-active-line .sx-line-code {
# MAGIC     border-left: 3px solid #2272B4;
# MAGIC     padding-left: 13px;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-active-line .sx-line-no { color: #1B5162; }
# MAGIC   .sx-wrap-py .sx-overlay {
# MAGIC     display: none;
# MAGIC     position: absolute;
# MAGIC     right: 18px;
# MAGIC     top: 110px;
# MAGIC     width: 340px;
# MAGIC     max-width: calc(100% - 36px);
# MAGIC     background: #E4F0FA;
# MAGIC     border: 1px solid #4299E0;
# MAGIC     border-left: 5px solid #2272B4;
# MAGIC     border-radius: 10px;
# MAGIC     box-shadow: 0 6px 18px rgba(27,49,57,0.18), 0 16px 36px rgba(27,49,57,0.14);
# MAGIC     padding: 14px 16px;
# MAGIC     z-index: 5;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-overlay-title {
# MAGIC     font-weight: 800;
# MAGIC     margin-bottom: 8px;
# MAGIC     color: #1B5162;
# MAGIC     font-size: 16px;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-overlay-body {
# MAGIC     color: #0b2026;
# MAGIC     line-height: 1.5;
# MAGIC     font-size: 14px;
# MAGIC   }
# MAGIC   .sx-wrap-py .sx-overlay-body code {
# MAGIC     font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
# MAGIC     background: #FFFFFF;
# MAGIC     border: 1px solid #C4CCD6;
# MAGIC     color: #1B5162;
# MAGIC     padding: 1px 5px;
# MAGIC     border-radius: 4px;
# MAGIC     font-size: 14px;
# MAGIC   }
# MAGIC </style>
# MAGIC <script>
# MAGIC (function () {
# MAGIC   var config = {
# MAGIC   "mountId": "mcp-tool-explorer",
# MAGIC   "title": "PYTHON",
# MAGIC   "hotspots": [
# MAGIC     {
# MAGIC       "title": "imports",
# MAGIC       "startLine": 1,
# MAGIC       "endLine": 2,
# MAGIC       "description": "<code>StructuredTool</code> is LangChain's tool class for a function with a typed argument schema. Pydantic's <code>create_model</code> and <code>Field</code> build that schema at runtime, because we only learn the arguments when MCP tells us."
# MAGIC     },
# MAGIC     {
# MAGIC       "title": "type mapping",
# MAGIC       "startLine": 4,
# MAGIC       "endLine": 5,
# MAGIC       "description": "MCP servers describe each argument with a JSON Schema type such as <code>\"number\"</code> or <code>\"string\"</code>. This lookup turns those into the Python types LangChain expects. Anything unrecognized falls back to <code>str</code>."
# MAGIC     },
# MAGIC     {
# MAGIC       "title": "one tool at a time",
# MAGIC       "startLine": 7,
# MAGIC       "endLine": 9,
# MAGIC       "description": "The function converts a single tool returned by <code>list_tools()</code>. Its name has the form <code>catalog__schema__function</code>, and it becomes the tool name the LLM sees."
# MAGIC     },
# MAGIC     {
# MAGIC       "title": "read the input schema",
# MAGIC       "startLine": 11,
# MAGIC       "endLine": 12,
# MAGIC       "description": "<code>inputSchema</code> is the JSON Schema the MCP server returned for this tool. Databricks generates it from the Unity Catalog function's parameters."
# MAGIC     },
# MAGIC     {
# MAGIC       "title": "build each field",
# MAGIC       "startLine": 13,
# MAGIC       "endLine": 19,
# MAGIC       "description": "Each argument becomes a <code>(python_type, Field(description=...))</code> pair. The description comes from the UC function's documentation, so the docstring you wrote when registering it reaches the LLM here."
# MAGIC     },
# MAGIC     {
# MAGIC       "title": "args model",
# MAGIC       "startLine": 20,
# MAGIC       "endLine": 20,
# MAGIC       "description": "<code>create_model</code> builds a Pydantic class on the fly, for example <code>classify_price_tier_uc_args(price: float)</code>. LangChain turns it into the argument schema the LLM fills in, and Pydantic validates the values before the call."
# MAGIC     },
# MAGIC     {
# MAGIC       "title": "forward calls to MCP",
# MAGIC       "startLine": 22,
# MAGIC       "endLine": 25,
# MAGIC       "description": "The LangChain tool has no logic of its own. <code>_invoke</code> forwards the arguments to <code>mcp_client.call_tool()</code> and joins the text parts of the MCP response. The UC function still runs in Databricks, under Unity Catalog permissions."
# MAGIC     },
# MAGIC     {
# MAGIC       "title": "assemble the tool",
# MAGIC       "startLine": 27,
# MAGIC       "endLine": 32,
# MAGIC       "description": "<code>StructuredTool.from_function</code> combines the pieces: the function to run, the tool name, the description from the UC function's comment, and the argument schema."
# MAGIC     },
# MAGIC     {
# MAGIC       "title": "convert every tool",
# MAGIC       "startLine": 34,
# MAGIC       "endLine": 38,
# MAGIC       "description": "Apply the wrapper to every tool the MCP server exposed, then list the results. These tools are passed to <code>create_agent</code> in section E, exactly like the inline <code>@tool</code> earlier."
# MAGIC     }
# MAGIC   ],
# MAGIC   "source": "from langchain_core.tools import StructuredTool\nfrom pydantic import Field, create_model\n\n# MCP describes arguments with JSON Schema types; LangChain needs Python types\nJSON_TO_PYTHON = {\"string\": str, \"number\": float, \"integer\": int, \"boolean\": bool}\n\ndef _make_mcp_tool(mcp_tool_meta):\n    \"\"\"Wrap a single MCP tool as a LangChain StructuredTool.\"\"\"\n    tool_name = mcp_tool_meta.name\n\n    # Build a Pydantic model from the MCP tool's input schema\n    schema_props = mcp_tool_meta.inputSchema.get(\"properties\", {})\n    field_definitions = {}\n    for prop_name, prop_info in schema_props.items():\n        prop_type = JSON_TO_PYTHON.get(prop_info.get(\"type\", \"string\"), str)\n        field_definitions[prop_name] = (\n            prop_type,\n            Field(description=prop_info.get(\"description\", \"\")),\n        )\n    args_model = create_model(f\"{tool_name}_args\", **field_definitions)\n\n    # Every call to the LangChain tool is forwarded to the MCP server\n    def _invoke(**kwargs):\n        result = mcp_client.call_tool(tool_name, kwargs)\n        return \"\".join([c.text for c in result.content])\n\n    return StructuredTool.from_function(\n        func=_invoke,\n        name=tool_name,\n        description=mcp_tool_meta.description or \"\",\n        args_schema=args_model,\n    )\n\nmcp_langchain_tools = [_make_mcp_tool(t) for t in tools_discovered]\n\nprint(f\"Loaded {len(mcp_langchain_tools)} MCP tool(s) as LangChain tools:\")\nfor t in mcp_langchain_tools:\n    print(f\"  - {t.name}\")"
# MAGIC };
# MAGIC   function escapeHtml(v) {
# MAGIC     return String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
# MAGIC   }
# MAGIC   var mount = document.getElementById(config.mountId);
# MAGIC   var lines = config.source.split("\n");
# MAGIC   var hotspots = config.hotspots;
# MAGIC   function hotspotFor(lineNo) {
# MAGIC     for (var i = 0; i < hotspots.length; i++) {
# MAGIC       if (lineNo >= hotspots[i].startLine && lineNo <= hotspots[i].endLine) return { h: hotspots[i], i: i };
# MAGIC     }
# MAGIC     return null;
# MAGIC   }
# MAGIC   var codeHtml = lines.map(function(line, idx) {
# MAGIC     var lineNo = idx + 1;
# MAGIC     var hs = hotspotFor(lineNo);
# MAGIC     var cls = hs ? "sx-line sx-hotspot sx-hotspot-" + hs.i : "sx-line";
# MAGIC     var attr = hs ? ' data-hotspot-index="' + hs.i + '"' : "";
# MAGIC     return '<div class="' + cls + '"' + attr + '><div class="sx-line-no">' + lineNo + '</div><div class="sx-line-code">' + escapeHtml(line) + '</div></div>';
# MAGIC   }).join("");
# MAGIC   var railHtml = '<div class="sx-rail">' + hotspots.map(function(h, i) {
# MAGIC     return '<button class="sx-chip" data-hotspot-index="' + i + '">' + escapeHtml(h.title) + '</button>';
# MAGIC   }).join("") + '</div>';
# MAGIC   mount.innerHTML = '<div class="sx-wrap-py"><div class="sx-title"><span>' + escapeHtml(config.title) + '</span></div>' + railHtml + '<div class="sx-code">' + codeHtml + '</div><div class="sx-overlay"><div class="sx-overlay-title"></div><div class="sx-overlay-body"></div></div></div>';
# MAGIC   var overlay = mount.querySelector(".sx-overlay");
# MAGIC   var overlayTitle = mount.querySelector(".sx-overlay-title");
# MAGIC   var overlayBody = mount.querySelector(".sx-overlay-body");
# MAGIC   var activeIdx = null;
# MAGIC   function clearActive() {
# MAGIC     mount.querySelectorAll(".sx-active-line").forEach(function(el) { el.classList.remove("sx-active-line"); });
# MAGIC     mount.querySelectorAll(".sx-chip.active").forEach(function(el) { el.classList.remove("active"); });
# MAGIC   }
# MAGIC   function activate(idx) {
# MAGIC     clearActive();
# MAGIC     if (idx === null) return;
# MAGIC     mount.querySelectorAll(".sx-hotspot-" + idx).forEach(function(el) { el.classList.add("sx-active-line"); });
# MAGIC     mount.querySelector('.sx-chip[data-hotspot-index="' + idx + '"]').classList.add("active");
# MAGIC   }
# MAGIC   function showHotspot(idx, triggerEl) {
# MAGIC     if (overlay.style.display === "block" && activeIdx === idx) {
# MAGIC       overlay.style.display = "none"; activeIdx = null; clearActive(); return;
# MAGIC     }
# MAGIC     var h = hotspots[idx];
# MAGIC     overlayTitle.textContent = h.title;
# MAGIC     overlayBody.innerHTML = h.description;
# MAGIC     overlay.style.display = "block";
# MAGIC     var wrap = mount.querySelector(".sx-wrap-py");
# MAGIC     var trigger = triggerEl || mount.querySelector(".sx-hotspot-" + idx);
# MAGIC     if (trigger) {
# MAGIC       var wrapRect = wrap.getBoundingClientRect();
# MAGIC       var trigRect = trigger.getBoundingClientRect();
# MAGIC       var top = trigRect.top - wrapRect.top;
# MAGIC       overlay.style.top = Math.max(18, Math.min(top, wrap.clientHeight - overlay.offsetHeight - 18)) + "px";
# MAGIC     }
# MAGIC     activeIdx = idx;
# MAGIC     activate(idx);
# MAGIC   }
# MAGIC   mount.querySelectorAll(".sx-hotspot").forEach(function(el) {
# MAGIC     el.addEventListener("click", function() { showHotspot(Number(el.dataset.hotspotIndex), el); });
# MAGIC   });
# MAGIC   mount.querySelectorAll(".sx-chip").forEach(function(chip) {
# MAGIC     chip.addEventListener("click", function() { showHotspot(Number(chip.dataset.hotspotIndex), null); });
# MAGIC   });
# MAGIC   mount.querySelector(".sx-wrap-py").addEventListener("click", function(e) {
# MAGIC     if (e.target.closest(".sx-hotspot") || e.target.closest(".sx-chip") || e.target.closest(".sx-overlay")) return;
# MAGIC     overlay.style.display = "none"; activeIdx = null; clearActive();
# MAGIC   });
# MAGIC   showHotspot(0, null);
# MAGIC })();
# MAGIC </script>

# COMMAND ----------

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

# MCP describes arguments with JSON Schema types; LangChain needs Python types
JSON_TO_PYTHON = {"string": str, "number": float, "integer": int, "boolean": bool}

def _make_mcp_tool(mcp_tool_meta):
    """Wrap a single MCP tool as a LangChain StructuredTool."""
    tool_name = mcp_tool_meta.name

    # Build a Pydantic model from the MCP tool's input schema
    schema_props = mcp_tool_meta.inputSchema.get("properties", {})
    field_definitions = {}
    for prop_name, prop_info in schema_props.items():
        prop_type = JSON_TO_PYTHON.get(prop_info.get("type", "string"), str)
        field_definitions[prop_name] = (
            prop_type,
            Field(description=prop_info.get("description", "")),
        )
    args_model = create_model(f"{tool_name}_args", **field_definitions)

    # Every call to the LangChain tool is forwarded to the MCP server
    def _invoke(**kwargs):
        result = mcp_client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in result.content])

    return StructuredTool.from_function(
        func=_invoke,
        name=tool_name,
        description=mcp_tool_meta.description or "",
        args_schema=args_model,
    )

mcp_langchain_tools = [_make_mcp_tool(t) for t in tools_discovered]

print(f"Loaded {len(mcp_langchain_tools)} MCP tool(s) as LangChain tools:")
for t in mcp_langchain_tools:
    print(f"  - {t.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Enable MLflow Tracing and Run the MCP-Powered Agent

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E1. Enable Autologging and UC Registration
# MAGIC
# MAGIC MLflow autologging for LangChain automatically captures execution traces for every agent run. Each trace records the full sequence of LLM calls, tool invocations, inputs, outputs, and latencies without requiring any code changes to the agent itself.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC When using LangChain, use <code>mlflow.langchain.autolog()</code>. Autologging captures: LLM call spans (model, prompt, completion tokens), tool call spans (function name, arguments, return value), and the overall agent run span with total latency.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.langchain.autolog()

uc = UnityCatalog(
    catalog_name = catalog_name,
    schema_name = schema_name,
    table_prefix = "lc_sa" # oai_sa: "OpenAI Single Agent"
)

experiment_path = f"/Users/{username}/langchain-sdk-training"

mlflow.set_experiment(
    experiment_name=experiment_path,
    trace_location=uc,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Build the MCP-Powered Agent
# MAGIC
# MAGIC We rebuild the agent using the same LLM and system prompt, but now the tools come from MCP rather than inline `@tool` definitions. The `create_agent` call is identical. 

# COMMAND ----------

mcp_agent_executor = create_agent(
    llm, 
    tools=mcp_langchain_tools, 
    system_prompt=system_prompt
    )

print("Agent created with MCP-sourced tools.")
print(f"  Tools: {[t.name for t in mcp_langchain_tools]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### E3. Test the MCP-Powered Agent
# MAGIC
# MAGIC With autologging enabled, each agent run produces a full execution trace. The trace captures the LLM reasoning step, the tool call to `classify_price_tier_uc` (now routed through MCP), and the final response generation.

# COMMAND ----------

traced_query = "I found a listing for $225/night in the Mission district. Is that a fair price?"
result = mcp_agent_executor.invoke(
    {
        "messages": [("human", traced_query)]
        }
    )
print(f"User:     {traced_query}")
print(f"Response: {result['messages'][-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### E4. Run Additional Queries

# COMMAND ----------

result = mcp_agent_executor.invoke({"messages": [("human", query_1)]})
print(f"User:     {query_1}")
print(f"Response: {result['messages'][-1].content}")

# COMMAND ----------

result = mcp_agent_executor.invoke({"messages": [("human", query_3)]})
print(f"User:     {query_3}")
print(f"Response: {result['messages'][-1].content}")

# COMMAND ----------

# DBTITLE 1,E5. View MLflow Traces
# MAGIC %md
# MAGIC ### E5. View MLflow Traces
# MAGIC Finally, now that we have some traces logged with MLflow, we can view the traces in either the **Experiments** UI or directly from the unified trace table stored in the schema `agent_training` with the table prefix `lc_sa`. 
# MAGIC
# MAGIC #### Experiment UI
# MAGIC 1. Navigate to **Experiments** in the left sidebar, right click, and select **Open Link in New Tab**. 
# MAGIC 2. Search for the experiment **langchain-sdk-training**. There you will see traces under the **Traces** tab. Click any trace to see a detailed breakdown of tools used and model calls via **ChatDatabricks**.
# MAGIC #### MLflow Unified Tables from Unity Catalog
# MAGIC 1. Navigate to the **Catalog Explorer** and search for **lc_sa**. This surfaces 6 default tables that have been created. 
# MAGIC     - 4 tables: **lc_sa_otel_logs**, **lc_sa_otel_spans**, **lc_sa_otel_annotations**, and **lc_sa_otel_annotations**
# MAGIC     - 2 views: **lc_sa_trace_unified**, and **lc_sa_trace_metadata**
# MAGIC 2. Click on the **_otel_spans** table and click on **Sample Data** and ask a question in the provided query box like _Which trace took the longest to run?_ This will automatically generate a SQL query (you might be asked to grant auto-run, click accept). The answer will be returned to you.
# MAGIC 3. Run the next cell to see logged traces here in the notebook. Notice the same information is captured in the column `spans`.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM lc_sa_trace_unified

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC In this demo, you:
# MAGIC
# MAGIC 1. Defined a `classify_price_tier` tool using LangChain's `@tool` decorator to classify Airbnb prices into market tiers
# MAGIC 2. Built a single Data Analyst agent with LangChain's `create_agent` and tested it with progressively complex pricing queries
# MAGIC 3. Registered `classify_price_tier` as a Python UC function for governance
# MAGIC 4. Discovered the UC function through a Managed MCP server and wrapped it as a LangChain `StructuredTool`
# MAGIC 5. Rebuilt the agent with MCP-sourced tools and enabled MLflow tracing with `mlflow.langchain.autolog()`
# MAGIC
# MAGIC The same `@tool` to UC to MCP progression works across frameworks -- whether you use LangChain (this demo) or the OpenAI Agents SDK (notebook 05).

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>