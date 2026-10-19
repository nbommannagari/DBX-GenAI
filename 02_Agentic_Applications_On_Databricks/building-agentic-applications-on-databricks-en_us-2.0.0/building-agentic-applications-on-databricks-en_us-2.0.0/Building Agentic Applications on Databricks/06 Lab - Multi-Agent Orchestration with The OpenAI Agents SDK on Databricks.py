# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Lab Overview
# MAGIC %md-sandbox
# MAGIC # Lab - Multi-Agent Orchestration with The OpenAI Agents SDK on Databricks
# MAGIC #### Duration: 35 minutes
# MAGIC ## Overview
# MAGIC In this lab, you will apply what you learned in the *Introduction to OpenAI Agents SDK on Databricks* demo to build a multi-agent system yourself. You will configure the SDK, connect two worker agents to Unity Catalog functions through Managed MCP servers, create a supervisor that routes requests with handoffs, refine the supervisor's routing instructions in its YAML config, and trace the full orchestration with MLflow.
# MAGIC
# MAGIC Each step has a **TODO** with `<FILL_IN>` placeholders. The instructions above each TODO tell you what to fill in, and you can reveal the answer if you get stuck.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC
# MAGIC 1. **Configure the OpenAI Agents SDK** to route requests through Databricks model services
# MAGIC 2. **Connect worker agents to Unity Catalog functions** using Managed MCP servers, with `DatabricksMCPClient` for discovery and `McpServer` for agent integration
# MAGIC 3. **Build a supervisor agent** that delegates user requests to worker agents with `handoffs`
# MAGIC 4. **Validate and refine routing** by testing queries and editing the supervisor's YAML instructions
# MAGIC 5. **Capture multi-agent execution** with MLflow autologging
# MAGIC 6. **Combine spans into a single trace** with `@mlflow.trace` and `mlflow.start_span`

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

# MAGIC %run ./Includes/Classroom-Setup-6

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. Lab Scenario
# MAGIC
# MAGIC <div class="scnB"><style>.scnB{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#0b2026;width:100%;max-width:100%;box-sizing:border-box;margin:0;padding:8px}.scnB .scen{background:#FFFFFF;color:#0b2026;padding:26px 30px;border-radius:12px;box-sizing:border-box;overflow:hidden;box-shadow:0 2px 16px rgba(27,49,57,0.10),0 1px 4px rgba(27,49,57,0.06);border:1px solid rgba(27,49,57,0.08)}.scnB .kicker{font-size:14px;text-transform:uppercase;letter-spacing:1.2px;color:#5A6F77;margin-bottom:10px;font-weight:700;display:flex;align-items:center;gap:10px}.scnB .body{font-size:15px;line-height:1.7;color:#0b2026}.scnB .body p{margin:0 0 14px}.scnB .body p:last-child{margin:0}.scnB .tabs{display:flex;gap:6px;margin-bottom:0}.scnB .tbtn{padding:9px 18px;border-radius:8px 8px 0 0;border:1px solid #DCE0E2;border-bottom:none;cursor:pointer;font-size:14px;font-weight:700;background:#EEEDE9;color:#5A6F77;font-family:inherit}.scnB .tbtn:hover{background:#DCE0E2;color:#1B3139}.scnB .tbtn-on{background:#2272B4;color:#FFFFFF;border-color:#2272B4}.scnB .tbtn-on:hover{background:#2272B4;color:#FFFFFF}.scnB .tstack{position:relative;display:grid}.scnB .tpane{grid-area:1 / 1 / 2 / 2;background:rgba(66,153,224,0.12);border:1px solid rgba(34,114,180,0.25);border-top:3px solid #2272B4;border-radius:0 8px 8px 8px;padding:18px 22px;font-size:15px;line-height:1.7;color:#0b2026;box-sizing:border-box;visibility:hidden;opacity:0;pointer-events:none}.scnB .tpane-on{visibility:visible;opacity:1;pointer-events:auto}.scnB code{background:rgba(27,49,57,0.07);padding:2px 6px;border-radius:4px;font-size:14px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#0b2026}.scnB ul{margin:8px 0 0 0;padding-left:20px}.scnB li{margin-bottom:10px}.scnB .eq{background:rgba(0,169,114,.10);border-left:4px solid #00A972;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0 0;font-size:14px;font-style:italic;line-height:1.6;color:#0b2026}.scnB .eq-q{background:#FFF6F4;border-left:4px solid #FF5F46;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0 0;font-size:14px;font-style:italic;line-height:1.6;color:#0b2026}</style><div class="scen">
# MAGIC <div class="kicker"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#5A6F77" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect x="4" y="8" width="16" height="12" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M9 13v2"/><path d="M15 13v2"/></svg>Scenario</div>
# MAGIC <div class="body">
# MAGIC <p>We have been tasked with building a multi-agent POC for the data team. The supervisor agent routes requests to two specialized sub-agents: a data analyst and a business analyst. Each agent is equipped with a UC function accessed through MCP. Select a tab below to explore each agent's role.</p>
# MAGIC </div>
# MAGIC <div class="tabs">
# MAGIC <button onclick="showTabB('supervisor')" id="btnB-supervisor" class="tbtn tbtn-on">Supervisor</button>
# MAGIC <button onclick="showTabB('analyst')" id="btnB-analyst" class="tbtn">Data Analyst</button>
# MAGIC <button onclick="showTabB('business')" id="btnB-business" class="tbtn">Business Analyst</button>
# MAGIC </div>
# MAGIC <div class="tstack">
# MAGIC <div id="tpB-supervisor" class="tpane tpane-on">
# MAGIC <strong>Role:</strong> The supervisor agent is the entry point for all incoming requests. It does not hold any tools itself &#x2014; instead, it reads the intent of each request and routes it to the appropriate sub-agent.<br><br>
# MAGIC <strong>Routing logic:</strong>
# MAGIC <ul>
# MAGIC <li>Questions about <em>pricing tiers, deal evaluation, or price classification</em> &#x2192; routed to the <strong>Data Analyst</strong></li>
# MAGIC <li>Questions about <em>neighborhood summaries, average prices, listing counts, or room type mix</em> &#x2192; routed to the <strong>Business Analyst</strong></li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC <div id="tpB-analyst" class="tpane">
# MAGIC <strong>Role:</strong> The data analyst agent classifies Airbnb listing prices into market tiers based on SF pricing distribution. It helps users understand whether a price point is Budget, Mid-Range, Premium, or Luxury.<br><br>
# MAGIC <strong>Tool:</strong> <code>classify_price_tier</code> (via MCP)<br><br>
# MAGIC <strong>Returns:</strong> A JSON object containing the price, tier, percentile band, and a short interpretation.<br><br>
# MAGIC <div class="eq-q">"Is $200/night a good deal for an Airbnb in SF?"</div>
# MAGIC </div>
# MAGIC <div id="tpB-business" class="tpane">
# MAGIC <strong>Role:</strong> The business analyst agent surfaces market insights for stakeholders. It aggregates data to answer strategic questions about neighborhoods, pricing, and listing mix.<br><br>
# MAGIC <strong>Tool:</strong> <code>neighborhood_summary</code> (via MCP)<br><br>
# MAGIC <strong>Returns:</strong> A table of per-neighborhood metrics including average price, total listings, and most common room type.<br><br>
# MAGIC <div class="eq-q">"Which neighborhoods have the highest average price?"</div>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div></div>
# MAGIC <script>
# MAGIC function showTabB(name) {
# MAGIC   ['supervisor','analyst','business'].forEach(function(t) {
# MAGIC     var pane = document.getElementById('tpB-' + t);
# MAGIC     var btn = document.getElementById('btnB-' + t);
# MAGIC     if (t === name) {
# MAGIC       pane.className = 'tpane tpane-on';
# MAGIC       btn.className = 'tbtn tbtn-on';
# MAGIC     } else {
# MAGIC       pane.className = 'tpane';
# MAGIC       btn.className = 'tbtn';
# MAGIC     }
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Import Libraries and Configure the SDK
# MAGIC
# MAGIC **TODO:** Configure the OpenAI Agents SDK the same way you did in notebook 05. Fill in the three `<FILL_IN>` placeholders to:
# MAGIC 1. Set the default client to `AsyncDatabricksOpenAI`, routed through **Unity Gateway model services** (`use_ai_gateway=True`).
# MAGIC 1. Set the default API to `"chat_completions"`.
# MAGIC 1. Disable the OpenAI trace processor by passing an empty list, so MLflow handles tracing instead.
# MAGIC
# MAGIC The imports, `WorkspaceClient`, and `nest_asyncio` setup are provided. As part of the classroom setup, a UC MLflow experiment has been created for you:
# MAGIC - UC table prefixes: `oai_ma_sdk`
# MAGIC - MLflow experiment: `oai-ma-sdk-training`
# MAGIC
# MAGIC Additional documentation and reading:
# MAGIC - [`set_default_openai_client`](https://openai.github.io/openai-agents-python/ref/#agents.set_default_openai_client)
# MAGIC - [`set_default_openai_api`](https://openai.github.io/openai-agents-python/ref/#agents.set_default_openai_api)
# MAGIC - [`set_trace_processors`](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md)

# COMMAND ----------

from agents import Agent, Runner, handoff, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from databricks.sdk import WorkspaceClient

import asyncio
import nest_asyncio
import logging

nest_asyncio.apply()

## Route the SDK through Databricks model services (recall notebook 05, section A1)
set_default_openai_client(<FILL_IN>)
set_default_openai_api(<FILL_IN>)
set_trace_processors(<FILL_IN>)  # Disable the OpenAI trace processor; MLflow handles tracing
logging.getLogger("openai").setLevel(logging.WARNING)

ws = WorkspaceClient()
workspace_host = ws.config.host

print("OpenAI Agents SDK configured to use Databricks model routing.")
print(f"Workspace host: {workspace_host}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from agents import Agent, Runner, handoff, set_default_openai_api, set_default_openai_client
# MAGIC from agents.tracing import set_trace_processors
# MAGIC from databricks_openai import AsyncDatabricksOpenAI
# MAGIC from databricks_openai.agents import McpServer
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC import asyncio
# MAGIC import nest_asyncio
# MAGIC import logging
# MAGIC
# MAGIC nest_asyncio.apply()
# MAGIC
# MAGIC set_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))
# MAGIC set_default_openai_api("chat_completions")
# MAGIC set_trace_processors([])  # Disable the OpenAI trace processor; MLflow handles tracing
# MAGIC logging.getLogger("openai").setLevel(logging.WARNING)
# MAGIC
# MAGIC ws = WorkspaceClient()
# MAGIC workspace_host = ws.config.host
# MAGIC
# MAGIC print("OpenAI Agents SDK configured to use Databricks model routing.")
# MAGIC print(f"Workspace host: {workspace_host}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Connect to UC Tools via MCP

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Discover Available UC Functions
# MAGIC
# MAGIC Before building agents, verify that the UC functions are registered and discoverable. This lab uses two ways to work with the Databricks-managed MCP server:
# MAGIC 1. `DatabricksMCPClient`: a synchronous API for direct MCP tool calls (good for notebooks)
# MAGIC 1. `McpServer`: an asynchronous context manager that connects MCP servers to OpenAI Agents SDK agents (useful for applications)
# MAGIC
# MAGIC **TODO:** Build the MCP server URL using the pattern above, `{workspace_host}/api/2.0/mcp/functions/{catalog}/{schema}`, then list the tools it exposes. You should see both `classify_price_tier` and `neighborhood_summary`.

# COMMAND ----------

from databricks_mcp import DatabricksMCPClient

## Point to the Managed MCP server for the UC functions in your catalog and schema
mcp_url = <FILL_IN>
mcp_client = DatabricksMCPClient(server_url=mcp_url, workspace_client=ws)

## Discover the tools the MCP server exposes
tools = mcp_client.<FILL_IN>
print(f"MCP server URL: {mcp_url}")
print(f"Discovered {len(tools)} UC function tool(s):")
for t in tools:
    print(f"  - {t.name}: {t.description}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from databricks_mcp import DatabricksMCPClient
# MAGIC
# MAGIC mcp_url = f"{workspace_host}/api/2.0/mcp/functions/{catalog_name}/{schema_name}"
# MAGIC mcp_client = DatabricksMCPClient(server_url=mcp_url, workspace_client=ws)
# MAGIC
# MAGIC tools = mcp_client.list_tools()
# MAGIC print(f"MCP server URL: {mcp_url}")
# MAGIC print(f"Discovered {len(tools)} UC function tool(s):")
# MAGIC for t in tools:
# MAGIC     print(f"  - {t.name}: {t.description}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Build MCP-Backed Agents
# MAGIC
# MAGIC Each agent will have its own configuration file. Additionally, each agent will have its own `McpServer` scoped to a single UC function. This ensures each agent can only access the tool it needs:
# MAGIC 1. `classify_price_tier` → **Data Analyst agent**
# MAGIC 1. `neighborhood_summary` → **Business Analyst agent**

# COMMAND ----------

# MAGIC %md
# MAGIC #### B2.1 Load the Agent Configuration
# MAGIC We will load the agent configuration file. This is a recommended practice since any updates to the agents' system prompts, name, etc. live outside this notebook. These variables will be stored in the dictionary `agents`:
# MAGIC - `agents["X"]['name']`: The agent's name
# MAGIC - `agents["X"]['llm_endpoint']`: The name of the Unity Catalog **model service** the agent calls (`agent-training-model`). In the next step you combine it with your catalog and schema into the service's fully qualified name.
# MAGIC - `agents["X"]['system_prompt']`: The agent's system prompt
# MAGIC
# MAGIC Running the following cell will store config variables for the data analyst, business data analyst, and the supervisor agent that will be used later.
# MAGIC     - Navigate to the folder **agent_config** and inspect the agent contents for each: **business_data_analyst_agent.yaml**, **data_analyst_agent.yaml**, and **supervisor_agent.yaml**

# COMMAND ----------

import yaml
from pathlib import Path

agent_files = {
    "da": "data_analyst_agent",
    "ba": "business_data_analyst_agent",
    "sa": "supervisor_agent",
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

# MAGIC %md
# MAGIC #### B2.2 Build the Model Service Name
# MAGIC
# MAGIC As part of the classroom setup, a Unity Catalog **model service** named `agent-training-model` was created in your catalog and schema. It routes requests through Unity Gateway to `databricks-gpt-5-2`. Agents reference the service, not the model, by its fully qualified name, `catalog.schema.service`, just as in notebook 05. The YAML configs store only the service name in `llm_endpoint`.
# MAGIC
# MAGIC **TODO:** Build `model_fqn` from your catalog, your schema, and the `llm_endpoint` value. All three agents use the same service, so you only need to build it once.

# COMMAND ----------

## Fully qualified model service name: catalog.schema.service
model_fqn = f"<FILL_IN>.<FILL_IN>.{agents['da']['llm_endpoint']}"
print(f"Model service: {model_fqn}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC model_fqn = f"{catalog_name}.{schema_name}.{agents['da']['llm_endpoint']}"
# MAGIC print(f"Model service: {model_fqn}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC #### B2.3 Data Analyst Agent
# MAGIC
# MAGIC **TODO:** Create an `McpServer` scoped to the `classify_price_tier` UC function, then create the Data Analyst agent from its YAML config.
# MAGIC - `McpServer.from_uc_function()` takes your catalog, your schema, and the function name.
# MAGIC - Pass the MCP server to the agent with `mcp_servers=` (a list) instead of `tools=`. Scoping each agent to its own MCP server means it can only see the tool it needs.
# MAGIC - Use `agents["da"]["system_prompt"]` for the instructions and `model_fqn` for the model.

# COMMAND ----------

## MCP server scoped to a single UC function
da_mcp_server = McpServer.<FILL_IN>(
    catalog=<FILL_IN>,
    schema=<FILL_IN>,
    function_name=<FILL_IN>,
    workspace_client=ws,
    timeout=60.0,
)

## Data Analyst agent, backed by the MCP server above
data_analyst_agent = Agent(
    name=agents["da"]["name"],
    instructions=<FILL_IN>,
    model=<FILL_IN>,
    mcp_servers=<FILL_IN>,
)
print(f"Created agent: {data_analyst_agent.name}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC da_mcp_server = McpServer.from_uc_function(
# MAGIC     catalog=catalog_name,
# MAGIC     schema=schema_name,
# MAGIC     function_name="classify_price_tier",
# MAGIC     workspace_client=ws,
# MAGIC     timeout=60.0,
# MAGIC )
# MAGIC
# MAGIC data_analyst_agent = Agent(
# MAGIC     name=agents["da"]["name"],
# MAGIC     instructions=agents["da"]["system_prompt"],
# MAGIC     model=model_fqn,
# MAGIC     mcp_servers=[da_mcp_server],
# MAGIC )
# MAGIC print(f"Created agent: {data_analyst_agent.name}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC #### B2.4 Business Analyst Agent
# MAGIC
# MAGIC **TODO:** Repeat the same pattern for the Business Analyst, this time writing the full calls yourself:
# MAGIC - An `McpServer` scoped to the `neighborhood_summary` UC function, stored in `ba_mcp_server`
# MAGIC - An `Agent` built from the `agents["ba"]` config, stored in `business_analyst_agent`

# COMMAND ----------

## MCP server scoped to the neighborhood_summary UC function
ba_mcp_server = <FILL_IN>

## Business Analyst agent, backed by ba_mcp_server
business_analyst_agent = <FILL_IN>

print(f"Created agent: {business_analyst_agent.name}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ba_mcp_server = McpServer.from_uc_function(
# MAGIC     catalog=catalog_name,
# MAGIC     schema=schema_name,
# MAGIC     function_name="neighborhood_summary",
# MAGIC     workspace_client=ws,
# MAGIC     timeout=60.0,
# MAGIC )
# MAGIC
# MAGIC business_analyst_agent = Agent(
# MAGIC     name=agents["ba"]["name"],
# MAGIC     instructions=agents["ba"]["system_prompt"],
# MAGIC     model=model_fqn,
# MAGIC     mcp_servers=[ba_mcp_server],
# MAGIC )
# MAGIC
# MAGIC print(f"Created agent: {business_analyst_agent.name}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Create a Supervisor Agent with Handoffs

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. Build the Supervisor
# MAGIC
# MAGIC The supervisor agent analyzes each user request and delegates to the appropriate worker using the **handoff** mechanism from the [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents). When the supervisor hands off to a worker, that worker takes over the conversation and responds directly to the user.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC The <code>handoffs</code> parameter tells the supervisor which agents it can delegate to. The supervisor's instructions guide its routing decisions. The LLM reads each worker's name and instructions to decide which one is best suited for the request.
# MAGIC See: <a href="https://developers.openai.com/api/docs/guides/agents/orchestration" style="color: #1976d2; text-decoration: underline;">Orchestration and handoffs</a>
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC **TODO:** Create the supervisor and give it both worker agents as handoffs. The supervisor has no tools of its own: its only job is to pick a worker.

# COMMAND ----------

supervisor = Agent(
    name=agents["sa"]["name"],
    instructions=agents["sa"]["system_prompt"],
    model=model_fqn,
    <FILL_IN>=[<FILL_IN>, <FILL_IN>],
)

print(f"Created supervisor with handoffs to: {[a.name for a in supervisor.handoffs]}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC supervisor = Agent(
# MAGIC     name=agents["sa"]["name"],
# MAGIC     instructions=agents["sa"]["system_prompt"],
# MAGIC     model=model_fqn,
# MAGIC     handoffs=[data_analyst_agent, business_analyst_agent],
# MAGIC )
# MAGIC
# MAGIC print(f"Created supervisor with handoffs to: {[a.name for a in supervisor.handoffs]}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# DBTITLE 1,C2. How the Supervisor Routes Requests
# MAGIC %md-sandbox
# MAGIC ### C2. How the Supervisor Routes Requests
# MAGIC
# MAGIC The supervisor sees each worker as a **handoff tool**. When a user sends a request, the supervisor's LLM reads the request, examines the available handoffs, and decides which worker should handle it.
# MAGIC
# MAGIC **Click on any component in the diagram** to learn more about it.
# MAGIC
# MAGIC <style>
# MAGIC .ma-node {
# MAGIC   border-radius: 14px; padding: 12px 18px; text-align: center; font-weight: 600; font-size: 14px;
# MAGIC   cursor: pointer; user-select: none; transition: transform 0.12s, box-shadow 0.15s;
# MAGIC   box-shadow: 0 2px 8px rgba(27,49,57,.07);
# MAGIC }
# MAGIC .ma-node:hover { transform: translateY(-2px); }
# MAGIC .ma-node.active { box-shadow: 0 4px 14px rgba(27,49,57,.14); transform: translateY(-2px); }
# MAGIC .ma-arrow { color: #999; font-size: 22px; padding: 0 8px; user-select: none; }
# MAGIC .ma-detail-wrap { overflow: hidden; max-height: 0; opacity: 0; transition: max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease; margin-top: 0; }
# MAGIC .ma-detail-wrap.open { max-height: 1200px; opacity: 1; margin-top: 16px; }
# MAGIC .ma-detail-card { background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07); font-size: 14pt; line-height: 1.7; color: #333; }
# MAGIC .ma-detail-card strong { color: #0b2026; }
# MAGIC .ma-detail-card code { background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-size: 13pt; }
# MAGIC .ma-example { background: #F9F7F4; color: #1B3139; border: 1px solid #DCE0E2; border-radius: 8px; padding: 14px 18px; margin-top: 12px; font-family: 'SF Mono','Cascadia Code','Fira Code',Consolas,monospace; font-size: 0.82rem; line-height: 1.6; }
# MAGIC .ma-example .q { color: #2563EB; } .ma-example .a { color: #059669; } .ma-example .r { color: #D97706; }
# MAGIC </style>
# MAGIC
# MAGIC <div style="max-width: 1100px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;">
# MAGIC   <div class="ma-node" style="background:transparent;color:#0b2026;border:2px solid #0b2026;" data-ma="input" onclick="maSelect('input')">
# MAGIC     User Query
# MAGIC   </div>
# MAGIC   <div class="ma-arrow">&#x2192;</div>
# MAGIC   <div class="ma-node" id="ma-supervisor" style="background:#FF5F46;color:#fff;" data-ma="supervisor" onclick="maSelect('supervisor')">
# MAGIC     Supervisor<br/><span style="font-weight:400;font-size:12px;">Reads instructions + handoffs</span>
# MAGIC   </div>
# MAGIC   <div class="ma-arrow">&#x2192;</div>
# MAGIC   <div style="display:flex;flex-direction:column;gap:6px;">
# MAGIC     <div class="ma-node" style="background:#2574B5;color:#fff;" data-ma="data" onclick="maSelect('data')">
# MAGIC       Data Analyst<br/><span style="font-weight:400;font-size:12px;">classify_price_tier via MCP</span>
# MAGIC     </div>
# MAGIC     <div class="ma-node" style="background:#02A36F;color:#fff;" data-ma="business" onclick="maSelect('business')">
# MAGIC       Business Analyst<br/><span style="font-weight:400;font-size:12px;">neighborhood_summary via MCP</span>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <div class="ma-arrow">&#x2192;</div>
# MAGIC   <div class="ma-node" style="background:transparent;color:#0b2026;border:2px solid #0b2026;" data-ma="output" onclick="maSelect('output')">
# MAGIC     Response
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="ma-detail-wrap" id="ma-detail-wrap">
# MAGIC   <div class="ma-detail-card" id="ma-detail-card"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var MA_DATA = {
# MAGIC   input: {
# MAGIC     color: '#1B5162',
# MAGIC     title: 'User Query (Input)',
# MAGIC     html: '<strong>What happens here:</strong> The user sends a natural language question. The system does not pre-classify the query. The raw text is passed directly to the supervisor agent.<br/><br/><strong>Examples:</strong><div class="ma-example"><span class="q">"Is $200/night a good deal for an Airbnb in SF?"</span> &#x2192; routes to Data Analyst<br/><span class="q">"Which neighborhoods have the highest average price?"</span> &#x2192; routes to Business Analyst</div>'
# MAGIC   },
# MAGIC   supervisor: {
# MAGIC     color: '#FF5F46',
# MAGIC     title: 'Supervisor Agent',
# MAGIC     html: '<strong>Role:</strong> Analyzes every incoming request and decides which worker should handle it.<br/><br/><strong>How it decides:</strong> The supervisor has <code>handoffs=[data_analyst_agent, business_analyst_agent]</code>. The LLM sees each handoff as a callable tool named <code>transfer_to_Data_Analyst</code> and <code>transfer_to_Business_Analyst</code>. Based on the user query and the worker descriptions in its instructions, it picks the best match.<br/><br/><strong>Example routing:</strong><div class="ma-example"><span class="q">User: "Is $200/night a good deal in SF?"</span><br/><span class="r">Supervisor thinks: This is about price classification...</span><br/><span class="a">Action: transfer_to_Data_Analyst</span></div><div class="ma-example" style="margin-top:8px;"><span class="q">User: "Which neighborhoods have the highest average price?"</span><br/><span class="r">Supervisor thinks: This is a market/business question about neighborhoods...</span><br/><span class="a">Action: transfer_to_Business_Analyst</span></div>'
# MAGIC   },
# MAGIC   data: {
# MAGIC     color: '#2574B5',
# MAGIC     title: 'Data Analyst Agent',
# MAGIC     html: '<strong>Role:</strong> Classifies Airbnb listing prices into market tiers based on SF pricing distribution.<br/><br/><strong>Tool:</strong> <code>classify_price_tier(price: float)</code> accessed via MCP from Unity Catalog. Returns a JSON object with tier (Budget, Mid-Range, Premium, Luxury), percentile band, and interpretation.<br/><br/><strong>Example interaction:</strong><div class="ma-example"><span class="q">User: "Is $200/night a good deal in SF?"</span><br/><span class="r">Agent calls: classify_price_tier(200.0)</span><br/><span class="a">Agent: "$200/night falls in the Premium tier (50-75th percentile). This is above average for SF listings."</span></div>'
# MAGIC   },
# MAGIC   business: {
# MAGIC     color: '#02A36F',
# MAGIC     title: 'Business Analyst Agent',
# MAGIC     html: '<strong>Role:</strong> Surfaces market insights for stakeholders by analyzing neighborhood-level pricing and listing mix data.<br/><br/><strong>Tool:</strong> <code>neighborhood_summary(neighborhood: str)</code> accessed via MCP from Unity Catalog. Returns average price, total listings, and most common room type per neighborhood.<br/><br/><strong>Example interaction:</strong><div class="ma-example"><span class="q">User: "Which neighborhoods have the highest average price?"</span><br/><span class="r">Agent calls: neighborhood_summary()</span><br/><span class="a">Agent: "The top neighborhoods by average price are: Sea Cliff ($284.50, 89% entire homes), Pacific Heights ($263.15, 72% entire homes), and Nob Hill ($241.80, 65% entire homes)."</span></div>'
# MAGIC   },
# MAGIC   output: {
# MAGIC     color: '#1B5162',
# MAGIC     title: 'Response (Output)',
# MAGIC     html: '<strong>What happens here:</strong> The worker agent that received the handoff produces the final response. The response goes directly back to the user.<br/><br/><strong>Key detail:</strong> The supervisor does NOT post-process or modify the worker output. Once a handoff occurs, the worker owns the conversation. This is why <code>result.last_agent.name</code> shows the worker, not the supervisor.<br/><br/><strong>In the trace:</strong> The MLflow trace captures the full flow: supervisor LLM call &#x2192; handoff decision &#x2192; worker LLM call &#x2192; tool calls (if any) &#x2192; final output.'
# MAGIC   }
# MAGIC };
# MAGIC var maCurrent = null;
# MAGIC function maSelect(id) {
# MAGIC   var wrap = document.getElementById('ma-detail-wrap');
# MAGIC   var card = document.getElementById('ma-detail-card');
# MAGIC   document.querySelectorAll('.ma-node').forEach(function(n) { n.classList.toggle('active', n.dataset.ma === id); });
# MAGIC   if (maCurrent === id) { wrap.classList.remove('open'); document.querySelectorAll('.ma-node').forEach(function(n) { n.classList.remove('active'); }); maCurrent = null; return; }
# MAGIC   maCurrent = id;
# MAGIC   var d = MA_DATA[id];
# MAGIC   card.innerHTML = '<div style="height:5px;border-radius:5px;margin:-4px 0 12px;background:' + d.color + ';"></div><div style="font-size:16pt;font-weight:700;margin-bottom:12px;color:' + d.color + ';">' + d.title + '</div><div style="font-size:15pt;color:#0b2026;line-height:1.7;">' + d.html + '</div>';
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Test Multi-Agent Routing

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Data Question
# MAGIC
# MAGIC This question is about price classification. The supervisor should route it to the Data Analyst agent, which will call `classify_price_tier` via MCP. You can verify this in MLflow interface as a part of running the next cell. 
# MAGIC
# MAGIC We are connecting to the Databricks managed MCP server via`McpServer`, which is an async context manager, hence we need to be mindful of using `async` and `await`. Similarly, `Runner.run()` is also async.
# MAGIC
# MAGIC **TODO:** Fill in:
# MAGIC - both MCP servers in the `async with` statement, since they must stay open while the agents run
# MAGIC - the `Runner` method that runs the supervisor (the same one you used in notebook 05)
# MAGIC - the `result` attributes for the agent that produced the final answer and for the final response text

# COMMAND ----------

data_query = "Is $200/night a good deal for an Airbnb in SF?"

async def test_data_query():
    async with <FILL_IN>, <FILL_IN>:
        result = await <FILL_IN>(supervisor, data_query)
        return result

result = asyncio.get_event_loop().run_until_complete(test_data_query())
print(f"User:       {data_query}")
print(f"Handled by: {result.<FILL_IN>.name}")
print(f"Response:   {result.<FILL_IN>}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC data_query = "Is $200/night a good deal for an Airbnb in SF?"
# MAGIC
# MAGIC async def test_data_query():
# MAGIC     async with da_mcp_server, ba_mcp_server:
# MAGIC         result = await Runner.run(supervisor, data_query)
# MAGIC         return result
# MAGIC
# MAGIC result = asyncio.get_event_loop().run_until_complete(test_data_query())
# MAGIC print(f"User:       {data_query}")
# MAGIC print(f"Handled by: {result.last_agent.name}")
# MAGIC print(f"Response:   {result.final_output}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Business Question
# MAGIC
# MAGIC **TODO:** Write your own question that the supervisor should hand off to the Business Analyst, which answers it by calling `neighborhood_summary` via MCP. Look at the Business Analyst's routing rules in `agent_config/supervisor_agent.yaml` for ideas, then check `Handled by` in the output to confirm the handoff.

# COMMAND ----------

## Write a question the supervisor should route to the Business Analyst
business_query = "<FILL_IN>"

async def test_business_query():
    async with da_mcp_server, ba_mcp_server:
        result = await Runner.run(supervisor, business_query)
        return result

result = asyncio.get_event_loop().run_until_complete(test_business_query())
print(f"User:       {business_query}")
print(f"Handled by: {result.last_agent.name}")
print(f"Response:   {result.final_output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Any neighborhood-level question works, for example:
# MAGIC business_query = "What is the average price of the Mission neighborhood?"
# MAGIC
# MAGIC async def test_business_query():
# MAGIC     async with da_mcp_server, ba_mcp_server:
# MAGIC         result = await Runner.run(supervisor, business_query)
# MAGIC         return result
# MAGIC
# MAGIC result = asyncio.get_event_loop().run_until_complete(test_business_query())
# MAGIC print(f"User:       {business_query}")
# MAGIC print(f"Handled by: {result.last_agent.name}")
# MAGIC print(f"Response:   {result.final_output}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC Why are there traces in the outputs above?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC <p style="margin-top: 10px;">Running the cell above will produce a trace showing a tool call took place. This is because <code>@mlflow.trace(span_type=SpanType.TOOL)</code> is used during <code>call_tool</code> (see the <a href="https://github.com/databricks/databricks-ai-bridge/blob/main/integrations/openai/src/databricks_openai/agents/mcp_server.py" style="color: #1976d2; text-decoration: underline;">source code</a>).</p>
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Mixed Routing Test
# MAGIC
# MAGIC Run multiple queries and show how the supervisor routes each one to the appropriate worker agent. Notice that in the output there are multiple traces. In the next section, we will see how to concatenate them into 1 trace and perform proper logging as a series of events so we have a more comprehensive picture of the flow from input to output via the supervisor agent.
# MAGIC
# MAGIC **TODO:** For each query, fill in the agent you expect the supervisor to hand off to, using `agents["da"]["name"]` or `agents["ba"]["name"]`. The last query combines a price with a neighborhood, so think about which tool can actually answer it: `classify_price_tier` only knows citywide price tiers, while `neighborhood_summary` knows each neighborhood's average price.
# MAGIC
# MAGIC Routing is an LLM decision, so a `MISMATCH` can appear on any run. You will tighten the routing rules in the next step.

# COMMAND ----------

import pandas as pd

## Expected agent for each query: agents["da"]["name"] or agents["ba"]["name"]
test_queries = [
    ("Is $150/night a good deal in SF?", <FILL_IN>),
    ("What is the average price in Mission?", <FILL_IN>),
    ("Would $350/night be considered luxury?", <FILL_IN>),
    ("Is the average price in Mission higher or lower than in Noe Valley?", <FILL_IN>),
    ("Is $250/night expensive for the Mission?", <FILL_IN>),
]

async def test_mixed_routing():
    async with da_mcp_server, ba_mcp_server:
        results = []
        for query, expected in test_queries:
            result = await Runner.run(supervisor, query)
            actual = result.last_agent.name
            status = "PASS" if actual == expected else "MISMATCH"
            results.append({
                "Query": query,
                "Expected Agent": expected,
                "Actual Agent": actual,
                "Status": status,
            })
        return results

results = asyncio.get_event_loop().run_until_complete(test_mixed_routing())
display(pd.DataFrame(results))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC
# MAGIC test_queries = [
# MAGIC     ("Is $150/night a good deal in SF?", agents["da"]["name"]),
# MAGIC     ("What is the average price in Mission?", agents["ba"]["name"]),
# MAGIC     ("Would $350/night be considered luxury?", agents["da"]["name"]),
# MAGIC     ("Is the average price in Mission higher or lower than in Noe Valley?", agents["ba"]["name"]),
# MAGIC     ## Needs the Mission's average price, which only neighborhood_summary provides
# MAGIC     ("Is $250/night expensive for the Mission?", agents["ba"]["name"]),
# MAGIC ]
# MAGIC
# MAGIC async def test_mixed_routing():
# MAGIC     async with da_mcp_server, ba_mcp_server:
# MAGIC         results = []
# MAGIC         for query, expected in test_queries:
# MAGIC             result = await Runner.run(supervisor, query)
# MAGIC             actual = result.last_agent.name
# MAGIC             status = "PASS" if actual == expected else "MISMATCH"
# MAGIC             results.append({
# MAGIC                 "Query": query,
# MAGIC                 "Expected Agent": expected,
# MAGIC                 "Actual Agent": actual,
# MAGIC                 "Status": status,
# MAGIC             })
# MAGIC         return results
# MAGIC
# MAGIC results = asyncio.get_event_loop().run_until_complete(test_mixed_routing())
# MAGIC display(pd.DataFrame(results))
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D4. Refine the Supervisor's Routing Instructions
# MAGIC
# MAGIC The supervisor's routing rules come from its instructions in `agent_config/supervisor_agent.yaml`, not from your code. This file is in your copy of the course folder, so you can edit it. The last query in D3 combines a price with a neighborhood, and the current rules don't say which worker owns that kind of question. Fix that in the instructions.
# MAGIC
# MAGIC **TODO:**
# MAGIC 1. Open `agent_config/supervisor_agent.yaml` from the workspace file browser.
# MAGIC 1. Under the `Route to Business Analyst for:` list in `system_prompt`, add a rule for questions that compare a specific price to a specific neighborhood. Keep the same indentation as the existing lines, because `system_prompt` is a YAML block.
# MAGIC 1. Save the file, then run the next cell. It reloads the YAML, rebuilds the supervisor with your new instructions, and re-runs the D3 tests.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```yaml
# MAGIC Route to Business Analyst for:
# MAGIC   - Neighborhood summaries and comparisons
# MAGIC   - Average prices by area
# MAGIC   - Listing counts and room type mix
# MAGIC   - Market trends and insights
# MAGIC   - Comparing a specific price to a specific neighborhood (for example,
# MAGIC     "Is $250/night expensive for the Mission?"), because answering requires
# MAGIC     that neighborhood's average price from neighborhood_summary
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

## Reload the edited supervisor config, rebuild the supervisor, and re-run the D3 routing tests
cfg = yaml.safe_load(Path("./agent_config/supervisor_agent.yaml").read_text())
agents["sa"]["system_prompt"] = cfg["system_prompt"]

supervisor = Agent(
    name=agents["sa"]["name"],
    instructions=agents["sa"]["system_prompt"],
    model=model_fqn,
    handoffs=[data_analyst_agent, business_analyst_agent],
)

results = asyncio.get_event_loop().run_until_complete(test_mixed_routing())
display(pd.DataFrame(results))

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Enable MLflow Tracing
# MAGIC Now that we have built and validated our multi-agent system, we enable autologging with MLflow to capture execution traces. This section will have fractured traces showing each handoff and tool call as a separate trace. We will see later how we can bring these together in 1 trace. For now, let's focus on the individual outputs to see how a multi-agent system appears as multiple traces. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E1. Enable Autologging
# MAGIC
# MAGIC We enable MLflow tracing so the next agent runs produce full execution traces.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC In the MLflow Experiment UI, multi-agent traces show nested spans: the supervisor's LLM call appears at the top level, followed by the handoff event, and then the worker's LLM and tool calls as child spans. This makes it easy to debug routing decisions and tool selection.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC **TODO:** Turn on MLflow autologging for the OpenAI integration, as you did in notebook 05 (section D1).

# COMMAND ----------

import mlflow

mlflow.<FILL_IN>.<FILL_IN>()

print("MLflow OpenAI autologging enabled.")
print(f"Traces will appear in experiment: {mlflow_experiment_path}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC print("MLflow OpenAI autologging enabled.")
# MAGIC print(f"Traces will appear in experiment: {mlflow_experiment_path}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E2. Run with Tracing
# MAGIC
# MAGIC Run the supervisor again with tracing enabled. The trace will capture the full handoff flow from supervisor to worker. The `transfer_to_Data_Analyst` and `transfer_to_Business_Analyst` tools do not exist in your code. The SDK generates them automatically from `handoffs=[data_analyst_agent, business_analyst_agent]` using the pattern `transfer_to_{agent.name}`.

# COMMAND ----------

async def run_with_tracing(query: str):
    async with da_mcp_server, ba_mcp_server:
        result = await Runner.run(supervisor, query)
        return result

result = asyncio.get_event_loop().run_until_complete(run_with_tracing(data_query))
print(f"User:       {data_query}")
print(f"Handled by: {result.last_agent.name}")
print(f"Response:   {result.final_output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## F. Distributed Tracing
# MAGIC
# MAGIC Notice the output from the previous section may produce multiple separate trace outputs. MLflow supports distributed tracing to stitch spans from different calls into a single unified trace.
# MAGIC
# MAGIC The key idea: wrap the entire multi-agent flow in a parent `@mlflow.trace` span. All child spans (from autologging, tool calls, and handoffs) automatically nest under it, producing one clean trace tree instead of scattered fragments.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC For cross-service distributed tracing (e.g., an agent calling another agent over HTTP), MLflow provides <code>get_tracing_context_headers_for_http_request()</code> and <code>set_tracing_context_from_http_request_headers()</code> to propagate trace context via W3C TraceContext headers. Both services must log to the same MLflow tracking server and experiment. On Databricks, the trace destination must be set to Unity Catalog.
# MAGIC See: <a href="https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/distributed-tracing/" style="color: #1976d2; text-decoration: underline;">MLflow Distributed Tracing docs</a>
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### F1. Build Traced Agent Wrappers
# MAGIC
# MAGIC To make handoffs visible in the trace, we wrap each agent's execution in its own `@mlflow.trace` span. The supervisor routes the question, then we call the selected worker inside a labeled span. This produces a trace tree like:
# MAGIC
# MAGIC ```
# MAGIC supervisor_orchestration
# MAGIC   +-- route_decision (supervisor LLM call)
# MAGIC   +-- data_analyst_execution OR business_analyst_execution
# MAGIC       +-- AsyncDatabricksCompletions (worker LLM + tool calls)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Create a trace wrapper for the agents defined above. Here we are setting the `span_type` to `AGENT` and defining the names of the agents as they will appear in the trace. Again, we are connecting to the Databricks managed MCP server via`McpServer`, which is an async context manager, hence we need to be mindful of using `async` and `await` as mentioned in section **D**.
# MAGIC
# MAGIC **TODO:** Fill in:
# MAGIC - the span type for each wrapper (the same span type you used for the agent wrapper in notebook 05, section D4)
# MAGIC - the MCP server each worker needs open while it runs
# MAGIC - the agent each wrapper runs

# COMMAND ----------

## Traced wrapper for the Data Analyst
@mlflow.trace(span_type=<FILL_IN>, name="data_analyst_execution")
async def run_data_analyst(question: str) -> str:
    async with <FILL_IN>:
        result = await Runner.run(<FILL_IN>, question)
        return result.final_output

## Traced wrapper for the Business Analyst
@mlflow.trace(span_type=<FILL_IN>, name="business_analyst_execution")
async def run_business_analyst(question: str) -> str:
    async with <FILL_IN>:
        result = await Runner.run(<FILL_IN>, question)
        return result.final_output

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC @mlflow.trace(span_type="AGENT", name="data_analyst_execution")
# MAGIC async def run_data_analyst(question: str) -> str:
# MAGIC     async with da_mcp_server:
# MAGIC         result = await Runner.run(data_analyst_agent, question)
# MAGIC         return result.final_output
# MAGIC
# MAGIC @mlflow.trace(span_type="AGENT", name="business_analyst_execution")
# MAGIC async def run_business_analyst(question: str) -> str:
# MAGIC     async with ba_mcp_server:
# MAGIC         result = await Runner.run(business_analyst_agent, question)
# MAGIC         return result.final_output
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Define the router and the parent trace. The router is a plain agent with **no handoffs**. It reuses the supervisor's instructions, so instead of handing off it replies with the name of the worker that should handle the request, and we read that reply as the route. The parent `@mlflow.trace` span, `supervisor_orchestration`, is provided.
# MAGIC
# MAGIC **TODO:** Fill in:
# MAGIC - the `mlflow` function that opens a child span named `route_decision` inside the parent trace
# MAGIC - the span methods that record the span's inputs and outputs
# MAGIC - the traced worker wrapper from step 1 that matches each route

# COMMAND ----------

## Router agent (no handoffs, just classifies the request)
router_agent = Agent(
    name="Router",
    instructions=agents["sa"]["system_prompt"],
    model=model_fqn,
)


@mlflow.trace(span_type="AGENT", name="supervisor_orchestration")
async def run_supervisor_traced(question: str) -> str:
    """Route to the correct agent and execute, with each step visible in the trace."""

    # Step 1: Record the routing decision as its own child span
    with mlflow.<FILL_IN>(name="route_decision") as span:
        span.<FILL_IN>({"question": question})
        route_result = await Runner.run(router_agent, question)
        route = route_result.final_output.strip().lower()
        span.<FILL_IN>({"route": route})

    # Step 2: Run the traced worker wrapper that matches the route
    if "business" in route:
        return await <FILL_IN>(question)
    else:
        return await <FILL_IN>(question)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC router_agent = Agent(
# MAGIC     name="Router",
# MAGIC     instructions=agents["sa"]["system_prompt"],
# MAGIC     model=model_fqn,
# MAGIC )
# MAGIC
# MAGIC
# MAGIC @mlflow.trace(span_type="AGENT", name="supervisor_orchestration")
# MAGIC async def run_supervisor_traced(question: str) -> str:
# MAGIC     """Route to the correct agent and execute, with each step visible in the trace."""
# MAGIC
# MAGIC     # Step 1: Record the routing decision as its own child span
# MAGIC     with mlflow.start_span(name="route_decision") as span:
# MAGIC         span.set_inputs({"question": question})
# MAGIC         route_result = await Runner.run(router_agent, question)
# MAGIC         route = route_result.final_output.strip().lower()
# MAGIC         span.set_outputs({"route": route})
# MAGIC
# MAGIC     # Step 2: Run the traced worker wrapper that matches the route
# MAGIC     if "business" in route:
# MAGIC         return await run_business_analyst(question)
# MAGIC     else:
# MAGIC         return await run_data_analyst(question)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### F2. Run and Inspect Unified Traces

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Run the next cell, then open the trace in the MLflow experiment. The parent span, `supervisor_orchestration`, is the caller. The worker span, such as `data_analyst_execution`, is the callee that calls the MCP server. The routing decision appears as its own `route_decision` span between them.

# COMMAND ----------

# Run a data question through the traced wrapper
response = await run_supervisor_traced(data_query)
print(f"Question: {data_query}")
print(f"Response: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC 2. As a final example, we pass a similar query to the business analyst agent. 

# COMMAND ----------

# Run a business question through the traced wrapper
response = await run_supervisor_traced(business_query)
print(f"Question: {business_query}")
print(f"Response: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## G. Conclusion
# MAGIC In this lab, you:
# MAGIC
# MAGIC 1. Configured the OpenAI Agents SDK to route requests through Databricks model services
# MAGIC 2. Discovered UC functions through a Managed MCP server and connected each worker agent to a single function with `McpServer`
# MAGIC 3. Built a supervisor agent that delegates user requests to the Data Analyst and Business Analyst with `handoffs`
# MAGIC 4. Tested routing across data, business, and mixed queries, then refined the routing by editing the supervisor's YAML instructions
# MAGIC 5. Captured multi-agent runs with MLflow autologging
# MAGIC 6. Combined the supervisor and worker spans into a single trace with `@mlflow.trace` and `mlflow.start_span`

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>