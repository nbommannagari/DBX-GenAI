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
# MAGIC # Demo - Agentic Tooling on Databricks
# MAGIC #### Duration: 20 minutes
# MAGIC ## Overview
# MAGIC
# MAGIC This demonstration focuses on how to create Unity Catalog (UC) functions using both SQL and Python for AI agent use cases, expose them via the Model Context Protocol (MCP), and test how a model chooses between them. You will combine concepts from both SQL and Python function creation, demonstrating how to build a comprehensive toolkit that leverages the strengths of each approach within a single agent workflow.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC - Create and register both SQL and Python functions in Unity Catalog for Agent use cases
# MAGIC - Perform initial testing of your UC functions using multiple approaches
# MAGIC - Understand how to equip functions with proper context for AI Agent use cases
# MAGIC - Discover and call UC functions as agent tools via the Model Context Protocol (MCP)
# MAGIC - Test how a model selects and calls your UC functions as tools
# MAGIC - Identify when different tools have been used and understand how the agent utilized each tool type
# MAGIC - Compare the strengths of SQL vs Python tools in agent workflows

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
# MAGIC
# MAGIC Run the next cell to install the libraries for this demo and set up your catalog, schema, and data.

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-2

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Inspect the Airbnb Dataset 
# MAGIC As a part of the classroom setup, the Airbnb dataset has been processed and stored as a Delta table within Unity Catalog. Run the next cell to query the first few rows of the dataset.

# COMMAND ----------

df = spark.read.table('sf_airbnb_listings')
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Initialize the Databricks Function Client
# MAGIC
# MAGIC Initialize the [Databricks Function Client](https://github.com/unitycatalog/unitycatalog/tree/b2d072e56661aedb84cce9be60292b2c54e12224/ai/core#databricks-managed-uc), which is a specialized interface for creating, managing, and running UC functions in Databricks. 
# MAGIC
# MAGIC For building agent tools with the open source UC library, please see [this documentation](https://docs.unitycatalog.io/ai/client/#databricks-function-client). This demonstration will focus on leveraging Databricks-managed UC for building both SQL and Python agent tools.
# MAGIC - Supported [execution modes](https://github.com/unitycatalog/unitycatalog/blob/main/ai/core/src/unitycatalog/ai/core/utils/execution_utils.py#L25) include **Serverless** (default) and **Local**. We will use **serverless** in this notebook. 

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

# COMMAND ----------

# DBTITLE 1,B. Define and Register UC SQL Functions
# MAGIC %md-sandbox
# MAGIC ## B. Define and Register UC SQL Functions
# MAGIC
# MAGIC Before digging into the code, it's important to understand some terminology. 
# MAGIC - Built-in functions like `SUM` and `AVG` are SQL functions, but these are specifically called **system functions**. However, a **SQL function** is any reusable computation that can be called in a SQL statement, even ones defined by users.
# MAGIC - Any function registered via Unity Catalog, regardless of being written in SQL or Python, is considered a **user-defined function (UDF)**.
# MAGIC
# MAGIC In this notebook we will use the term **SQL function** or **function** to mean a function that is or will be registered to UC, hence a SQL UDF. 
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <div>
# MAGIC       <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC         For more information on UDFs in Unity Catalog, please see
# MAGIC         this documentation (<a href="https://docs.databricks.com/aws/en/udf/unity-catalog" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/udf/unity-catalog" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/udf/unity-catalog" style="color: #1976d2; text-decoration: underline;">GCP</a>).
# MAGIC       </p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. SQL Tool: Airbnb Data Analysis
# MAGIC
# MAGIC We start by creating SQL functions that will serve as our agent tools for analyzing the San Francisco Airbnb listings data. These functions include proper documentation that will help the agent understand how to use them. This tool will be created with **SQL only**. 
# MAGIC
# MAGIC #### Recommendations for SQL Functions
# MAGIC The following SQL functions follow recommended practices:
# MAGIC 1. **Clear parameter names and types**: Use descriptive parameter names with appropriate SQL data types
# MAGIC 2. **Comprehensive comments**: Use `COMMENT` clauses for both the function and each parameter to provide clear descriptions
# MAGIC 3. **Deterministic behavior**: Mark functions as `DETERMINISTIC` when they always return the same result for the same inputs
# MAGIC 4. **Proper return type**: Explicitly specify the return data type
# MAGIC 5. **Error handling**: Consider edge cases like NULL values in the function logic

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION avg_neigh_price(
# MAGIC   neighborhood_name STRING COMMENT "The neighborhood name to filter by (e.g., 'Mission', 'Upper Market')"
# MAGIC )
# MAGIC RETURNS DOUBLE
# MAGIC LANGUAGE SQL
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Calculates the average listing price for a specific neighborhood in San Francisco. Returns the average price as a numeric value. Price strings are cleaned and converted to numeric values before averaging.'
# MAGIC RETURN 
# MAGIC SELECT AVG(CAST(REGEXP_REPLACE(price, '[^0-9.]', '') AS DOUBLE))
# MAGIC FROM sf_airbnb_listings
# MAGIC WHERE neighbourhood_cleansed = neighborhood_name
# MAGIC   AND price IS NOT NULL
# MAGIC   AND REGEXP_REPLACE(price, '[^0-9.]', '') != ''

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Test SQL Tool Using SQL Syntax
# MAGIC
# MAGIC Let's verify that our SQL function works correctly by testing it with various inputs directly in SQL. This helps ensure our function behaves as expected before an agent uses it.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test average price function
# MAGIC SELECT avg_neigh_price('Mission') AS mission_avg_price

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Define and Register UC Python Functions
# MAGIC
# MAGIC Now we'll create Python functions that complement our SQL tools by providing capabilities that would be difficult or impossible with SQL alone. Python functions enable advanced data processing, external API integration, and complex business logic. Note we can choose to wrap the Python logic in SQL syntax or use the `DatabricksFunctionClient()`. We will choose to leverage the latter since we already demonstrated how to use SQL to create a function. 

# COMMAND ----------

# DBTITLE 1,C1. Best Practices for Python Agent Tools
# MAGIC %md
# MAGIC    
# MAGIC ### C1. Best Practices for Python Agent Tools
# MAGIC
# MAGIC #### Recommended Practices
# MAGIC
# MAGIC 1. **Explicit type hints**: Provide clear Python type hints for all arguments and the return value. This helps `create_python_function()` infer the function interface and helps agents understand expected inputs and outputs
# MAGIC 2. **Use explicit named parameters**: Prefer clearly named, explicitly typed parameters so the tool schema is easy to understand and use
# MAGIC 3. **Supported data types**: Ensure input and output types are compatible with Databricks and Spark SQL type systems. Refer to the supported types guidance in the Databricks documentation ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/create-custom-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/create-custom-tool) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/create-custom-tool)) and [Spark SQL data types documentation](https://spark.apache.org/docs/latest/sql-ref-datatypes.html)
# MAGIC 4. **Write clear docstrings**: Use a structured docstring format, such as [Google-style formatting](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods), to describe what the function does, each argument, and the return value
# MAGIC     - Make descriptions precise and meaningful so agents can better determine when to use the tool
# MAGIC 5. **Import external libraries inside the function when needed**: For functions that depend on external libraries, importing them inside the function body is the safest pattern and aligns with Databricks examples for Python UC tools and UDF-style execution

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Python Tool: Extract Airbnb Listing Information
# MAGIC
# MAGIC We'll create a Python function that demonstrates capabilities beyond SQL's reach - making HTTP requests to external APIs and parsing HTML content. This function will:
# MAGIC
# MAGIC 1. Fetch HTML content from an Airbnb posting using the listing ID
# MAGIC 2. Extract and parse key information including description, number of reviews, and rating
# MAGIC 3. Return formatted text that can be easily consumed by an AI agent

# COMMAND ----------

def airbnb_posting_info(id: int) -> str:
    """
    Fetches Airbnb posting information as formatted text.

    Args:
        id (int): Airbnb listing ID (e.g., 958)

    Returns:
        str: Formatted listing information (description, reviews, and rating) or error message
    """
    import requests
    import re

    api_url = f"https://www.airbnb.com/rooms/{id}"
    
    try:
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            html = response.text
            
            # Extract description
            desc = re.search(r'"metaDescription":"([^"]+)"', html)
            if desc:
                description = desc.group(1).replace('\\n', ' ')
                parts = description.split(' · ')
                description = ' · '.join(parts[2:]) if len(parts) > 2 else description
            else:
                description = "Description not found"
            
            # Extract review count and rating
            reviews = re.search(r'"reviewCount":(\d+)', html)
            rating = re.search(r'"starRating":([\d.]+)', html)
            
            reviews = reviews.group(1) if reviews else "N/A"
            rating = rating.group(1) if rating else "N/A"
            
            return f"""Description: {description}

Reviews: {reviews}
Rating: {rating} stars"""
        else:
            return f'Request failed with status code: {response.status_code}'
    
    except requests.exceptions.RequestException as e:
        return f'Request error: {str(e)}'

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Test Python Function at Notebook Level
# MAGIC
# MAGIC Before registering the function to Unity Catalog, it's important to test it at the notebook level to ensure it works as expected. This allows you to catch any errors early and validate the output format.

# COMMAND ----------

info = airbnb_posting_info(958)
print(info)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C4. Register Python Tool Using `DatabricksFunctionClient()`
# MAGIC
# MAGIC Now that we've validated our function works correctly, we can register it to Unity Catalog using the `DatabricksFunctionClient`. 
# MAGIC
# MAGIC We use `client.create_python_function()` and pass the following parameters:
# MAGIC
# MAGIC - **`func`**: The Python function object we just created (`airbnb_posting_info`)
# MAGIC - **`catalog`**: The catalog name, which we stored earlier as `catalog_name`
# MAGIC - **`schema`**: The schema name, which we stored earlier as `schema_name`
# MAGIC - **`replace`**: Set to `True` to overwrite the stored Python function if it already exists

# COMMAND ----------

function_info = client.create_python_function(
  func=airbnb_posting_info,
  catalog=catalog_name,
  schema=schema_name,
  replace=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C5. Test Python Tool Using `DatabricksFunctionClient()`
# MAGIC
# MAGIC Let's test the Python-based function using the `execute_function()` API to ensure it works correctly when called through Unity Catalog. Note that you will receive the same response as when we performed the query for the function defined within the current Python interpreter session. 

# COMMAND ----------

result = client.execute_function(
    function_name=f"{catalog_name}.{schema_name}.airbnb_posting_info",
    parameters={"id": 958}
)

print(result.value)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Discover and Test Tools via Model Context Protocol (MCP)
# MAGIC
# MAGIC Now that both SQL and Python tools are registered in Unity Catalog, let's see how an agent would actually **discover** and **call** them. Databricks provides managed MCP servers that expose UC functions as tools following the open [Model Context Protocol](https://modelcontextprotocol.io/) standard.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">MCP is the <strong>recommended way</strong> to connect UC functions to agents. Rather than hard-coding function names, agents discover available tools dynamically through the MCP server. The MCP server acts as the bridge between your UC functions and any agent framework.</p>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">This means you <strong>register once</strong> in Unity Catalog and your tools are available to any agent, any framework, with full governance.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Initialize the MCP Client
# MAGIC
# MAGIC `DatabricksMCPClient` connects to whatever MCP server URL you give it, including Databricks-managed MCP servers like `/api/2.0/mcp/functions/{catalog}/{schema}`. 

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Import `WorkspaceClient` from the Databricks Python SDK and create an authenticated client for the workspace.
# MAGIC 2. Import the MCP client class `DatabricksMCPClient` and build the URL for the managed MCP server that exposes all Unity Catalog functions in the given catalog and schema.
# MAGIC 3. Import `nest_asyncio` and call `nest_asyncio.apply()`, which patches the notebook’s running event loop so you can safely call code that uses asyncio under the hood.

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

ws = WorkspaceClient()
workspace_host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"

mcp_url = f"{workspace_host}/api/2.0/mcp/functions/{catalog_name}/{schema_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Create a `DatabricksMCPClient` instance pointed at the managed MCP server URL. 

# COMMAND ----------

mcp_client = DatabricksMCPClient(
    server_url=mcp_url, 
    workspace_client=ws
    )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D2. Discover Available Tools
# MAGIC
# MAGIC Let's list all the tools the MCP server exposes. Notice that both our SQL function (`avg_neigh_price`) and Python function (`airbnb_posting_info`) are discoverable. An agent would use these descriptions to decide which tool to call.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">When you call <code>list_tools()</code> each method call sends a standard HTTP POST request to the server URL with a JSON-RPC 2.0 payload. </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

tools = mcp_client.list_tools()

for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"  Description: {tool.description}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Call a Tool via MCP
# MAGIC
# MAGIC We can also execute a tool through the MCP protocol, just as an agent would. Let's call `avg_neigh_price` to verify it works end-to-end through MCP. Select the tool by name rather than by its position in the list, because the list order isn't guaranteed.

# COMMAND ----------

# MCP tool names have the form catalog__schema__function
tool_name = f"{catalog_name}__{schema_name}__avg_neigh_price"

result = await mcp_client.acall_tool(tool_name, {"neighborhood_name": "Mission"})

print(f"Result: {result}")
print(f"Answer: ${result.structuredContent['rows'][0][0]}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## E. Test Your Tools with a Model
# MAGIC
# MAGIC An agent's model decides which tool to call, and what arguments to pass, by reading each tool's name, description, and parameters. In this section, you send a question to a model, first without tools and then with both of your tools, and see what it decides to do.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC We won't dive into this code much. The <em>Introduction to OpenAI Agents SDK on Databricks</em> demo, later in the course, covers connecting to a model and building agents in detail. For now, focus on the output.<br/><br/>The code calls a <strong>model service</strong>, <code>agent-training-model</code>, that the classroom setup created in your catalog and schema. A model service is our way of connecting to a model endpoint. We'll cover the details later in the course.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### E1. Connect to a Model Using an SDK
# MAGIC
# MAGIC Run the next cell to connect to the model service and define `ask_model`. It sends a question to the model, offers it the UC functions you name as tools, and prints either the model's answer or the tool calls it wants to make. It doesn't run the tools: running them and passing the results back to the model is the job of an agent.

# COMMAND ----------

import json
from databricks_openai import DatabricksOpenAI

# A model service is our way of connecting to a model endpoint (covered later in the course)
model_service = f"{catalog_name}.{schema_name}.agent-training-model"
llm_client = DatabricksOpenAI(workspace_client=ws, use_ai_gateway=True)

def ask_model(question, tool_names=None):
    """Send a question to the model, optionally offering UC functions as tools.

    Prints the model's answer, or the tool calls it wants to make and the
    arguments it would pass. The tools are not run.
    """
    offered = [t for t in mcp_client.list_tools() if t.name.split("__")[-1] in (tool_names or [])]
    tool_specs = [
        {"type": "function",
         "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
        for t in offered
    ]
    extra = {"tools": tool_specs} if tool_specs else {}
    response = llm_client.chat.completions.create(
        model=model_service,
        messages=[{"role": "user", "content": question}],
        **extra,
    )
    message = response.choices[0].message
    if message.tool_calls:
        print("The model wants to call:")
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            print(f"  - {call.function.name.split('__')[-1]}({args})")
    else:
        print(message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Ask Without Tools
# MAGIC
# MAGIC This question needs both data analysis and external information. With no tools, the model can only answer from what it already knows. Expect a general answer, or a description of the information it would need.

# COMMAND ----------

question = "Compare the average price in Mission with the detailed information for listing 958. Which provides better value?"

ask_model(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ### E3. Ask With Both Tools
# MAGIC
# MAGIC Now offer the model both tools. Notice that it chooses to call `avg_neigh_price` for the neighborhood average and `airbnb_posting_info` for the listing, and that it fills in each argument, such as `neighborhood_name` and `id`, from the question. It knows how because of the names, type hints, and descriptions you gave each function. A model's choices can vary slightly from run to run.

# COMMAND ----------

ask_model(question, tool_names=["avg_neigh_price", "airbnb_posting_info"])

# COMMAND ----------

# DBTITLE 1,F. Managed MCPs
# MAGIC %md
# MAGIC ## F. Managed MCPs
# MAGIC In addition to custom MCPs like the ones shown above, we can connect to out-of-the-box MCPs using the Databricks SDK. To see what OOB MCPs are available: 
# MAGIC 1. Navigate to **Unity Gateway** using the menu to the left
# MAGIC 2. Select **MCPs** to see the list of available MCP servers
# MAGIC
# MAGIC We will test the **Genie One** MCP.
# MAGIC
# MAGIC 3. Select **genie_one_mcp** under **MCPs**
# MAGIC 4. Click on **Overview** and view the **Get started** information. Here you will see different ways to call the MCP services from your agentic applications. 
# MAGIC
# MAGIC All account users have `EXECUTE` permissions on the `system.ai` schema per this notebook's lab configuration. 

# COMMAND ----------

# DBTITLE 1,F1. Genie One MCP
# MAGIC %md
# MAGIC ### F1. Genie One MCP
# MAGIC We'll ask a simple question: explain Lakebase to a new Databricks user. This will be accomplished in a series of steps:
# MAGIC 1. Setup client connection
# MAGIC 2. Ask a question
# MAGIC     - Genie answers are asynchronous, so we will need to handle that with care
# MAGIC 3. Poll until Genie has a complete answer
# MAGIC 4. Get the full query result

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Connect to the client like before, but this time with a different server URL (one that points to **Genie One** instead of our custom tools). Notice in this case, we'll need to get the Workspace ID to construct the correct managed URL for Genie One. 

# COMMAND ----------

workspace_id = ws.get_workspace_id()
full_url = f"{workspace_host}?o={workspace_id}"
genie_url = f"{workspace_host}/ai-gateway/mcp-services/system.ai.genie_one_mcp?o={workspace_id}"
print(f" URL with Workspace ID: {full_url}")

mcp_client_managed = DatabricksMCPClient(
    server_url = genie_url, 
    workspace_client=ws
)

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Define question you wish to ask using the client we just created and its `call_tool` method. 

# COMMAND ----------

ask_result = mcp_client_managed.call_tool(
    "genie_ask",{
        "question": "Explain Lakebase to me, I'm new to Databricks."
        }
    )

# COMMAND ----------

# MAGIC %md
# MAGIC 3. At this point:
# MAGIC     - **Unity Gateway** has received the HTTP POST and has resolved the URL to connect to the Genie One MCP server. 
# MAGIC     - It has also authenticated you to check that you have `EXECUTE` permission on `system.ai`. 
# MAGIC     - The JSON-RPC request has been forwarded to the server. Let's now poll for the results. To do this, we need to extract the `conversation_id` and `response_id` from the ask result.
# MAGIC

# COMMAND ----------

import json 

response = json.loads(ask_result.content[0].text)
conversation_id = response["conversation_id"]
response_id = response["response_id"]
status = response["status"]

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC Now begin polling (we will do 30 polls with 3 seconds between each). 
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Why <code>await acall_tool()</code> instead of <code>call_tool()</code>?</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Notebook cells on <strong>Serverless</strong> execute inside a running <code>asyncio</code> event loop. The sync <code>call_tool()</code> is a convenience wrapper around the async <code>acall_tool()</code> — internally it creates an <code>asyncio.TaskGroup</code> and calls <code>loop.run_until_complete()</code> to bridge from sync to async. When that bridge runs inside an <em>already-running</em> event loop, the <code>TaskGroup</code>'s exception propagation can conflict with the outer loop's task management, raising an <code>ExceptionGroup</code> that escapes normal <code>try/except</code> handling.</p>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">The fix is to stay fully async: use <code>await acall_tool()</code> and <code>await asyncio.sleep()</code> so the code cooperates with the event loop rather than nesting inside it. Databricks Serverless supports top-level <code>await</code> in cells, making this straightforward.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

import asyncio

max_polls = 30
poll_text = ""
for i in range(max_polls):
    await asyncio.sleep(3)

    poll_result = await mcp_client_managed.acall_tool("genie_poll_response", {
            "conversation_id": conversation_id,
            "response_id": response_id
        })
    poll_text = poll_result.content[0].text or ""

    if "Genie has finished" in poll_text:
        status = "completed"
        break
    elif "Still running" in poll_text:
        # Show progress steps extracted from the text
        steps = [line.strip() for line in poll_text.split("\n") if line.strip().startswith("\u2022")]
        for step in steps:
            print(f"  \u2192 {step}")
        print(f"  \u23f3 Poll {i + 1}: Still processing...")
    else:
        print(f"  \u23f3 Poll {i + 1}: Waiting for response...")

print(f"\nFinal status: {status}")

# The completed response includes the answer narrative
if status == "completed":
    print(poll_text)

    # Step 3: If Genie ran SQL, try to fetch the full result set
    try:
        query_result = mcp_client_managed.call_tool("genie_get_query_result", {
            "conversation_id": conversation_id,
            "response_id": response_id
        })
        data = json.loads(query_result.content[0].text)
        print(f"\nColumns: {[col['name'] for col in data.get('columns', [])]}")
        for row in data.get("rows", [])[:5]:
            print(row)
    except (json.JSONDecodeError, TypeError, IndexError, KeyError, BaseException):
        pass  # Query result may not be available as structured data

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC You've now learned how to create a comprehensive AI agent toolkit by combining both SQL and Python functions in Unity Catalog and exposing them via the Model Context Protocol (MCP). Throughout this demonstration, you've gained hands-on experience with:
# MAGIC
# MAGIC - **Building both SQL and Python UC functions** for AI agents using multiple registration approaches
# MAGIC - **Understanding the strengths of each approach** - SQL for data analysis and Python for external integrations and complex logic
# MAGIC - **Discovering and calling tools via MCP** - using the Databricks-managed MCP server to dynamically discover UC functions and invoke them programmatically, just as an agent framework would
# MAGIC - **Connecting to Databricks managed MCPs** like Genie One to query data with natural language, demonstrating the breadth of the MCP ecosystem beyond custom tools
# MAGIC - **Testing functions individually and collectively** through multiple methods including direct execution, `DatabricksFunctionClient()`, MCP, a model, and Genie One
# MAGIC - **Creating comprehensive agent toolkits** that can handle diverse user queries requiring both data analysis and external information
# MAGIC - **Seeing how a model selects tools** by offering it your SQL and Python tools and watching which ones it calls, and with what arguments
# MAGIC
# MAGIC By combining Unity Catalog's governance framework with MCP's open standard for tool discovery, SQL's analytical capabilities, and Python's computational flexibility, you're now equipped to build secure, scalable AI agent solutions that can intelligently interact with both your internal data and external systems. This comprehensive approach enables you to create production-ready agent tools that leverage the best of both worlds while maintaining enterprise governance and security standards.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>