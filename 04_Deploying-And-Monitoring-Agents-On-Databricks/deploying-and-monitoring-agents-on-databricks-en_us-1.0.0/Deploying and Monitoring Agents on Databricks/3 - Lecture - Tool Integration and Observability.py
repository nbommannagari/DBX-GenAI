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

# MAGIC %md
# MAGIC # Lecture - Tool Integration and Observability
# MAGIC
# MAGIC ## Overview
# MAGIC This lecture covers how deployed agents connect to external tools through the Model Context Protocol (MCP) and how MLflow Tracing provides the observability foundation for production monitoring. Together, these capabilities make agents extensible and observable — the prerequisites for evaluation and monitoring covered later in the course.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lecture you will be able to:
# MAGIC - Describe how MCP enables runtime tool discovery and invocation for deployed agents
# MAGIC - Differentiate managed, external, and custom MCP servers on Databricks
# MAGIC - Explain the MCP connection lifecycle from initialization through shutdown
# MAGIC - Identify what MLflow Tracing captures and how it connects deployment to evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Model Context Protocol

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. What Is MCP?
# MAGIC
# MAGIC The **Model Context Protocol (MCP)** is an open protocol for connecting AI applications to external systems. Think of it like a USB-C port for AI: just as USB-C provides a standardized physical interface for connecting peripherals, MCP provides a standardized software interface that lets AI applications communicate with data sources, tools, and workflows.
# MAGIC
# MAGIC Without MCP, every tool integration requires custom code: API wrappers, authentication handling, response parsing. MCP eliminates this by defining a common protocol that any compliant client and server can speak.
# MAGIC
# MAGIC MCP follows a **client-server architecture** with three participants:
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;font-family:system-ui,sans-serif;">
# MAGIC   <div style="border:2px solid #2574B5;border-radius:10px;padding:12px 14px;background:#fff;">
# MAGIC     <div style="font-size:14px;font-weight:600;color:#2574B5;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">MCP Host (Your App)</div>
# MAGIC     <div style="background:#2574B5;color:#fff;padding:8px 14px;border-radius:6px;font-weight:600;font-size:14px;text-align:center;">MCP Client</div>
# MAGIC   </div>
# MAGIC   <div style="display:flex;flex-direction:column;align-items:center;padding:0 8px;">
# MAGIC     <div style="color:#02A36F;font-size:14px;font-weight:600;">tools/list</div>
# MAGIC     <div style="color:#999;font-size:20px;">&#x2194;</div>
# MAGIC     <div style="color:#FF5F46;font-size:14px;font-weight:600;">tools/call</div>
# MAGIC   </div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:14px 18px;border-radius:10px;font-weight:600;font-size:14px;text-align:center;min-width:110px;">MCP Server<br/><span style="font-weight:400;font-size:14px;">Tools, Resources, Prompts</span></div>
# MAGIC </div>
# MAGIC
# MAGIC | Participant | Role |
# MAGIC |---|---|
# MAGIC | **MCP Host** | The AI application (e.g., a Databricks App) that manages one or more MCP clients |
# MAGIC | **MCP Client** | A component inside the host that maintains a dedicated connection to a single MCP server |
# MAGIC | **MCP Server** | A program that exposes tools, resources, or prompts to connected clients |
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">MCP servers expose three types of primitives: <strong>Tools</strong> (executable functions), <strong>Resources</strong> (read-only data sources), and <strong>Prompts</strong> (reusable interaction templates). In practice, tool primitives are the most common integration point for agents.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Model Context Protocol overview ([AWS](https://docs.databricks.com/aws/en/generative-ai/mcp/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/mcp/))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A2. MCP on Databricks
# MAGIC
# MAGIC Databricks provides three categories of MCP server to cover different integration needs:
# MAGIC
# MAGIC | Type | Description | Example |
# MAGIC |---|---|---|
# MAGIC | **Managed MCP** | Pre-configured servers that give immediate access to Databricks features | Unity Catalog functions, Genie spaces, AI Search indexes |
# MAGIC | **External MCP** | Managed connections to MCP servers hosted outside of Databricks | Third-party APIs, SaaS integrations |
# MAGIC | **Custom MCP** | A custom MCP server you write and host as a Databricks App | Domain-specific tools, internal services |
# MAGIC
# MAGIC The most common starting point is a **Managed MCP server backed by Unity Catalog functions**. You write a Python function, register it in UC, and the managed MCP server automatically exposes it as a tool that any connected agent can discover and invoke. The agent does not need custom code to call the function. The MCP protocol handles discovery, invocation, and response formatting.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">A single host can connect to multiple servers simultaneously, each through its own client instance. This lets an agent access tools from Unity Catalog functions, external APIs, and custom services all through the same protocol.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Managed MCP servers ([AWS](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/managed-mcp) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/mcp/managed-mcp))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/deploying-and-monitoring-agents-on-databricks-en_us-1.0.0/images/20261010T172734Z/Deploying and Monitoring Agents on Databricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to learn more about <strong>MCP server types on Databricks</strong>? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/deploying-and-monitoring-agents-on-databricks-en_us-1.0.0/images/20261010T172734Z/Deploying and Monitoring Agents on Databricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-query" style="flex: 1;">What managed MCP servers are available on Databricks? How do I connect a Genie space as an MCP server in my agent?</span>
# MAGIC     <button onclick="
# MAGIC       var text = document.getElementById('genie-query').innerText;
# MAGIC       var ta = document.createElement('textarea');
# MAGIC       ta.value = text;
# MAGIC       ta.style.position = 'fixed';
# MAGIC       ta.style.opacity = '0';
# MAGIC       document.body.appendChild(ta);
# MAGIC       ta.select();
# MAGIC       document.execCommand('copy');
# MAGIC       document.body.removeChild(ta);
# MAGIC       this.innerText = 'Copied!';
# MAGIC       var btn = this;
# MAGIC       setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);
# MAGIC     " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A3. MCP Lifecycle
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC Every MCP connection follows a structured lifecycle:
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="background:#2574B5;color:#fff;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:110px;">Initialize<br/><span style="font-weight:400;font-size:14px;">Capability handshake</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="border:2px dashed #999;border-radius:10px;padding:8px 10px;display:flex;align-items:center;gap:0;">
# MAGIC     <div style="position:relative;top:-20px;left:4px;font-size:14px;font-weight:600;color:#999;letter-spacing:0.5px;text-transform:uppercase;position:absolute;margin-top:-18px;">Operation</div>
# MAGIC     <div style="background:#02A36F;color:#fff;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:110px;">Discover<br/><span style="font-weight:400;font-size:14px;">tools/list</span></div>
# MAGIC     <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC     <div style="background:#FF5F46;color:#fff;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:110px;">Invoke<br/><span style="font-weight:400;font-size:14px;">tools/call</span></div>
# MAGIC   </div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:110px;">Shutdown<br/><span style="font-weight:400;font-size:14px;">Close connection</span></div>
# MAGIC </div>
# MAGIC
# MAGIC - **Initialization**: The client sends an `initialize` request declaring its protocol version and capabilities. The server responds with its own capabilities. The client then sends an `initialized` notification to confirm readiness before normal message exchange begins.
# MAGIC - **Tool discovery**: The client sends a `tools/list` request. The server returns all available tools with their names, descriptions, and input schemas. The agent uses these descriptions to decide when to call each tool.
# MAGIC - **Tool invocation**: When the agent decides to use a tool, the client sends a `tools/call` request with the tool name and arguments. The server executes the function and returns the result.
# MAGIC - **Shutdown**: Either side can close the connection gracefully.
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Connecting to UC functions via MCP</summary>
# MAGIC <div class="cb"><pre><span class="cm"># MCP integration using the Databricks OpenAI connector</span>
# MAGIC <span class="kw">from</span> databricks.sdk <span class="kw">import</span> WorkspaceClient
# MAGIC <span class="kw">from</span> databricks_openai.agents <span class="kw">import</span> McpServer
# MAGIC <span class="kw">from</span> agents <span class="kw">import</span> Agent, Runner
# MAGIC
# MAGIC workspace_client = <span class="fn">WorkspaceClient</span>()
# MAGIC
# MAGIC <span class="cm"># 1. Initialize -- connect to UC functions as an MCP server (async context manager)</span>
# MAGIC <span class="kw">async with</span> <span class="fn">McpServer.from_uc_function</span>(
# MAGIC     catalog=<span class="st">"my_catalog"</span>,
# MAGIC     schema=<span class="st">"my_schema"</span>,
# MAGIC     workspace_client=workspace_client,
# MAGIC     name=<span class="st">"uc-functions"</span>,
# MAGIC ) <span class="kw">as</span> uc_server:
# MAGIC     <span class="cm"># 2. Discovery -- tools are listed automatically</span>
# MAGIC     <span class="cm"># 3. Invocation -- agent calls tools as needed</span>
# MAGIC     agent = <span class="fn">Agent</span>(
# MAGIC         name=<span class="st">"my_agent"</span>,
# MAGIC         model=<span class="st">"databricks-claude-sonnet-4-5"</span>,
# MAGIC         mcp_servers=[uc_server],
# MAGIC     )
# MAGIC     result = <span class="kw">await</span> <span class="fn">Runner.run</span>(agent, <span class="st">"Look up info for Acme Corp"</span>)</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Tools are <strong>discovered at runtime</strong>, not hardcoded. If a new UC function is registered on the MCP server, connected agents can discover and use it without redeployment. When MLflow Tracing is active, every MCP tool call is captured as a span within the trace.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A4. MCP in a Deployed App
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC In a Databricks App, MCP servers are typically initialized at startup and shared across requests. The app's service principal must have permissions to access the underlying resources (UC functions, external endpoints) that the MCP servers expose.
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: FastAPI server with MCP tool integration</summary>
# MAGIC <div class="cb"><pre><span class="cm"># server.py -- FastAPI app with MCP integration</span>
# MAGIC <span class="kw">from</span> contextlib <span class="kw">import</span> asynccontextmanager
# MAGIC <span class="kw">from</span> fastapi <span class="kw">import</span> FastAPI
# MAGIC <span class="kw">from</span> databricks.sdk <span class="kw">import</span> WorkspaceClient
# MAGIC <span class="kw">from</span> databricks_openai.agents <span class="kw">import</span> McpServer
# MAGIC <span class="kw">from</span> agents <span class="kw">import</span> Agent, Runner
# MAGIC <span class="kw">import</span> os
# MAGIC
# MAGIC workspace_client = <span class="fn">WorkspaceClient</span>()
# MAGIC
# MAGIC <span class="dc">@asynccontextmanager</span>
# MAGIC <span class="kw">async def</span> <span class="fn">lifespan</span>(app):
# MAGIC     <span class="cm"># Initialize MCP server at startup (async context manager)</span>
# MAGIC     <span class="kw">async with</span> <span class="fn">McpServer.from_uc_function</span>(
# MAGIC         catalog=os.environ[<span class="st">"UC_TOOL_CATALOG"</span>],
# MAGIC         schema=os.environ[<span class="st">"UC_TOOL_SCHEMA"</span>],
# MAGIC         workspace_client=workspace_client,
# MAGIC         name=<span class="st">"uc-functions"</span>,
# MAGIC     ) <span class="kw">as</span> uc_server:
# MAGIC         app.state.agent = <span class="fn">Agent</span>(
# MAGIC             name=<span class="st">"deployed_agent"</span>,
# MAGIC             model=os.environ[<span class="st">"SERVING_ENDPOINT"</span>],
# MAGIC             mcp_servers=[uc_server],
# MAGIC         )
# MAGIC         <span class="kw">yield</span>
# MAGIC
# MAGIC app = <span class="fn">FastAPI</span>(lifespan=lifespan)
# MAGIC
# MAGIC <span class="dc">@app.post</span>(<span class="st">"/chat"</span>)
# MAGIC <span class="kw">async def</span> <span class="fn">chat</span>(request: ChatRequest):
# MAGIC     result = <span class="kw">await</span> <span class="fn">Runner.run</span>(app.state.agent, request.message)
# MAGIC     <span class="kw">return</span> {<span class="st">"response"</span>: result.final_output}</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">The UC catalog and schema are injected as environment variables through <code>databricks.yml</code>, keeping the app code portable across environments. The service principal assigned to the app must have <code>EXECUTE</code> permission on the UC functions. The <code>McpServer.from_uc_function</code> must be used as an async context manager to properly manage the server connection lifecycle.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Observability with MLflow Tracing

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Adding Tracing to Your Agent
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC MLflow Tracing records the execution of each agent request as a structured **trace**, a tree of **spans** that represent individual operations. The simplest way to enable tracing is the `@mlflow.trace` decorator on your agent's entry point.
# MAGIC
# MAGIC Each span captures:
# MAGIC - **Inputs and outputs**: What went into the operation and what came out
# MAGIC - **Timing**: Start time, end time, and duration
# MAGIC - **Span type**: The kind of operation (e.g., `AGENT`, `CHAT_MODEL`, `TOOL`, `RETRIEVER`)
# MAGIC - **Status**: Whether the operation succeeded or failed
# MAGIC
# MAGIC If the agent calls supported libraries (e.g., the OpenAI SDK), child spans are generated automatically for each library call. You can also add manual spans with `mlflow.start_span()` for custom logic.
# MAGIC
# MAGIC <div style="margin:18px auto;font-family:'Menlo','Consolas',monospace;font-size:14px;line-height:2;background:#f7f8fc;border:1px solid #dde0e8;border-radius:10px;padding:18px 22px;max-width:480px;">
# MAGIC   <div style="color:#2574B5;font-weight:700;">AGENT <span style="color:#999;font-weight:400;">handle_request</span></div>
# MAGIC   <div style="padding-left:20px;border-left:2px solid #dde0e8;margin-left:8px;">
# MAGIC     <div style="color:#F8A805;font-weight:700;">CHAT_MODEL <span style="color:#999;font-weight:400;">chat.completions</span></div>
# MAGIC     <div style="color:#02A36F;font-weight:700;">TOOL <span style="color:#999;font-weight:400;">lookup_weather</span></div>
# MAGIC     <div style="color:#F8A805;font-weight:700;">CHAT_MODEL <span style="color:#999;font-weight:400;">chat.completions</span></div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: @mlflow.trace decorator and manual spans</summary>
# MAGIC <div class="cb"><pre><span class="kw">import</span> mlflow
# MAGIC
# MAGIC <span class="dc">@mlflow.trace</span>
# MAGIC <span class="kw">def</span> <span class="fn">handle_request</span>(user_message: <span class="fn">str</span>) -> <span class="fn">str</span>:
# MAGIC     <span class="cm"># Child spans generated automatically for OpenAI calls</span>
# MAGIC     <span class="cm"># CHAT_MODEL spans capture LLM interactions</span>
# MAGIC     <span class="cm"># TOOL spans capture MCP tool invocations</span>
# MAGIC     response = agent.<span class="fn">run</span>(user_message)
# MAGIC     <span class="kw">return</span> response
# MAGIC
# MAGIC <span class="cm"># Manual span for custom logic</span>
# MAGIC <span class="dc">@mlflow.trace</span>
# MAGIC <span class="kw">def</span> <span class="fn">handle_request_v2</span>(user_message: <span class="fn">str</span>) -> <span class="fn">str</span>:
# MAGIC     <span class="kw">with</span> mlflow.<span class="fn">start_span</span>(name=<span class="st">"preprocessing"</span>) <span class="kw">as</span> span:
# MAGIC         cleaned = <span class="fn">preprocess</span>(user_message)
# MAGIC         span.<span class="fn">set_inputs</span>({<span class="st">"raw"</span>: user_message})
# MAGIC         span.<span class="fn">set_outputs</span>({<span class="st">"cleaned"</span>: cleaned})
# MAGIC     response = agent.<span class="fn">run</span>(cleaned)
# MAGIC     <span class="kw">return</span> response</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">A typical agent trace contains a root <code>AGENT</code> span with child spans for <code>CHAT_MODEL</code> calls, <code>TOOL</code> invocations, and <code>RETRIEVER</code> operations. This hierarchy shows exactly how the agent processed the request: which tools it called, in what order, and how long each step took.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC MLflow Tracing with function decorators ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/app-instrumentation/manual-tracing/function-decorator) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/app-instrumentation/manual-tracing/function-decorator) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/app-instrumentation/manual-tracing/function-decorator))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Traces in Production
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC In production, traces are logged to an **MLflow Experiment** backed by Unity Catalog. This means traces are stored as governed objects with full lineage and access control. The experiment serves as the central hub where all observability and evaluation data converges.
# MAGIC
# MAGIC Traces are the foundation that the rest of the monitoring pipeline builds on:
# MAGIC - **Scorers evaluate traces**, not raw requests. Any agent that produces MLflow traces can be evaluated using the same scorer infrastructure.
# MAGIC - **Trace archival** streams traces to Delta tables for SQL-based analysis and dashboarding.
# MAGIC - **Metric backfill** applies new scorers retroactively to traces logged before the scorer existed.
# MAGIC
# MAGIC The experiment name is typically injected as an environment variable through `databricks.yml`, so the agent code sets it at startup:
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Setting the MLflow experiment at startup</summary>
# MAGIC <div class="cb"><pre><span class="kw">import</span> mlflow
# MAGIC <span class="kw">import</span> os
# MAGIC
# MAGIC mlflow.<span class="fn">set_experiment</span>(os.environ[<span class="st">"MLFLOW_EXPERIMENT_NAME"</span>])</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Tracing must be enabled before scorers can be useful. If your agent is not producing traces, there is nothing for scorers to evaluate. The first step in any monitoring setup is ensuring traces are flowing to an MLflow Experiment.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC MLflow Tracing overview ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B3. Storing Traces in Unity Catalog
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC By default, traces are stored in the MLflow control plane. For production workloads, you can store OpenTelemetry traces directly in **Unity Catalog Delta tables**. This gives you long-term retention, SQL-queryable trace data, and governance through UC schema and table permissions rather than experiment-level ACLs.
# MAGIC
# MAGIC To bind an experiment to Unity Catalog storage, pass a `trace_location` parameter when creating or setting the experiment:
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Creating a UC-backed experiment</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.entities.trace_location <span class="kw">import</span> UnityCatalog
# MAGIC
# MAGIC experiment = mlflow.<span class="fn">set_experiment</span>(
# MAGIC     experiment_name=<span class="st">"my-agent-experiment"</span>,
# MAGIC     trace_location=<span class="fn">UnityCatalog</span>(
# MAGIC         catalog_name=<span class="st">"my_catalog"</span>,
# MAGIC         schema_name=<span class="st">"my_schema"</span>,
# MAGIC         table_prefix=<span class="st">"agent_traces"</span>,
# MAGIC     ),
# MAGIC )</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC This automatically creates four Delta tables in the specified schema:
# MAGIC
# MAGIC | Table | Contents |
# MAGIC |---|---|
# MAGIC | `<prefix>_otel_spans` | Individual span records (the primary trace data) |
# MAGIC | `<prefix>_otel_annotations` | Assessments and feedback attached to traces |
# MAGIC | `<prefix>_otel_logs` | Log entries associated with trace execution |
# MAGIC | `<prefix>_otel_metrics` | Metric values computed by scorers |
# MAGIC
# MAGIC Once stored in Delta tables, traces can be queried directly with SQL through a Databricks SQL warehouse, enabling custom reporting and dashboards.
# MAGIC
# MAGIC **Governance**: Access is controlled through UC table permissions. Users need `USE_CATALOG`, `USE_SCHEMA`, and explicit `MODIFY` + `SELECT` grants on each table. Note that `ALL_PRIVILEGES` is **not sufficient** — `MODIFY` and `SELECT` must be granted explicitly.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Once you bind an experiment to a UC trace location, you <strong>cannot reassign</strong> it to a different location. However, multiple experiments can share the same UC location. Users with access to the UC tables can view all traces stored there, regardless of which experiment the traces belong to. Ingestion is capped at 200 traces per second per workspace and 100 MB per second per table.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Traces in Unity Catalog ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/trace-unity-catalog) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/trace-unity-catalog))

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC