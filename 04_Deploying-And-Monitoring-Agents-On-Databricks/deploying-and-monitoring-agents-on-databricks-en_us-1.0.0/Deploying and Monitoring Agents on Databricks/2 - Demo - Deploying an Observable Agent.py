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
# MAGIC # Demo - Deploying an Observable Agent
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This demo progressively builds a production-ready agent with deployment, observability, and tool integration in a single flow. You will start with a bare-bones agent deployed as a Databricks App via Declarative Automation Bundles (DABs), layer on MLflow tracing for full request-level visibility, and then attach a Unity Catalog function as a tool through the Model Context Protocol (MCP). By the end, every agent interaction (including tool calls) will produce structured MLflow traces you can inspect in the Databricks UI.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this demonstration, you will be able to:
# MAGIC
# MAGIC 1. Deploy an agent on Databricks Apps using DABs
# MAGIC 2. Add MLflow tracing for observability
# MAGIC 3. Integrate tools via MCP and Unity Catalog functions
# MAGIC 4. Verify traces in the MLflow UI

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. Required Compute Environment
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Select Serverless Compute</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Before starting this notebook, select the required compute environment listed below.</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Serverless Compute, Version 5</strong>: How to select an environment version (<a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V5</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="
# MAGIC border-left: 4px solid #F44336;
# MAGIC background: #FFEBEE;
# MAGIC padding: 14px 18px;
# MAGIC border-radius: 4px;
# MAGIC margin: 16px 0;
# MAGIC ">
# MAGIC <strong style="display:block; color:#c62828; margin-bottom:6px; font-size: 1.1em;">GIT FOLDER REQUIREMENT</strong>
# MAGIC <div style="color:#333;">
# MAGIC
# MAGIC The Declarative Automation Bundles (DABs) UI is only available when you are working inside a Git folder in the workspace.
# MAGIC
# MAGIC Before opening the DABs UI, make sure you have:
# MAGIC - Created or cloned a Git folder in the workspace with all associated labs files
# MAGIC - Git is outside the scope of this lab. For more information visit the Databricks Git folders ([AWS](https://docs.databricks.com/aws/en/repos/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/repos/) | [GCP](https://docs.databricks.com/gcp/en/repos/)) documentation.
# MAGIC - **If you are working in Vocareum:** Vocareum does not automatically come with a git folder as a part of the workspace creation. Please use the official documentation to set up Git Folders if working in **Databricks Workspace**.
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Classroom Setup

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-2

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Define and Deploy a Basic Agent
# MAGIC
# MAGIC In this section you will create the files needed to deploy a simple chatbot as a Databricks App. The agent uses the OpenAI Agents SDK with its client redirected to a Databricks AI Model Serving endpoint.
# MAGIC
# MAGIC **Note🚨:** Please replace the code below in the files inside the **`simple-agent`** folder.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. Project Files
# MAGIC
# MAGIC The basic agent requires six files. Select each tab below to view the file contents and an explanation of its role.
# MAGIC
# MAGIC <style>
# MAGIC .c1-tab-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 0; }
# MAGIC .c1-tab {
# MAGIC   background: #F9F7F4; border: 2px solid transparent; border-top: 5px solid #1B5162;
# MAGIC   border-radius: 8px 8px 0 0; padding: 10px 18px; cursor: pointer; user-select: none;
# MAGIC   font-weight: 700; font-size: 13pt; color: #0b2026; transition: background 0.15s, transform 0.1s;
# MAGIC }
# MAGIC .c1-tab:hover { transform: translateY(-2px); }
# MAGIC .c1-tab.c1-active { background: #fff; border-left-color: #1B5162; border-right-color: #1B5162; border-bottom-color: #fff; }
# MAGIC .c1-panel-wrap {
# MAGIC   border: 2px solid #1B5162; border-top: none; border-radius: 0 0 10px 10px;
# MAGIC   background: #fff; overflow: hidden; max-height: 0; opacity: 0;
# MAGIC   transition: max-height 0.4s ease, opacity 0.3s ease;
# MAGIC }
# MAGIC .c1-panel-wrap.c1-open { max-height: 4000px; opacity: 1; }
# MAGIC .c1-panel { padding: 20px 24px; }
# MAGIC .c1-panel p { font-size: 14pt; line-height: 1.6; color: #333; margin: 0 0 12px 0; }
# MAGIC .c1-panel ul { margin: 8px 0 12px 20px; font-size: 14pt; line-height: 1.6; color: #333; }
# MAGIC .c1-panel li { margin-bottom: 6px; }
# MAGIC .c1-code-wrap { position: relative; margin: 12px 0 16px 0; }
# MAGIC .c1-copy-btn {
# MAGIC   position: absolute; top: 10px; right: 12px; z-index: 2; padding: 5px 12px;
# MAGIC   font-size: 11pt; font-weight: 600; background: rgba(0,0,0,0.06); color: #383a42;
# MAGIC   border: 1px solid rgba(0,0,0,0.1); border-radius: 6px; cursor: pointer; transition: background 0.15s, color 0.15s, border-color 0.15s;
# MAGIC }
# MAGIC .c1-copy-btn:hover { background: rgba(0,0,0,0.1); }
# MAGIC .c1-code {
# MAGIC   border: 1px solid #000; border-radius: 10px; padding: 20px 22px; padding-top: 44px; font-family: 'Menlo','Consolas',monospace;
# MAGIC   font-size: 12.5pt; line-height: 1.75; overflow-x: auto; background: #fafafa; color: #383a42;
# MAGIC }
# MAGIC .c1-code .tk-kw { color: #a626a4; } .c1-code .tk-fn { color: #4078f2; }
# MAGIC .c1-code .tk-str { color: #50a14f; } .c1-code .tk-num { color: #986801; }
# MAGIC .c1-code .tk-cmt { color: #a0a1a7; } .c1-code .tk-dim { color: #696c77; }
# MAGIC .c1-code .tk-key { color: #e45649; }
# MAGIC .c1-code .line { display: block; padding: 1px 6px; border-radius: 3px; transition: background 0.25s, color 0.25s; }
# MAGIC </style>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <div class="c1-tab-row">
# MAGIC   <div class="c1-tab c1-active" data-c1tab="0" onclick="c1TabSelect(0)">agent.py</div>
# MAGIC   <div class="c1-tab" data-c1tab="1" onclick="c1TabSelect(1)">server.py</div>
# MAGIC   <div class="c1-tab" data-c1tab="2" onclick="c1TabSelect(2)">pyproject.toml</div>
# MAGIC   <div class="c1-tab" data-c1tab="3" onclick="c1TabSelect(3)">app.yaml</div>
# MAGIC   <div class="c1-tab" data-c1tab="4" onclick="c1TabSelect(4)">databricks.yml</div>
# MAGIC   <div class="c1-tab" data-c1tab="5" onclick="c1TabSelect(5)">requirements.txt</div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="c1-panel-wrap c1-open" id="c1-panel-wrap">
# MAGIC   <div class="c1-panel" id="c1-panel"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var C1_FILES = [
# MAGIC   {
# MAGIC     name: "agent.py",
# MAGIC     lang: "python",
# MAGIC     desc: '<p>This file defines the agent and an async function to run it. The key integration point is <code>set_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))</code>, which routes all model calls through Databricks instead of the public OpenAI service.</p><p><strong>Key components:</strong></p><ul><li><code>Agent</code> and <code>Runner</code>: Define and execute the agent workflow from the OpenAI Agents SDK</li><li><code>AsyncDatabricksOpenAI</code>: An async client that communicates with Databricks AI Model Serving endpoints using the OpenAI-compatible interface</li><li><code>set_default_openai_api("chat_completions")</code>: Ensures requests use the chat completions format</li><li><code>SERVING_ENDPOINT_NAME</code>: Read from an environment variable injected by the DAB at deploy time</li></ul>',
# MAGIC     code: 'from agents import Agent, Runner, set_default_openai_api, set_default_openai_client\nfrom databricks_openai import AsyncDatabricksOpenAI\nimport os\n\n# Read the endpoint name injected by Databricks Apps\nSERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME")\n\n# Point the OpenAI Agents SDK at Databricks instead of OpenAI\nset_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))\nset_default_openai_api("chat_completions")\n\n# Define the agent\nagent = Agent(\n    name="Simple Chatbot",\n    instructions="You are a helpful assistant. Answer the user\'s questions clearly and concisely.",\n    model=SERVING_ENDPOINT_NAME,\n)\n\n\nasync def run_agent(messages: list[dict]) -> str:\n    """Run the agent with a list of messages and return the final reply."""\n    result = await Runner.run(agent, messages)\n    return result.final_output'
# MAGIC   },
# MAGIC   {
# MAGIC     name: "server.py",
# MAGIC     lang: "python",
# MAGIC     desc: '<p>This FastAPI server exposes two endpoints: <code>GET /</code> returns a chat HTML page, and <code>POST /invocations</code> forwards conversation messages to the agent and returns the reply.</p>',
# MAGIC     code: 'import uvicorn\nfrom pathlib import Path\n\nfrom fastapi import FastAPI\nfrom fastapi.responses import HTMLResponse\nfrom pydantic import BaseModel\n\nfrom agent import run_agent\n\napp = FastAPI()\n\nCHAT_UI = Path(__file__).parent / "chat.html"\n\n\nclass ChatRequest(BaseModel):\n    input: list[dict]\n\n\n@app.get("/", response_class=HTMLResponse)\nasync def root():\n    return CHAT_UI.read_text()\n\n\n@app.post("/invocations")\nasync def invocations(request: ChatRequest):\n    reply = await run_agent(request.input)\n    return {"output": reply}\n\n\ndef main():\n    uvicorn.run("server:app", host="0.0.0.0", port=8000)\n\n\nif __name__ == "__main__":\n    main()'
# MAGIC   },
# MAGIC   {
# MAGIC     name: "pyproject.toml",
# MAGIC     lang: "toml",
# MAGIC     desc: '<p>Standard Python project metadata and dependency declaration. The <code>[project.scripts]</code> entry creates the <code>start-server</code> CLI command referenced by <code>app.yaml</code>.</p>',
# MAGIC     code: '[project]\nname = "simple-chatbot"\nversion = "0.1.0"\nrequires-python = ">=3.11"\ndependencies = [\n    "openai-agents>=0.6.0",\n    "mcp<2",\n   "databricks-openai==0.15.0",\n    "databricks-sdk==0.108.0",\n    "fastapi>=0.115.0",\n    "uvicorn>=0.34.0",\n    "python-dotenv",\n    "uv",\n    "databricks-vectorsearch==0.68",\n]\n\n[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n[tool.hatch.build.targets.wheel]\ninclude = ["agent.py", "server.py"]\n\n[project.scripts]\nstart-server = "server:main"'
# MAGIC   },
# MAGIC   {
# MAGIC     name: "app.yaml",
# MAGIC     lang: "yaml",
# MAGIC     desc: '<p>The Databricks App manifest. It declares which environment variables to inject (resolved from DAB resources) and the startup command.</p>',
# MAGIC     code: 'env:\n  - name: SERVING_ENDPOINT_NAME\n    valueFrom: serving-endpoint\n\ncommand: ["uv", "run", "start-server"]'
# MAGIC   },
# MAGIC   {
# MAGIC     name: "databricks.yml",
# MAGIC     lang: "yaml",
# MAGIC     desc: '<p>The Declarative Automation Bundle (DAB) manifest. It names the app, grants it access to a model serving endpoint, and targets the <code>dev</code> deployment mode. The app name uses bundle substitutions (<code>${workspace.current_user.id}</code>, <code>${bundle.target}</code>) so each user gets a unique app.</p><div class="callout" style="border-left:4px solid #1976d2;background:rgba(25,118,210,0.06);padding:10px 12px;margin-top:10px;font-size:14pt;line-height:1.6;border-radius:4px;"><strong>Not in a Git folder?</strong> This file is only needed for DABs-based deployment. If you are deploying via the Apps UI instead, skip this file and configure the app resources manually on the app Overview page.</div>',
# MAGIC     code: 'bundle:\n  name: simple_chatbot\n\nresources:\n  apps:\n    simple_chatbot:\n      name: agent-${workspace.current_user.id}-${bundle.target}\n      description: "Simple chatbot agent"\n      source_code_path: ./\n      resources:\n        - name: serving-endpoint\n          serving_endpoint:\n            name: "databricks-gpt-5-2"\n            permission: CAN_QUERY\n\ntargets:\n  dev:\n    mode: development\n    default: true'
# MAGIC   },
# MAGIC   {
# MAGIC     name: "requirements.txt",
# MAGIC     lang: "text",
# MAGIC     desc: '<p>Databricks Apps automatically runs <code>pip install -r requirements.txt</code> during deployment. Installing <code>uv</code> here ensures the <code>uv run start-server</code> command in <code>app.yaml</code> works correctly.</p>',
# MAGIC     code: 'uv'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var c1TabCurrent = 0;
# MAGIC
# MAGIC function c1Esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
# MAGIC
# MAGIC function c1PyTokenize(line) {
# MAGIC   var out = '';
# MAGIC   var re = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b(?:from|import|def|class|return|if|elif|else|for|while|try|except|finally|with|as|pass|break|continue|yield|lambda|async|await|None|True|False|and|or|not|in|is)\b)|(@\w[\w.]*)|(\b\w+(?=\s*\())|(\b\d+\.?\d*\b)/g;
# MAGIC   var last = 0; var m;
# MAGIC   while ((m = re.exec(line)) !== null) {
# MAGIC     if (m.index > last) out += c1Esc(line.substring(last, m.index));
# MAGIC     if (m[1]) out += '<span class="tk-cmt">' + c1Esc(m[1]) + '</span>';
# MAGIC     else if (m[2]) out += '<span class="tk-str">' + c1Esc(m[2]) + '</span>';
# MAGIC     else if (m[3]) out += '<span class="tk-kw">' + c1Esc(m[3]) + '</span>';
# MAGIC     else if (m[4]) out += '<span class="tk-fn">' + c1Esc(m[4]) + '</span>';
# MAGIC     else if (m[5]) out += '<span class="tk-fn">' + c1Esc(m[5]) + '</span>';
# MAGIC     else if (m[6]) out += '<span class="tk-num">' + c1Esc(m[6]) + '</span>';
# MAGIC     last = m.index + m[0].length;
# MAGIC   }
# MAGIC   if (last < line.length) out += c1Esc(line.substring(last));
# MAGIC   return out || '\u00a0';
# MAGIC }
# MAGIC
# MAGIC function c1YamlTokenize(line) {
# MAGIC   var out = '';
# MAGIC   var re = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b(?:true|false|null)\b)|(^\s*)([\w][\w\-\.\[\]]*)(\s*[:=])/gm;
# MAGIC   var last = 0; var m;
# MAGIC   while ((m = re.exec(line)) !== null) {
# MAGIC     if (m.index > last) out += c1Esc(line.substring(last, m.index));
# MAGIC     if (m[1]) out += '<span class="tk-cmt">' + c1Esc(m[1]) + '</span>';
# MAGIC     else if (m[2]) out += '<span class="tk-str">' + c1Esc(m[2]) + '</span>';
# MAGIC     else if (m[3]) out += '<span class="tk-kw">' + c1Esc(m[3]) + '</span>';
# MAGIC     else if (m[5]) { out += c1Esc(m[4]) + '<span class="tk-key">' + c1Esc(m[5]) + '</span>' + c1Esc(m[6]); }
# MAGIC     last = m.index + m[0].length;
# MAGIC   }
# MAGIC   if (last < line.length) out += c1Esc(line.substring(last));
# MAGIC   return out || '\u00a0';
# MAGIC }
# MAGIC
# MAGIC function c1Tokenize(code, lang) {
# MAGIC   var lines = code.split('\n');
# MAGIC   var result = [];
# MAGIC   for (var i = 0; i < lines.length; i++) {
# MAGIC     var raw = lines[i];
# MAGIC     var indent = 0;
# MAGIC     while (indent < raw.length && raw[indent] === ' ') indent++;
# MAGIC     var sp = '';
# MAGIC     for (var j = 0; j < indent; j++) sp += '\u00a0';
# MAGIC     var content = raw.substring(indent);
# MAGIC     var html;
# MAGIC     if (lang === 'python') html = c1PyTokenize(content);
# MAGIC     else if (lang === 'yaml' || lang === 'toml') html = c1YamlTokenize(content);
# MAGIC     else html = c1Esc(content) || '\u00a0';
# MAGIC     result.push('<span class="line">' + sp + html + '</span>');
# MAGIC   }
# MAGIC   return result.join('');
# MAGIC }
# MAGIC
# MAGIC function c1TabSelect(id) {
# MAGIC   c1TabCurrent = id;
# MAGIC   document.querySelectorAll('.c1-tab').forEach(function(t) {
# MAGIC     t.classList.toggle('c1-active', parseInt(t.getAttribute('data-c1tab')) === id);
# MAGIC   });
# MAGIC   var f = C1_FILES[id];
# MAGIC   var btnId = 'c1-copy-btn-' + id;
# MAGIC   var wrapId = 'c1-code-wrap';
# MAGIC   document.getElementById('c1-panel').innerHTML =
# MAGIC     f.desc +
# MAGIC     '<div class="c1-code-wrap" id="' + wrapId + '">' +
# MAGIC       '<button class="c1-copy-btn" id="' + btnId + '" onclick="c1CopyCode(' + id + ')">Copy</button>' +
# MAGIC       '<div class="c1-code">' + c1Tokenize(f.code, f.lang) + '</div>' +
# MAGIC     '</div>';
# MAGIC   document.getElementById('c1-panel-wrap').classList.add('c1-open');
# MAGIC }
# MAGIC
# MAGIC function c1CopyCode(id) {
# MAGIC   var code = C1_FILES[id].code;
# MAGIC   var btn = document.getElementById('c1-copy-btn-' + id);
# MAGIC   navigator.clipboard.writeText(code).then(function() {
# MAGIC     btn.textContent = 'Copied!';
# MAGIC     setTimeout(function() { btn.textContent = 'Copy'; }, 2000);
# MAGIC   }, function() {
# MAGIC     var t = document.createElement('textarea');
# MAGIC     t.value = code; document.body.appendChild(t); t.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(t);
# MAGIC     btn.textContent = 'Copied!';
# MAGIC     setTimeout(function() { btn.textContent = 'Copy'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC
# MAGIC c1TabSelect(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC Run the next cell to generate a unique identifier if you are deploying the app via the UI. That is, you are _not_ working in a git folder.

# COMMAND ----------

user_email = spark.sql("SELECT current_user()").first()[0]
uniqueIdentifier1 = user_email.split("@")[0].replace('labuser','').replace("_","-").replace(".","-")
print("Use the following app name if you are deploying your app via the UI:\n")
print(f"agent-{uniqueIdentifier1}-dev")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C2. Deploy the App
# MAGIC
# MAGIC There are two ways to deploy, depending on whether your workspace files are inside a Git folder. Select a card below.
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .c2d-row { display: block; white-space: nowrap; font-size: 0; width: 100%; box-sizing: border-box; }
# MAGIC
# MAGIC .c2d-box {
# MAGIC   display: inline-block;
# MAGIC   width: 48%;
# MAGIC   margin-right: 4%;
# MAGIC   min-height: 90px;
# MAGIC   background: #F9F7F4;
# MAGIC   border-top: 8px solid transparent;
# MAGIC   border-left: 2px solid transparent;
# MAGIC   border-right: 2px solid transparent;
# MAGIC   border-bottom: 2px solid transparent;
# MAGIC   border-radius: 8px;
# MAGIC   padding: 12px 8px;
# MAGIC   text-align: center;
# MAGIC   box-sizing: border-box;
# MAGIC   cursor: pointer;
# MAGIC   user-select: none;
# MAGIC   vertical-align: top;
# MAGIC   transition: transform 0.12s;
# MAGIC }
# MAGIC
# MAGIC .c2d-box:last-child { margin-right: 0; }
# MAGIC
# MAGIC .c2d-box:hover { transform: translateY(-2px); }
# MAGIC
# MAGIC .c2d-box.active {
# MAGIC   background: #fff;
# MAGIC   border-left-color: var(--pc);
# MAGIC   border-right-color: var(--pc);
# MAGIC   border-bottom-color: var(--pc);
# MAGIC }
# MAGIC
# MAGIC .c2d-title {
# MAGIC   display: block;
# MAGIC   font-size: 13pt;
# MAGIC   font-weight: 700;
# MAGIC   color: #0b2026;
# MAGIC   line-height: 1.3;
# MAGIC   white-space: normal;
# MAGIC   pointer-events: none;
# MAGIC }
# MAGIC
# MAGIC .c2d-subtitle {
# MAGIC   display: block;
# MAGIC   font-size: 11pt;
# MAGIC   font-weight: 400;
# MAGIC   color: #666;
# MAGIC   line-height: 1.3;
# MAGIC   white-space: normal;
# MAGIC   pointer-events: none;
# MAGIC   margin-top: 4px;
# MAGIC }
# MAGIC
# MAGIC .c2d-detail-wrap {
# MAGIC   overflow: hidden;
# MAGIC   max-height: 0;
# MAGIC   opacity: 0;
# MAGIC   transition: max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease;
# MAGIC   margin-top: 0;
# MAGIC }
# MAGIC
# MAGIC .c2d-detail-wrap.open {
# MAGIC   max-height: 1600px;
# MAGIC   opacity: 1;
# MAGIC   margin-top: 12px;
# MAGIC }
# MAGIC
# MAGIC .c2d-detail-card {
# MAGIC   background: #F9F7F4;
# MAGIC   border-radius: 10px;
# MAGIC   padding: 20px;
# MAGIC   box-sizing: border-box;
# MAGIC   border-top: 7px solid #ccc;
# MAGIC }
# MAGIC
# MAGIC .c2d-steps-block {
# MAGIC   border-left: 4px solid;
# MAGIC   border-radius: 0 6px 6px 0;
# MAGIC   padding: 14px 14px;
# MAGIC   margin-bottom: 14px;
# MAGIC   font-size: 14.5pt;
# MAGIC   line-height: 1.6;
# MAGIC }
# MAGIC
# MAGIC .c2d-steps-block ol {
# MAGIC   margin: 8px 0 0 0;
# MAGIC   padding-left: 20px;
# MAGIC }
# MAGIC
# MAGIC .c2d-steps-block li {
# MAGIC   margin-bottom: 8px;
# MAGIC }
# MAGIC
# MAGIC .c2d-steps-block ul {
# MAGIC   margin: 4px 0 8px 20px;
# MAGIC   list-style: disc;
# MAGIC }
# MAGIC
# MAGIC .c2d-info-block {
# MAGIC   border-left: 4px solid;
# MAGIC   border-radius: 0 6px 6px 0;
# MAGIC   padding: 12px 14px;
# MAGIC   font-size: 14pt;
# MAGIC   line-height: 1.5;
# MAGIC   font-weight: 500;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <div style="background: #F8F9FC; border: 3px solid #1B5162; border-radius: 10px; padding: 24px; box-sizing: border-box;">
# MAGIC
# MAGIC   <div class="c2d-row">
# MAGIC     <span class="c2d-box active" data-id="0" onclick="c2dSelect(0)" style="--pc:#1B5162; border-top-color:#1B5162;">
# MAGIC       <span class="c2d-title">DABs UI</span>
# MAGIC       <span class="c2d-subtitle">Git folder</span>
# MAGIC     </span><span class="c2d-box" data-id="1" onclick="c2dSelect(1)" style="--pc:#FF5F46; border-top-color:#FF5F46;">
# MAGIC       <span class="c2d-title">Apps UI</span>
# MAGIC       <span class="c2d-subtitle">No Git folder</span>
# MAGIC     </span>
# MAGIC   </div>
# MAGIC
# MAGIC   <div class="c2d-detail-wrap" id="c2d-detail-wrap">
# MAGIC     <div class="c2d-detail-card" id="c2d-detail-card"></div>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var C2D_CARDS = [
# MAGIC   {
# MAGIC     title: "DABs UI (Git folder)",
# MAGIC     color: "#1B5162",
# MAGIC     bg: "rgba(27,81,98,0.10)",
# MAGIC     infoBg: "rgba(27,81,98,0.10)",
# MAGIC     content: `
# MAGIC       <ol>
# MAGIC         <li>Navigate inside the project folder and click on any file (e.g., <strong>databricks.yml</strong>). A rocket icon will appear. Click it.</li>
# MAGIC         <li>Click <strong>Deploy</strong> in the <strong>Deployments</strong> section. This DAB is configured to deploy to <strong>dev</strong> only.</li>
# MAGIC         <li>Once validation clears and the deployment summary appears, click <strong>Deploy</strong> again to confirm.</li>
# MAGIC         <li>Under <strong>Bundle resource</strong>, click the app link (e.g., <strong>agent-userid-dev</strong>) to open the <strong>Apps</strong> page.</li>
# MAGIC         <li>Click <strong>Start</strong> to provision compute. After a few minutes, click <strong>Deploy</strong> to push the source code to the running app.
# MAGIC           <ul>
# MAGIC             <li>Navigate back to the source folder of the app <strong>simple-agent</strong> and click the three vertical dots next to the folder name. Select <strong>Copy URL/path</strong> and select <strong>full path</strong>.</li>
# MAGIC             <li>Paste this in the <strong>source code path</strong> textbox in the app's <strong>Create deployment</strong> menu.</li>
# MAGIC             <li>Click <strong>Deploy</strong>.</li>
# MAGIC           </ul>
# MAGIC         </li>
# MAGIC         <li>Once the app shows <strong>Running</strong> status, click the endpoint to open the chat UI and verify the agent responds by sending a query like <em>"Please tell me about Apache Spark."</em></li>
# MAGIC       </ol>
# MAGIC     `,
# MAGIC     info: '<strong>Requires a Git folder.</strong> The DABs UI automates resource creation and permission grants from your <code>databricks.yml</code> file.'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "Apps UI (no Git folder)",
# MAGIC     color: "#FF5F46",
# MAGIC     bg: "rgba(255,95,70,0.10)",
# MAGIC     infoBg: "rgba(255,95,70,0.10)",
# MAGIC     content: `
# MAGIC       <ol>
# MAGIC         <li>In the workspace sidebar, click <strong>+ New</strong> and select <strong>App</strong>.</li>
# MAGIC         <li>Click <strong>Create a custom app</strong> in the <strong>Apps</strong> landing page.</li>
# MAGIC         <li>Give the app a name (e.g., <strong>agent-uniqueIdentifier-dev</strong> as indicated in the previous cell) and click <strong>Create app</strong>.</li>
# MAGIC         <li>Wait for the compute to provision.
# MAGIC         </li>
# MAGIC         <li>Configure the app resources manually: add a <strong>model service</strong> resource and map it to <code>GPT 5.2</code> with <code>CAN EXECUTE</code> permission.
# MAGIC           <ul>
# MAGIC             <li>Click on <strong>(⋮)</strong> icon then on  <strong>Edit</strong> at the top right.</li>
# MAGIC             <li>Click <strong>Resources</strong> at the top of the setup wizard.</li>
# MAGIC             <li>Click <strong>+ Add resource</strong> and select <strong>Model</strong>.</li>
# MAGIC             <li>In the dropdown menu search for <strong>GPT 5.2</strong>. Make sure <strong>Permission</strong> is set to <strong>Can execute</strong>. Make sure <strong>Resource key</strong> is set to <strong>serving-endpoint</strong>.</li>
# MAGIC             <li>Click <strong>Save</strong>.</li>
# MAGIC           </ul>
# MAGIC         </li>
# MAGIC         <li>On the app's Overview page, click <strong>Deploy</strong> and set the <strong>Source path</strong> to the workspace folder containing your project files.
# MAGIC           <ul>
# MAGIC             <li>Navigate back to the source folder of the app <strong>simple-agent</strong> and click the three vertical dots next to the folder name. Select <strong>Copy URL/path</strong> and select <strong>full path</strong>.</li>
# MAGIC             <li>Paste this in the <strong>source code path</strong> textbox in the app's <strong>Create deployment</strong> menu.</li>
# MAGIC             <li>Click <strong>Deploy</strong>.</li>
# MAGIC           </ul>
# MAGIC         </li>
# MAGIC         <li>Click the endpoint URL to open the chat UI and verify the agent responds by sending a query like <em>"Please tell me about Apache Spark."</em></li>
# MAGIC       </ol>
# MAGIC     `,
# MAGIC     info: '<strong>No Git folder required.</strong> You configure app resources manually on the Overview page.'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var c2dCurrent = null;
# MAGIC
# MAGIC function c2dSelect(id) {
# MAGIC   var wrap = document.getElementById('c2d-detail-wrap');
# MAGIC   var card = document.getElementById('c2d-detail-card');
# MAGIC   var c = C2D_CARDS[id];
# MAGIC
# MAGIC   document.querySelectorAll('.c2d-box').forEach(function(b) {
# MAGIC     var isActive = parseInt(b.dataset.id, 10) === id;
# MAGIC     b.classList.toggle('active', isActive);
# MAGIC   });
# MAGIC
# MAGIC   if (c2dCurrent === id) {
# MAGIC     wrap.classList.remove('open');
# MAGIC     document.querySelectorAll('.c2d-box').forEach(function(b) {
# MAGIC       b.classList.remove('active');
# MAGIC     });
# MAGIC     c2dCurrent = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   c2dCurrent = id;
# MAGIC
# MAGIC   card.style.borderTopColor = c.color;
# MAGIC   card.innerHTML =
# MAGIC     '<div style="font-size:18pt;font-weight:700;margin-bottom:12px;color:#0b2026;">' + c.title + '</div>'
# MAGIC     + '<div class="c2d-steps-block" style="background:' + c.bg + ';border-color:' + c.color + ';">'
# MAGIC     + c.content
# MAGIC     + '</div>'
# MAGIC     + '<div class="c2d-info-block" style="background:' + c.infoBg + ';border-color:' + c.color + ';">'
# MAGIC     + c.info
# MAGIC     + '</div>';
# MAGIC
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC
# MAGIC c2dSelect(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Add MLflow Tracing
# MAGIC
# MAGIC With the basic agent running, the next step is to add observability. You will update three files to wire in MLflow tracing and experiment tracking so every conversation produces a structured trace.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D1. Updated Files
# MAGIC
# MAGIC Select each tab below to view the updated file contents. Each file now includes MLflow tracing and experiment configuration.
# MAGIC
# MAGIC <style>
# MAGIC .d1-tab-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 0; }
# MAGIC .d1-tab {
# MAGIC   background: #F9F7F4; border: 2px solid transparent; border-top: 5px solid #2574B5;
# MAGIC   border-radius: 8px 8px 0 0; padding: 10px 18px; cursor: pointer; user-select: none;
# MAGIC   font-weight: 700; font-size: 13pt; color: #0b2026; transition: background 0.15s, transform 0.1s;
# MAGIC }
# MAGIC .d1-tab:hover { transform: translateY(-2px); }
# MAGIC .d1-tab.d1-active { background: #fff; border-left-color: #2574B5; border-right-color: #2574B5; border-bottom-color: #fff; }
# MAGIC .d1-panel-wrap {
# MAGIC   border: 2px solid #2574B5; border-top: none; border-radius: 0 0 10px 10px;
# MAGIC   background: #fff; overflow: hidden; max-height: 0; opacity: 0;
# MAGIC   transition: max-height 0.4s ease, opacity 0.3s ease;
# MAGIC }
# MAGIC .d1-panel-wrap.d1-open { max-height: 4000px; opacity: 1; }
# MAGIC .d1-panel { padding: 20px 24px; }
# MAGIC .d1-panel p { font-size: 14pt; line-height: 1.6; color: #333; margin: 0 0 12px 0; }
# MAGIC .d1-panel ul { margin: 8px 0 12px 20px; font-size: 14pt; line-height: 1.6; color: #333; }
# MAGIC .d1-panel li { margin-bottom: 6px; }
# MAGIC .d1-changed-box {
# MAGIC   border-left: 4px solid #2574B5; background: rgba(37,116,181,0.08);
# MAGIC   border-radius: 0 6px 6px 0; padding: 14px 16px; margin: 14px 0; font-size: 13.5pt; line-height: 1.6;
# MAGIC }
# MAGIC .d1-code-wrap { position: relative; margin: 12px 0 16px 0; }
# MAGIC .d1-copy-btn {
# MAGIC   position: absolute; top: 10px; right: 12px; z-index: 2; padding: 5px 12px;
# MAGIC   font-size: 11pt; font-weight: 600; background: rgba(0,0,0,0.06); color: #383a42;
# MAGIC   border: 1px solid rgba(0,0,0,0.1); border-radius: 6px; cursor: pointer; transition: background 0.15s, color 0.15s, border-color 0.15s;
# MAGIC }
# MAGIC .d1-copy-btn:hover { background: rgba(0,0,0,0.1); }
# MAGIC .d1-code {
# MAGIC   border: 1px solid #000; border-radius: 10px; padding: 20px 22px; padding-top: 44px; font-family: 'Menlo','Consolas',monospace;
# MAGIC   font-size: 12.5pt; line-height: 1.75; overflow-x: auto; background: #fafafa; color: #383a42;
# MAGIC }
# MAGIC .d1-code .tk-kw { color: #a626a4; } .d1-code .tk-fn { color: #4078f2; }
# MAGIC .d1-code .tk-str { color: #50a14f; } .d1-code .tk-num { color: #986801; }
# MAGIC .d1-code .tk-cmt { color: #a0a1a7; } .d1-code .tk-dim { color: #696c77; }
# MAGIC .d1-code .tk-key { color: #e45649; }
# MAGIC .d1-code .line { display: block; padding: 1px 6px; border-radius: 3px; transition: background 0.25s, color 0.25s; }
# MAGIC </style>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <div class="d1-tab-row">
# MAGIC   <div class="d1-tab d1-active" data-d1tab="0" onclick="d1TabSelect(0)">agent.py</div>
# MAGIC   <div class="d1-tab" data-d1tab="1" onclick="d1TabSelect(1)">databricks.yml</div>
# MAGIC   <div class="d1-tab" data-d1tab="2" onclick="d1TabSelect(2)">app.yaml</div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="d1-panel-wrap d1-open" id="d1-panel-wrap">
# MAGIC   <div class="d1-panel" id="d1-panel"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var D1_FILES = [
# MAGIC   {
# MAGIC     name: "agent.py",
# MAGIC     lang: "python",
# MAGIC     desc: '<p>Replace the contents of <strong>agent.py</strong> with the version below. The additions are: <code>mlflow</code> and <code>json</code> imports, tracking URI configuration, experiment setup from environment variables, and the <code>@mlflow.trace</code> decorator with <code>mlflow.start_run</code> logging inside <code>run_agent()</code>.</p>',
# MAGIC     changes: '<strong>What changed:</strong><ul><li><code>mlflow.set_tracking_uri("databricks")</code>: Routes all traces and runs to your Databricks workspace</li><li><code>mlflow.set_experiment()</code>: Targets the experiment created by the DAB, using the ID injected via environment variable</li><li><code>@mlflow.trace</code>: Wraps every call in a root span that captures inputs, outputs, and timing</li><li><code>mlflow.start_run(nested=True)</code>: Logs parameters and artifacts to the experiment for each invocation</li></ul>',
# MAGIC     code: 'from agents import Agent, Runner, set_default_openai_api, set_default_openai_client\nfrom databricks_openai import AsyncDatabricksOpenAI\nimport json\nimport mlflow\nimport os\n\n# Point the OpenAI Agents SDK at Databricks\nset_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))\nset_default_openai_api("chat_completions")\n\nmlflow.set_tracking_uri("databricks")\n\n# Read environment variables injected by the DAB\nSERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME")\nexperiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")\nexperiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")\n\nif experiment_id:\n    mlflow.set_experiment(experiment_id=experiment_id)\nelif experiment_name:\n    mlflow.set_experiment(experiment_name)\n\n# Define the agent\nagent = Agent(\n    name="Simple Chatbot",\n    instructions="You are a helpful assistant. Answer the user\'s questions clearly and concisely.",\n    model=SERVING_ENDPOINT_NAME,\n)\n\n\n@mlflow.trace(name="mlflow_chatbot_run", span_type="AGENT", attributes={"agent_name": "MLflow Chatbot"})\nasync def run_agent(messages: list[dict]) -> str:\n    """\n    Run the agent with a list of messages and return the final reply.\n    - @mlflow.trace creates a trace + root span with inputs/outputs.\n    - mlflow.start_run logs params/artifacts to the experiment.\n    """\n    with mlflow.start_run(nested=True):\n        mlflow.log_param("agent_name", agent.name)\n        mlflow.log_param("model_endpoint", agent.model)\n        mlflow.log_param("num_messages", len(messages))\n        mlflow.log_text(json.dumps(messages, indent=2), "inputs/messages.json")\n        result = await Runner.run(agent, messages)\n        final_output = result.final_output\n        mlflow.log_text(final_output, "outputs/final_output.txt")\n        return final_output'
# MAGIC   },
# MAGIC   {
# MAGIC     name: "databricks.yml",
# MAGIC     lang: "yaml",
# MAGIC     desc: '<p>Replace the contents of <strong>databricks.yml</strong> with the version below. This adds an MLflow experiment as a bundle-managed resource and grants the app permission to write to it.</p><div class="callout" style="border-left:4px solid #1976d2;background:rgba(25,118,210,0.06);padding:10px 12px;margin-top:10px;font-size:14pt;line-height:1.6;border-radius:4px;"><strong>Not in a Git folder?</strong> This file update is only needed for DABs-based deployment. If you are using the Apps UI, skip this and add the experiment resource manually on the app Overview page during the deploy step.</div>',
# MAGIC     changes: '<strong>What changed:</strong><ul><li><code>resources.experiments.chatbot_experiment</code>: Declares an MLflow experiment that DABs creates on deploy. This is a new resource added to the same app.</li><li><code>app-experiment</code> resource entry: Grants the app\'s service principal <code>CAN_EDIT</code> on the experiment so it can create runs and log traces</li><li>The app description updated to reflect the new capability. The bundle name and app name remain the same; redeploying updates the existing app in place.</li></ul>',
# MAGIC     code: 'bundle:\n  name: simple_chatbot\n\nresources:\n  experiments:\n    chatbot_experiment:\n      name: /Users/${workspace.current_user.userName}/simple-chatbot-experiment\n  apps:\n    simple_chatbot:\n      name: agent-${workspace.current_user.id}-${bundle.target}\n      description: "Simple chatbot agent"\n      source_code_path: ./\n      resources:\n        - name: serving-endpoint\n          serving_endpoint:\n            name: databricks-gpt-5-2\n            permission: CAN_QUERY\n        - name: app-experiment\n          experiment:\n            experiment_id: ${resources.experiments.chatbot_experiment.id}\n            permission: CAN_EDIT\n\ntargets:\n  dev:\n    mode: development\n    default: true'
# MAGIC   },
# MAGIC   {
# MAGIC     name: "app.yaml",
# MAGIC     lang: "yaml",
# MAGIC     desc: '<p>Replace the contents of <strong>app.yaml</strong> with the version below. This injects the MLflow experiment ID as an environment variable.</p>',
# MAGIC     changes: '<strong>What changed:</strong><ul><li>The new <code>MLFLOW_EXPERIMENT_ID</code> entry uses <code>valueFrom</code> to resolve the experiment ID from the <code>app-experiment</code> resource declared in <code>databricks.yml</code>. The agent reads this at startup to target the correct experiment.</li></ul>',
# MAGIC     code: 'env:\n  - name: SERVING_ENDPOINT_NAME\n    valueFrom: serving-endpoint\n  - name: MLFLOW_EXPERIMENT_ID\n    valueFrom: app-experiment\n\ncommand: ["uv", "run", "start-server"]'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var d1TabCurrent = 0;
# MAGIC
# MAGIC function d1Esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
# MAGIC
# MAGIC function d1PyTokenize(line) {
# MAGIC   var out = '';
# MAGIC   var re = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b(?:from|import|def|class|return|if|elif|else|for|while|try|except|finally|with|as|pass|break|continue|yield|lambda|async|await|None|True|False|and|or|not|in|is)\b)|(@\w[\w.]*)|(\b\w+(?=\s*\())|(\b\d+\.?\d*\b)/g;
# MAGIC   var last = 0; var m;
# MAGIC   while ((m = re.exec(line)) !== null) {
# MAGIC     if (m.index > last) out += d1Esc(line.substring(last, m.index));
# MAGIC     if (m[1]) out += '<span class="tk-cmt">' + d1Esc(m[1]) + '</span>';
# MAGIC     else if (m[2]) out += '<span class="tk-str">' + d1Esc(m[2]) + '</span>';
# MAGIC     else if (m[3]) out += '<span class="tk-kw">' + d1Esc(m[3]) + '</span>';
# MAGIC     else if (m[4]) out += '<span class="tk-fn">' + d1Esc(m[4]) + '</span>';
# MAGIC     else if (m[5]) out += '<span class="tk-fn">' + d1Esc(m[5]) + '</span>';
# MAGIC     else if (m[6]) out += '<span class="tk-num">' + d1Esc(m[6]) + '</span>';
# MAGIC     last = m.index + m[0].length;
# MAGIC   }
# MAGIC   if (last < line.length) out += d1Esc(line.substring(last));
# MAGIC   return out || '\u00a0';
# MAGIC }
# MAGIC
# MAGIC function d1YamlTokenize(line) {
# MAGIC   var out = '';
# MAGIC   var re = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b(?:true|false|null)\b)|(^\s*)([\w][\w\-\.\[\]]*)(\s*[:=])/gm;
# MAGIC   var last = 0; var m;
# MAGIC   while ((m = re.exec(line)) !== null) {
# MAGIC     if (m.index > last) out += d1Esc(line.substring(last, m.index));
# MAGIC     if (m[1]) out += '<span class="tk-cmt">' + d1Esc(m[1]) + '</span>';
# MAGIC     else if (m[2]) out += '<span class="tk-str">' + d1Esc(m[2]) + '</span>';
# MAGIC     else if (m[3]) out += '<span class="tk-kw">' + d1Esc(m[3]) + '</span>';
# MAGIC     else if (m[5]) { out += d1Esc(m[4]) + '<span class="tk-key">' + d1Esc(m[5]) + '</span>' + d1Esc(m[6]); }
# MAGIC     last = m.index + m[0].length;
# MAGIC   }
# MAGIC   if (last < line.length) out += d1Esc(line.substring(last));
# MAGIC   return out || '\u00a0';
# MAGIC }
# MAGIC
# MAGIC function d1Tokenize(code, lang) {
# MAGIC   var lines = code.split('\n');
# MAGIC   var result = [];
# MAGIC   for (var i = 0; i < lines.length; i++) {
# MAGIC     var raw = lines[i];
# MAGIC     var indent = 0;
# MAGIC     while (indent < raw.length && raw[indent] === ' ') indent++;
# MAGIC     var sp = '';
# MAGIC     for (var j = 0; j < indent; j++) sp += '\u00a0';
# MAGIC     var content = raw.substring(indent);
# MAGIC     var html;
# MAGIC     if (lang === 'python') html = d1PyTokenize(content);
# MAGIC     else if (lang === 'yaml' || lang === 'toml') html = d1YamlTokenize(content);
# MAGIC     else html = d1Esc(content) || '\u00a0';
# MAGIC     result.push('<span class="line">' + sp + html + '</span>');
# MAGIC   }
# MAGIC   return result.join('');
# MAGIC }
# MAGIC
# MAGIC function d1TabSelect(id) {
# MAGIC   d1TabCurrent = id;
# MAGIC   document.querySelectorAll('.d1-tab').forEach(function(t) {
# MAGIC     t.classList.toggle('d1-active', parseInt(t.getAttribute('data-d1tab')) === id);
# MAGIC   });
# MAGIC   var f = D1_FILES[id];
# MAGIC   var btnId = 'd1-copy-btn-' + id;
# MAGIC   var wrapId = 'd1-code-wrap';
# MAGIC   var html = f.desc;
# MAGIC   if (f.changes) {
# MAGIC     html += '<div class="d1-changed-box">' + f.changes + '</div>';
# MAGIC   }
# MAGIC   html += '<div class="d1-code-wrap" id="' + wrapId + '">' +
# MAGIC       '<button class="d1-copy-btn" id="' + btnId + '" onclick="d1CopyCode(' + id + ')">Copy</button>' +
# MAGIC       '<div class="d1-code">' + d1Tokenize(f.code, f.lang) + '</div>' +
# MAGIC     '</div>';
# MAGIC   document.getElementById('d1-panel').innerHTML = html;
# MAGIC   document.getElementById('d1-panel-wrap').classList.add('d1-open');
# MAGIC }
# MAGIC
# MAGIC function d1CopyCode(id) {
# MAGIC   var code = D1_FILES[id].code;
# MAGIC   var btn = document.getElementById('d1-copy-btn-' + id);
# MAGIC   navigator.clipboard.writeText(code).then(function() {
# MAGIC     btn.textContent = 'Copied!';
# MAGIC     setTimeout(function() { btn.textContent = 'Copy'; }, 2000);
# MAGIC   }, function() {
# MAGIC     var t = document.createElement('textarea');
# MAGIC     t.value = code; document.body.appendChild(t); t.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(t);
# MAGIC     btn.textContent = 'Copied!';
# MAGIC     setTimeout(function() { btn.textContent = 'Copy'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC
# MAGIC d1TabSelect(0);
# MAGIC </script>

# COMMAND ----------

uniqueIdentifier2 = user_email.split("@")[0].replace('labuser','labuser_').replace(".","_")
print("Use the following MLflow name if you are deploying your app via the UI:\n")
print(f"[dev {uniqueIdentifier2}] simple-chatbot-experiment")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D2. Redeploy
# MAGIC
# MAGIC Redeploy using whichever method you used in Section C2. This update adds an MLflow experiment resource to the app. Select a card below.
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .d2d-row { display: block; white-space: nowrap; font-size: 0; width: 100%; box-sizing: border-box; }
# MAGIC
# MAGIC .d2d-box {
# MAGIC   display: inline-block;
# MAGIC   width: 48%;
# MAGIC   margin-right: 4%;
# MAGIC   min-height: 90px;
# MAGIC   background: #F9F7F4;
# MAGIC   border-top: 8px solid transparent;
# MAGIC   border-left: 2px solid transparent;
# MAGIC   border-right: 2px solid transparent;
# MAGIC   border-bottom: 2px solid transparent;
# MAGIC   border-radius: 8px;
# MAGIC   padding: 12px 8px;
# MAGIC   text-align: center;
# MAGIC   box-sizing: border-box;
# MAGIC   cursor: pointer;
# MAGIC   user-select: none;
# MAGIC   vertical-align: top;
# MAGIC   transition: transform 0.12s;
# MAGIC }
# MAGIC
# MAGIC .d2d-box:last-child { margin-right: 0; }
# MAGIC
# MAGIC .d2d-box:hover { transform: translateY(-2px); }
# MAGIC
# MAGIC .d2d-box.active {
# MAGIC   background: #fff;
# MAGIC   border-left-color: var(--pc);
# MAGIC   border-right-color: var(--pc);
# MAGIC   border-bottom-color: var(--pc);
# MAGIC }
# MAGIC
# MAGIC .d2d-title {
# MAGIC   display: block;
# MAGIC   font-size: 13pt;
# MAGIC   font-weight: 700;
# MAGIC   color: #0b2026;
# MAGIC   line-height: 1.3;
# MAGIC   white-space: normal;
# MAGIC   pointer-events: none;
# MAGIC }
# MAGIC
# MAGIC .d2d-subtitle {
# MAGIC   display: block;
# MAGIC   font-size: 11pt;
# MAGIC   font-weight: 400;
# MAGIC   color: #666;
# MAGIC   line-height: 1.3;
# MAGIC   white-space: normal;
# MAGIC   pointer-events: none;
# MAGIC   margin-top: 4px;
# MAGIC }
# MAGIC
# MAGIC .d2d-detail-wrap {
# MAGIC   overflow: hidden;
# MAGIC   max-height: 0;
# MAGIC   opacity: 0;
# MAGIC   transition: max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease;
# MAGIC   margin-top: 0;
# MAGIC }
# MAGIC
# MAGIC .d2d-detail-wrap.open {
# MAGIC   max-height: 1600px;
# MAGIC   opacity: 1;
# MAGIC   margin-top: 12px;
# MAGIC }
# MAGIC
# MAGIC .d2d-detail-card {
# MAGIC   background: #F9F7F4;
# MAGIC   border-radius: 10px;
# MAGIC   padding: 20px;
# MAGIC   box-sizing: border-box;
# MAGIC   border-top: 7px solid #ccc;
# MAGIC }
# MAGIC
# MAGIC .d2d-steps-block {
# MAGIC   border-left: 4px solid;
# MAGIC   border-radius: 0 6px 6px 0;
# MAGIC   padding: 14px 14px;
# MAGIC   margin-bottom: 14px;
# MAGIC   font-size: 14.5pt;
# MAGIC   line-height: 1.6;
# MAGIC }
# MAGIC
# MAGIC .d2d-steps-block ol {
# MAGIC   margin: 8px 0 0 0;
# MAGIC   padding-left: 20px;
# MAGIC }
# MAGIC
# MAGIC .d2d-steps-block li {
# MAGIC   margin-bottom: 8px;
# MAGIC }
# MAGIC
# MAGIC .d2d-steps-block ul {
# MAGIC   margin: 4px 0 8px 20px;
# MAGIC   list-style: disc;
# MAGIC }
# MAGIC
# MAGIC .d2d-info-block {
# MAGIC   border-left: 4px solid;
# MAGIC   border-radius: 0 6px 6px 0;
# MAGIC   padding: 12px 14px;
# MAGIC   font-size: 14pt;
# MAGIC   line-height: 1.5;
# MAGIC   font-weight: 500;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <div style="background: #F8F9FC; border: 3px solid #2574B5; border-radius: 10px; padding: 24px; box-sizing: border-box;">
# MAGIC
# MAGIC   <div class="d2d-row">
# MAGIC     <span class="d2d-box active" data-id="0" onclick="d2dSelect(0)" style="--pc:#2574B5; border-top-color:#2574B5;">
# MAGIC       <span class="d2d-title">DABs UI</span>
# MAGIC       <span class="d2d-subtitle">Git folder</span>
# MAGIC     </span><span class="d2d-box" data-id="1" onclick="d2dSelect(1)" style="--pc:#FF5F46; border-top-color:#FF5F46;">
# MAGIC       <span class="d2d-title">Apps UI</span>
# MAGIC       <span class="d2d-subtitle">No Git folder</span>
# MAGIC     </span>
# MAGIC   </div>
# MAGIC
# MAGIC   <div class="d2d-detail-wrap" id="d2d-detail-wrap">
# MAGIC     <div class="d2d-detail-card" id="d2d-detail-card"></div>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var D2D_CARDS = [
# MAGIC   {
# MAGIC     title: "DABs UI (Git folder)",
# MAGIC     color: "#2574B5",
# MAGIC     bg: "rgba(37,116,181,0.10)",
# MAGIC     infoBg: "rgba(37,116,181,0.10)",
# MAGIC     content: `
# MAGIC       <ol>
# MAGIC         <li>Click the rocket icon on any file in the project folder.</li>
# MAGIC         <li>Click <strong>Deploy</strong>, wait for validation, then click <strong>Deploy</strong> again.</li>
# MAGIC         <li>Navigate to the app and click <strong>Deploy</strong> to push the updated source.</li>
# MAGIC         <li>Notice that you now have a new <strong>App resource</strong> for your <strong>MLflow experiment</strong>.</li>
# MAGIC         <li>After deployment, click on the URL to open the app.</li>
# MAGIC       </ol>
# MAGIC     `,
# MAGIC     info: '<strong>Requires a Git folder.</strong> The DABs UI detects the new experiment resource in <code>databricks.yml</code> and creates it automatically. Wait for the app status to return to <strong>Running</strong>.'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "Apps UI (no Git folder)",
# MAGIC     color: "#FF5F46",
# MAGIC     bg: "rgba(255,95,70,0.10)",
# MAGIC     infoBg: "rgba(255,95,70,0.10)",
# MAGIC     content: `
# MAGIC       <ol>
# MAGIC         <li>Navigate to the app Overview page and click <strong>(⋮)</strong> icon then on  <strong>Edit</strong> at the top right.</li>
# MAGIC         <li>Click <strong>resources</strong> at the top of the setup wizard.</li>
# MAGIC         <li>Click <strong>+ Add resource</strong> and select <strong>MLflow experiment</strong>.</li>
# MAGIC         <li>In the <strong>Select experiment</strong> dropdown, select <strong>Create new experiment</strong>.</li>
# MAGIC         <li>Give it the name <strong>[dev labuser_XXX] simple-chatbot-experiment</strong>, using your own labuser name as printed above.</li>
# MAGIC         <li>For the <strong>trace storage location</strong> select <strong>In the Experiment</strong>. Note that there is an option for storing MLflow experiments in <strong>Unity Catalog</strong>, which we will not cover in this course. 
# MAGIC         <li>Click <strong>Create experiment</strong>.</li>
# MAGIC         <li>Make sure <strong>Permission</strong> is set to <strong>Can edit</strong> and <strong>Resource key</strong> is set to <strong>app-experiment</strong>.</li>
# MAGIC         <li>Click <strong>Save</strong>.</li>
# MAGIC         <li>Back on the Overview page, click <strong>Deploy</strong> to push the updated source.</li>
# MAGIC       </ol>
# MAGIC     `,
# MAGIC     info: '<strong>No Git folder required.</strong> You add the experiment resource manually on the Overview page. Wait for the app status to return to <strong>Running</strong>.'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var d2dCurrent = null;
# MAGIC
# MAGIC function d2dSelect(id) {
# MAGIC   var wrap = document.getElementById('d2d-detail-wrap');
# MAGIC   var card = document.getElementById('d2d-detail-card');
# MAGIC   var c = D2D_CARDS[id];
# MAGIC
# MAGIC   document.querySelectorAll('.d2d-box').forEach(function(b) {
# MAGIC     var isActive = parseInt(b.dataset.id, 10) === id;
# MAGIC     b.classList.toggle('active', isActive);
# MAGIC   });
# MAGIC
# MAGIC   if (d2dCurrent === id) {
# MAGIC     wrap.classList.remove('open');
# MAGIC     document.querySelectorAll('.d2d-box').forEach(function(b) {
# MAGIC       b.classList.remove('active');
# MAGIC     });
# MAGIC     d2dCurrent = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   d2dCurrent = id;
# MAGIC
# MAGIC   card.style.borderTopColor = c.color;
# MAGIC   card.innerHTML =
# MAGIC     '<div style="font-size:18pt;font-weight:700;margin-bottom:12px;color:#0b2026;">' + c.title + '</div>'
# MAGIC     + '<div class="d2d-steps-block" style="background:' + c.bg + ';border-color:' + c.color + ';">'
# MAGIC     + c.content
# MAGIC     + '</div>'
# MAGIC     + '<div class="d2d-info-block" style="background:' + c.infoBg + ';border-color:' + c.color + ';">'
# MAGIC     + c.info
# MAGIC     + '</div>';
# MAGIC
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC
# MAGIC d2dSelect(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D3. Test and Verify Traces
# MAGIC
# MAGIC With MLflow tracing wired in, every conversation now produces a structured trace. Verify this is working before moving on.
# MAGIC
# MAGIC 1. Open the app URL and send the same query as last time: "Please tell me about Apache Spark."
# MAGIC 2. Wait for the agent to respond.
# MAGIC 3. Navigate to the **Apps** page in the workspace sidebar and select your app.
# MAGIC 4. In the **App resources** section, locate the **MLflow Experiment ID**. Click on it to open the experiment.
# MAGIC 5. Click the **Traces** tab. You should see a trace entry for the query you just sent.
# MAGIC 6. Click the traces to expand it. You will see:
# MAGIC    - A **root span** (`mlflow_chatbot_run` is located under **Details & Timeline**) capturing the full agent invocation with inputs, output, and total latency
# MAGIC    - **Child spans** from the Agents SDK runner showing each step the agent took will appear here as well (though this agent is too simple to have any at this point)
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">At this stage, the trace shows the LLM interaction but no tool calls. The agent can only chat. In the next section, you will add MCP tools and see tool call spans appear as children within the trace.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Add MCP Tools
# MAGIC
# MAGIC With tracing in place, you will now extend the agent to call external tools via the Model Context Protocol (MCP). MCP is an open standard that lets AI applications discover and invoke tools hosted on external servers. On Databricks, Managed MCP servers can expose Unity Catalog functions, Genie spaces, and AI Search indexes through a single interface.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E1. Final Files
# MAGIC
# MAGIC Select each tab below to view the final file versions with MCP tool integration. Each tab highlights what changed from Section D.
# MAGIC
# MAGIC <style>
# MAGIC .e1-tab-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 0; }
# MAGIC .e1-tab {
# MAGIC   background: #F9F7F4; border: 2px solid transparent; border-top: 5px solid #02A36F;
# MAGIC   border-radius: 8px 8px 0 0; padding: 10px 18px; cursor: pointer; user-select: none;
# MAGIC   font-weight: 700; font-size: 13pt; color: #0b2026; transition: background 0.15s, transform 0.1s;
# MAGIC }
# MAGIC .e1-tab:hover { transform: translateY(-2px); }
# MAGIC .e1-tab.e1-active { background: #fff; border-left-color: #02A36F; border-right-color: #02A36F; border-bottom-color: #fff; }
# MAGIC .e1-panel-wrap {
# MAGIC   border: 2px solid #02A36F; border-top: none; border-radius: 0 0 10px 10px;
# MAGIC   background: #fff; overflow: hidden; max-height: 0; opacity: 0;
# MAGIC   transition: max-height 0.4s ease, opacity 0.3s ease;
# MAGIC }
# MAGIC .e1-panel-wrap.e1-open { max-height: 4000px; opacity: 1; }
# MAGIC .e1-panel { padding: 20px 24px; }
# MAGIC .e1-panel p { font-size: 14pt; line-height: 1.6; color: #333; margin: 0 0 12px 0; }
# MAGIC .e1-panel ul { margin: 8px 0 12px 20px; font-size: 14pt; line-height: 1.6; color: #333; }
# MAGIC .e1-panel li { margin-bottom: 6px; }
# MAGIC .e1-changed-box {
# MAGIC   border-left: 4px solid #02A36F; background: rgba(2,163,111,0.08);
# MAGIC   border-radius: 0 6px 6px 0; padding: 14px 16px; margin: 14px 0; font-size: 13.5pt; line-height: 1.6;
# MAGIC }
# MAGIC .e1-code-wrap { position: relative; margin: 12px 0 16px 0; }
# MAGIC .e1-copy-btn {
# MAGIC   position: absolute; top: 10px; right: 12px; z-index: 2; padding: 5px 12px;
# MAGIC   font-size: 11pt; font-weight: 600; background: rgba(0,0,0,0.06); color: #383a42;
# MAGIC   border: 1px solid rgba(0,0,0,0.1); border-radius: 6px; cursor: pointer; transition: background 0.15s, color 0.15s, border-color 0.15s;
# MAGIC }
# MAGIC .e1-copy-btn:hover { background: rgba(0,0,0,0.1); }
# MAGIC .e1-code {
# MAGIC   border: 1px solid #000; border-radius: 10px; padding: 20px 22px; padding-top: 44px; font-family: 'Menlo','Consolas',monospace;
# MAGIC   font-size: 12.5pt; line-height: 1.75; overflow-x: auto; background: #fafafa; color: #383a42;
# MAGIC }
# MAGIC .e1-code .tk-kw { color: #a626a4; } .e1-code .tk-fn { color: #4078f2; }
# MAGIC .e1-code .tk-str { color: #50a14f; } .e1-code .tk-num { color: #986801; }
# MAGIC .e1-code .tk-cmt { color: #a0a1a7; } .e1-code .tk-dim { color: #696c77; }
# MAGIC .e1-code .tk-key { color: #e45649; }
# MAGIC .e1-code .line { display: block; padding: 1px 6px; border-radius: 3px; transition: background 0.25s, color 0.25s; }
# MAGIC </style>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <div class="e1-tab-row">
# MAGIC   <div class="e1-tab e1-active" data-e1tab="0" onclick="e1TabSelect(0)">agent.py</div>
# MAGIC   <div class="e1-tab" data-e1tab="1" onclick="e1TabSelect(1)">app.yaml</div>
# MAGIC   <div class="e1-tab" data-e1tab="2" onclick="e1TabSelect(2)">databricks.yml</div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="e1-panel-wrap e1-open" id="e1-panel-wrap">
# MAGIC   <div class="e1-panel" id="e1-panel"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var E1_FILES = [
# MAGIC   {
# MAGIC     name: "agent.py",
# MAGIC     lang: "python",
# MAGIC     desc: '<p>Replace the contents of <strong>agent.py</strong> with the final version below. This adds MCP imports, a <code>WorkspaceClient</code> for authentication, an <code>McpServer</code> instance pointing at a UC function, the <code>mcp_servers</code> parameter on the Agent, and the <code>async with mcp_server:</code> context manager around <code>Runner.run()</code>.</p>',
# MAGIC     changes: '<strong>What changed from Section D:</strong><ul><li><code>McpServer</code> and <code>WorkspaceClient</code> imports added</li><li><code>McpServer.from_uc_function()</code>: Creates a handle to the Managed MCP server exposing a UC function as a tool</li><li><code>mcp_servers=[mcp_server]</code> on the <code>Agent</code> constructor: Registers the MCP server so the SDK discovers available tools at runtime</li><li><code>async with mcp_server:</code>: Opens the MCP connection before <code>Runner.run()</code> and closes it after, ensuring tools are available during execution</li></ul>',
# MAGIC     code: 'from agents import Agent, Runner, set_default_openai_api, set_default_openai_client\nfrom databricks_openai import AsyncDatabricksOpenAI\nfrom databricks_openai.agents import McpServer\nfrom databricks.sdk import WorkspaceClient\nimport json\nimport mlflow\nimport os\n\n# Point the OpenAI Agents SDK at Databricks\nset_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))\nset_default_openai_api("chat_completions")\n\nmlflow.set_tracking_uri("databricks")\n\n# Read environment variables injected by the DAB\nSERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME")\nexperiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")\nexperiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")\n\nif experiment_id:\n    mlflow.set_experiment(experiment_id=experiment_id)\nelif experiment_name:\n    mlflow.set_experiment(experiment_name)\n\n# MCP configuration\ncatalog_name = os.getenv("CATALOG_NAME")\nfunction_name = os.getenv("FUNCTION_1_NAME")\n\nws = WorkspaceClient()\n\nmcp_server = McpServer.from_uc_function(\n    catalog=catalog_name,\n    schema="agent_deployment",\n    workspace_client=ws,\n    timeout=60.0,\n)\n\n# Define the agent with MCP tool access\nagent = Agent(\n    name="Simple Chatbot",\n    instructions="You are a helpful assistant. Answer the user\'s questions clearly and concisely.",\n    model=SERVING_ENDPOINT_NAME,\n    mcp_servers=[mcp_server],\n)\n\n\n@mlflow.trace(\n    name="tool_calling_chatbot_run",\n    span_type="AGENT",\n    attributes={"agent_name": "Tool-calling Chatbot"},\n)\nasync def run_agent(messages: list[dict]) -> str:\n    with mlflow.start_run(nested=True):\n        mlflow.log_param("agent_name", agent.name)\n        mlflow.log_param("model_endpoint", agent.model)\n        mlflow.log_param("num_messages", len(messages))\n        mlflow.log_text(json.dumps(messages, indent=2), "inputs/messages.json")\n    # McpServer is an async context manager: Opens the connection before the run\n    async with mcp_server:\n        result = await Runner.run(agent, messages)\n    final_output = result.final_output\n    mlflow.log_text(final_output, "outputs/final_output.txt")\n    return final_output'
# MAGIC   },
# MAGIC   {
# MAGIC     name: "app.yaml",
# MAGIC     lang: "yaml",
# MAGIC     desc: '<p>Replace the contents of <strong>app.yaml</strong> with the final version below. This adds <code>CATALOG_NAME</code> and <code>FUNCTION_1_NAME</code> environment variables.</p>',
# MAGIC     changes: '<strong>What changed from Section D:</strong><ul><li>The two new entries use static <code>value</code> fields rather than <code>valueFrom</code>. Update <code>CATALOG_NAME</code> with your actual catalog name (e.g., <code>labuser_sparky_mcspark</code>). The bundle-level <code>variables</code> block in <code>databricks.yml</code> keeps these values parameterized at the bundle level.</li></ul>',
# MAGIC     code: 'env:\n  - name: SERVING_ENDPOINT_NAME\n    valueFrom: serving-endpoint\n  - name: MLFLOW_EXPERIMENT_ID\n    valueFrom: app-experiment\n  - name: CATALOG_NAME\n    value: # <---- Update with your catalog name\n  - name: FUNCTION_1_NAME\n    value: avg_neigh_price\n\ncommand: ["uv", "run", "start-server"]'
# MAGIC   },
# MAGIC   {
# MAGIC     name: "databricks.yml",
# MAGIC     lang: "yaml",
# MAGIC     desc: '<p>Replace the contents of <strong>databricks.yml</strong> with the final version below. This adds a <code>variables</code> block for the catalog and function names, and a <code>uc_securable</code> resource entry to grant the app <code>EXECUTE</code> permission on the UC function.</p><div class="callout" style="border-left:4px solid #1976d2;background:rgba(25,118,210,0.06);padding:10px 12px;margin-top:10px;font-size:14pt;line-height:1.6;border-radius:4px;"><strong>Not in a Git folder?</strong> This file update is only needed for DABs-based deployment. If you are using the Apps UI, skip this and add the UC function resource with <code>EXECUTE</code> permission manually on the app Overview page during the deploy step.</div>',
# MAGIC     changes: '<strong>What changed from Section D:</strong><ul><li><code>variables</code> block: Parameterizes the catalog and function names so they can be overridden at deploy time with <code>--var</code></li><li><code>uc-function-1</code> resource entry: Grants the app\'s service principal <code>EXECUTE</code> permission on the Unity Catalog function so the MCP server can invoke it at runtime</li></ul>',
# MAGIC     code: 'bundle:\n  name: simple_chatbot\n\nvariables:\n  catalog_name:\n    description: "Per-user lab catalog, e.g. labuser_matthew_mccoy"\n    default: # <---- Update with your catalog name\n  function_1_name:\n    description: "Function to grab the average neighborhood price"\n    default: avg_neigh_price\n\nresources:\n  experiments:\n    chatbot_experiment:\n      name: /Users/${workspace.current_user.userName}/simple-chatbot-experiment\n  apps:\n    simple_chatbot:\n      name: agent-${workspace.current_user.id}-${bundle.target}\n      description: "Simple chatbot agent"\n      source_code_path: ./\n      resources:\n        - name: serving-endpoint\n          serving_endpoint:\n            name: databricks-gpt-5-2\n            permission: CAN_QUERY\n        - name: app-experiment\n          experiment:\n            experiment_id: ${resources.experiments.chatbot_experiment.id}\n            permission: CAN_EDIT\n        - name: uc-function-1\n          uc_securable:\n            securable_full_name: ${var.catalog_name}.agent_deployment.${var.function_1_name}\n            securable_type: FUNCTION\n            permission: EXECUTE\n\ntargets:\n  dev:\n    mode: development\n    default: true'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var e1TabCurrent = 0;
# MAGIC
# MAGIC function e1Esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
# MAGIC
# MAGIC function e1PyTokenize(line) {
# MAGIC   var out = '';
# MAGIC   var re = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b(?:from|import|def|class|return|if|elif|else|for|while|try|except|finally|with|as|pass|break|continue|yield|lambda|async|await|None|True|False|and|or|not|in|is)\b)|(@\w[\w.]*)|(\b\w+(?=\s*\())|(\b\d+\.?\d*\b)/g;
# MAGIC   var last = 0; var m;
# MAGIC   while ((m = re.exec(line)) !== null) {
# MAGIC     if (m.index > last) out += e1Esc(line.substring(last, m.index));
# MAGIC     if (m[1]) out += '<span class="tk-cmt">' + e1Esc(m[1]) + '</span>';
# MAGIC     else if (m[2]) out += '<span class="tk-str">' + e1Esc(m[2]) + '</span>';
# MAGIC     else if (m[3]) out += '<span class="tk-kw">' + e1Esc(m[3]) + '</span>';
# MAGIC     else if (m[4]) out += '<span class="tk-fn">' + e1Esc(m[4]) + '</span>';
# MAGIC     else if (m[5]) out += '<span class="tk-fn">' + e1Esc(m[5]) + '</span>';
# MAGIC     else if (m[6]) out += '<span class="tk-num">' + e1Esc(m[6]) + '</span>';
# MAGIC     last = m.index + m[0].length;
# MAGIC   }
# MAGIC   if (last < line.length) out += e1Esc(line.substring(last));
# MAGIC   return out || '\u00a0';
# MAGIC }
# MAGIC
# MAGIC function e1YamlTokenize(line) {
# MAGIC   var out = '';
# MAGIC   var re = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b(?:true|false|null)\b)|(^\s*)([\w][\w\-\.\[\]]*)(\s*[:=])/gm;
# MAGIC   var last = 0; var m;
# MAGIC   while ((m = re.exec(line)) !== null) {
# MAGIC     if (m.index > last) out += e1Esc(line.substring(last, m.index));
# MAGIC     if (m[1]) out += '<span class="tk-cmt">' + e1Esc(m[1]) + '</span>';
# MAGIC     else if (m[2]) out += '<span class="tk-str">' + e1Esc(m[2]) + '</span>';
# MAGIC     else if (m[3]) out += '<span class="tk-kw">' + e1Esc(m[3]) + '</span>';
# MAGIC     else if (m[5]) { out += e1Esc(m[4]) + '<span class="tk-key">' + e1Esc(m[5]) + '</span>' + e1Esc(m[6]); }
# MAGIC     last = m.index + m[0].length;
# MAGIC   }
# MAGIC   if (last < line.length) out += e1Esc(line.substring(last));
# MAGIC   return out || '\u00a0';
# MAGIC }
# MAGIC
# MAGIC function e1Tokenize(code, lang) {
# MAGIC   var lines = code.split('\n');
# MAGIC   var result = [];
# MAGIC   for (var i = 0; i < lines.length; i++) {
# MAGIC     var raw = lines[i];
# MAGIC     var indent = 0;
# MAGIC     while (indent < raw.length && raw[indent] === ' ') indent++;
# MAGIC     var sp = '';
# MAGIC     for (var j = 0; j < indent; j++) sp += '\u00a0';
# MAGIC     var content = raw.substring(indent);
# MAGIC     var html;
# MAGIC     if (lang === 'python') html = e1PyTokenize(content);
# MAGIC     else if (lang === 'yaml' || lang === 'toml') html = e1YamlTokenize(content);
# MAGIC     else html = e1Esc(content) || '\u00a0';
# MAGIC     result.push('<span class="line">' + sp + html + '</span>');
# MAGIC   }
# MAGIC   return result.join('');
# MAGIC }
# MAGIC
# MAGIC function e1TabSelect(id) {
# MAGIC   e1TabCurrent = id;
# MAGIC   document.querySelectorAll('.e1-tab').forEach(function(t) {
# MAGIC     t.classList.toggle('e1-active', parseInt(t.getAttribute('data-e1tab')) === id);
# MAGIC   });
# MAGIC   var f = E1_FILES[id];
# MAGIC   var btnId = 'e1-copy-btn-' + id;
# MAGIC   var wrapId = 'e1-code-wrap';
# MAGIC   var html = f.desc;
# MAGIC   if (f.changes) {
# MAGIC     html += '<div class="e1-changed-box">' + f.changes + '</div>';
# MAGIC   }
# MAGIC   html += '<div class="e1-code-wrap" id="' + wrapId + '">' +
# MAGIC       '<button class="e1-copy-btn" id="' + btnId + '" onclick="e1CopyCode(' + id + ')">Copy</button>' +
# MAGIC       '<div class="e1-code">' + e1Tokenize(f.code, f.lang) + '</div>' +
# MAGIC     '</div>';
# MAGIC   document.getElementById('e1-panel').innerHTML = html;
# MAGIC   document.getElementById('e1-panel-wrap').classList.add('e1-open');
# MAGIC }
# MAGIC
# MAGIC function e1CopyCode(id) {
# MAGIC   var code = E1_FILES[id].code;
# MAGIC   var btn = document.getElementById('e1-copy-btn-' + id);
# MAGIC   navigator.clipboard.writeText(code).then(function() {
# MAGIC     btn.textContent = 'Copied!';
# MAGIC     setTimeout(function() { btn.textContent = 'Copy'; }, 2000);
# MAGIC   }, function() {
# MAGIC     var t = document.createElement('textarea');
# MAGIC     t.value = code; document.body.appendChild(t); t.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(t);
# MAGIC     btn.textContent = 'Copied!';
# MAGIC     setTimeout(function() { btn.textContent = 'Copy'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC
# MAGIC e1TabSelect(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E2. Redeploy
# MAGIC
# MAGIC Redeploy using whichever method you used in Sections C2 and D2. This update adds a UC function resource so the agent can call tools via MCP. Select a card below.
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .e2d-row { display: block; white-space: nowrap; font-size: 0; width: 100%; box-sizing: border-box; }
# MAGIC
# MAGIC .e2d-box {
# MAGIC   display: inline-block;
# MAGIC   width: 48%;
# MAGIC   margin-right: 4%;
# MAGIC   min-height: 90px;
# MAGIC   background: #F9F7F4;
# MAGIC   border-top: 8px solid transparent;
# MAGIC   border-left: 2px solid transparent;
# MAGIC   border-right: 2px solid transparent;
# MAGIC   border-bottom: 2px solid transparent;
# MAGIC   border-radius: 8px;
# MAGIC   padding: 12px 8px;
# MAGIC   text-align: center;
# MAGIC   box-sizing: border-box;
# MAGIC   cursor: pointer;
# MAGIC   user-select: none;
# MAGIC   vertical-align: top;
# MAGIC   transition: transform 0.12s;
# MAGIC }
# MAGIC
# MAGIC .e2d-box:last-child { margin-right: 0; }
# MAGIC
# MAGIC .e2d-box:hover { transform: translateY(-2px); }
# MAGIC
# MAGIC .e2d-box.active {
# MAGIC   background: #fff;
# MAGIC   border-left-color: var(--pc);
# MAGIC   border-right-color: var(--pc);
# MAGIC   border-bottom-color: var(--pc);
# MAGIC }
# MAGIC
# MAGIC .e2d-title {
# MAGIC   display: block;
# MAGIC   font-size: 13pt;
# MAGIC   font-weight: 700;
# MAGIC   color: #0b2026;
# MAGIC   line-height: 1.3;
# MAGIC   white-space: normal;
# MAGIC   pointer-events: none;
# MAGIC }
# MAGIC
# MAGIC .e2d-subtitle {
# MAGIC   display: block;
# MAGIC   font-size: 11pt;
# MAGIC   font-weight: 400;
# MAGIC   color: #666;
# MAGIC   line-height: 1.3;
# MAGIC   white-space: normal;
# MAGIC   pointer-events: none;
# MAGIC   margin-top: 4px;
# MAGIC }
# MAGIC
# MAGIC .e2d-detail-wrap {
# MAGIC   overflow: hidden;
# MAGIC   max-height: 0;
# MAGIC   opacity: 0;
# MAGIC   transition: max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease;
# MAGIC   margin-top: 0;
# MAGIC }
# MAGIC
# MAGIC .e2d-detail-wrap.open {
# MAGIC   max-height: 1600px;
# MAGIC   opacity: 1;
# MAGIC   margin-top: 12px;
# MAGIC }
# MAGIC
# MAGIC .e2d-detail-card {
# MAGIC   background: #F9F7F4;
# MAGIC   border-radius: 10px;
# MAGIC   padding: 20px;
# MAGIC   box-sizing: border-box;
# MAGIC   border-top: 7px solid #ccc;
# MAGIC }
# MAGIC
# MAGIC .e2d-steps-block {
# MAGIC   border-left: 4px solid;
# MAGIC   border-radius: 0 6px 6px 0;
# MAGIC   padding: 14px 14px;
# MAGIC   margin-bottom: 14px;
# MAGIC   font-size: 14.5pt;
# MAGIC   line-height: 1.6;
# MAGIC }
# MAGIC
# MAGIC .e2d-steps-block ol {
# MAGIC   margin: 8px 0 0 0;
# MAGIC   padding-left: 20px;
# MAGIC }
# MAGIC
# MAGIC .e2d-steps-block li {
# MAGIC   margin-bottom: 8px;
# MAGIC }
# MAGIC
# MAGIC .e2d-steps-block ul {
# MAGIC   margin: 4px 0 8px 20px;
# MAGIC   list-style: disc;
# MAGIC }
# MAGIC
# MAGIC .e2d-info-block {
# MAGIC   border-left: 4px solid;
# MAGIC   border-radius: 0 6px 6px 0;
# MAGIC   padding: 12px 14px;
# MAGIC   font-size: 14pt;
# MAGIC   line-height: 1.5;
# MAGIC   font-weight: 500;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <div style="background: #F8F9FC; border: 3px solid #02A36F; border-radius: 10px; padding: 24px; box-sizing: border-box;">
# MAGIC
# MAGIC   <div class="e2d-row">
# MAGIC     <span class="e2d-box active" data-id="0" onclick="e2dSelect(0)" style="--pc:#02A36F; border-top-color:#02A36F;">
# MAGIC       <span class="e2d-title">DABs UI</span>
# MAGIC       <span class="e2d-subtitle">Git folder</span>
# MAGIC     </span><span class="e2d-box" data-id="1" onclick="e2dSelect(1)" style="--pc:#FF5F46; border-top-color:#FF5F46;">
# MAGIC       <span class="e2d-title">Apps UI</span>
# MAGIC       <span class="e2d-subtitle">No Git folder</span>
# MAGIC     </span>
# MAGIC   </div>
# MAGIC
# MAGIC   <div class="e2d-detail-wrap" id="e2d-detail-wrap">
# MAGIC     <div class="e2d-detail-card" id="e2d-detail-card"></div>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var E2D_CARDS = [
# MAGIC   {
# MAGIC     title: "DABs UI (Git folder)",
# MAGIC     color: "#02A36F",
# MAGIC     bg: "rgba(2,163,111,0.10)",
# MAGIC     infoBg: "rgba(2,163,111,0.10)",
# MAGIC     content: `
# MAGIC       <ol>
# MAGIC         <li>Click the rocket icon on any file in the project folder.</li>
# MAGIC         <li>Click <strong>Deploy</strong>, wait for validation, then click <strong>Deploy</strong> again.</li>
# MAGIC         <li>Navigate to the app and click <strong>Deploy</strong> to push the updated source.</li>
# MAGIC         <li>Notice the new <strong>UC function</strong> resource now appears under <strong>App resources</strong>.</li>
# MAGIC         <li>After deployment, click on the URL to open the app.</li>
# MAGIC       </ol>
# MAGIC     `,
# MAGIC     info: '<strong>Requires a Git folder.</strong> The DABs UI detects the new UC function resource in <code>databricks.yml</code> and grants <code>EXECUTE</code> permission automatically. Wait for the app status to return to <strong>Running</strong>.'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "Apps UI (no Git folder)",
# MAGIC     color: "#FF5F46",
# MAGIC     bg: "rgba(255,95,70,0.10)",
# MAGIC     infoBg: "rgba(255,95,70,0.10)",
# MAGIC     content: `
# MAGIC       <ol>
# MAGIC         <li>Navigate to the app Overview page and click <strong>(⋮)</strong> icon then on  <strong>Edit</strong> at the top right.</li>
# MAGIC         <li>Click <strong>resources</strong> at the top of the setup wizard.</li>
# MAGIC         <li>Click <strong>+ Add resource</strong> and select <strong>UC function</strong>.</li>
# MAGIC         <li>Enter the full function name: <code>{catalog}.agent_deployment.avg_neigh_price</code> (replace <code>{catalog}</code> with your catalog name).</li>
# MAGIC         <li> Click <strong>Confirm</strong>.</li>
# MAGIC         <li>Make sure <strong>Permission</strong> is set to <strong>Can execute</strong> and <strong>Resource key</strong> is set to <strong>uc-function-1</strong>.</li>
# MAGIC         <li>Click <strong>Save</strong>.</li>
# MAGIC         <li>Back on the Overview page, click <strong>Deploy</strong> to push the updated source.</li>
# MAGIC       </ol>
# MAGIC     `,
# MAGIC     info: '<strong>No Git folder required.</strong> You add the UC function resource manually on the Overview page. Wait for the app status to return to <strong>Running</strong>.'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var e2dCurrent = null;
# MAGIC
# MAGIC function e2dSelect(id) {
# MAGIC   var wrap = document.getElementById('e2d-detail-wrap');
# MAGIC   var card = document.getElementById('e2d-detail-card');
# MAGIC   var c = E2D_CARDS[id];
# MAGIC
# MAGIC   document.querySelectorAll('.e2d-box').forEach(function(b) {
# MAGIC     var isActive = parseInt(b.dataset.id, 10) === id;
# MAGIC     b.classList.toggle('active', isActive);
# MAGIC   });
# MAGIC
# MAGIC   if (e2dCurrent === id) {
# MAGIC     wrap.classList.remove('open');
# MAGIC     document.querySelectorAll('.e2d-box').forEach(function(b) {
# MAGIC       b.classList.remove('active');
# MAGIC     });
# MAGIC     e2dCurrent = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   e2dCurrent = id;
# MAGIC
# MAGIC   card.style.borderTopColor = c.color;
# MAGIC   card.innerHTML =
# MAGIC     '<div style="font-size:18pt;font-weight:700;margin-bottom:12px;color:#0b2026;">' + c.title + '</div>'
# MAGIC     + '<div class="e2d-steps-block" style="background:' + c.bg + ';border-color:' + c.color + ';">'
# MAGIC     + c.content
# MAGIC     + '</div>'
# MAGIC     + '<div class="e2d-info-block" style="background:' + c.infoBg + ';border-color:' + c.color + ';">'
# MAGIC     + c.info
# MAGIC     + '</div>';
# MAGIC
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC
# MAGIC e2dSelect(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E3. Test and Verify Tool Call Traces
# MAGIC
# MAGIC The agent now has access to the `avg_neigh_price` UC function through MCP. Verify that tool calls appear in the traces.
# MAGIC
# MAGIC 1. Open the app URL and send a query that triggers the tool, such as: *"What is the average price for listings in the Mission neighborhood?"*
# MAGIC 2. Wait for the agent to respond. The response should include actual data from the UC function.
# MAGIC 3. Navigate to the **Apps** page in the workspace sidebar and select your app.
# MAGIC 4. Under the **App resources** section, find the **MLflow Experiment ID** and click on it to open the experiment.
# MAGIC
# MAGIC 5. Click the **Traces** tab and open the most recent trace. You should now see:
# MAGIC    - The **root span** (`tool_calling_chatbot_run`) capturing the full agent invocation
# MAGIC    - **Child spans** from the Agents SDK runner showing each reasoning step
# MAGIC    - **MCP tool spans** for `tools/list` (capability discovery) and `tools/call` (the actual function invocation). Click a `tools/call` span to inspect the arguments passed to `avg_neigh_price` and the value it returned.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Compare this trace to the one from Section D3. The addition of MCP tools added new child spans without any changes to the tracing code itself. The <code>@mlflow.trace</code> decorator and the Agents SDK automatically capture MCP interactions as part of the trace tree.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## F. Verify Traces in MLflow UI
# MAGIC
# MAGIC Once the app is running, send a few messages through the chat interface to generate traces. Try a message that triggers a tool call, such as: *"What is the average price for listings in the Mission neighborhood?"*
# MAGIC
# MAGIC 1. In your Databricks workspace, navigate to **Experiments** and open the experiment named **simple-chatbot-experiment** (under your user path).
# MAGIC 2. Click the **Traces** tab. You should see a trace entry for each message you sent.
# MAGIC 3. Click a trace to expand it. You will see:
# MAGIC    - The **root span** (`tool_calling_chatbot_run`) capturing the full agent invocation: inputs, final output, and total latency
# MAGIC    - **Child spans** from the Agents SDK runner showing each step: the initial completion, tool call decision, MCP round-trip, and final synthesis
# MAGIC    - **MCP tool spans** showing the arguments sent to the UC function and the value it returned
# MAGIC 4. Back on the experiment landing page, click the **Evaluation runs** tab. Each invocation appears as a nested run with logged parameters (`agent_name`, `model_endpoint`, `num_messages`) and artifacts (`inputs/messages.json`, `outputs/final_output.txt`).
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Traces vs. Runs</strong>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Traces</strong> capture the execution graph of a single request (spans, timing, inputs, and outputs) and are best for debugging individual interactions.</li>
# MAGIC <li><strong>Runs</strong> capture experiment-level metadata (parameters, metrics, and artifacts) and are best for comparing behavior across many interactions over time.</li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## G. Clean Up
# MAGIC
# MAGIC When you are finished with this demo, remove the resources created during deployment.
# MAGIC
# MAGIC 1. Navigate to **Apps** under **Compute** and select the three vertical dots next to your app. Select **Delete** to stop and remove the app.
# MAGIC 2. Navigate to **Experiments**, open the experiment, and delete it if you no longer need the trace and run history.

# COMMAND ----------

# MAGIC %md
# MAGIC ## H. Conclusion
# MAGIC
# MAGIC In this demo, you progressively built a production-ready agent deployed as a Databricks App:
# MAGIC
# MAGIC 1. **Deployed a basic agent** using the OpenAI Agents SDK, FastAPI, and Declarative Automation Bundles
# MAGIC 2. **Added MLflow tracing** with `@mlflow.trace` and `mlflow.start_run` to capture per-request execution spans, parameters, and artifacts
# MAGIC 3. **Integrated MCP tools** by connecting a Unity Catalog function through a Managed MCP server, making real data lookups available to the agent
# MAGIC 4. **Verified traces** in the MLflow UI, inspecting root spans, Agents SDK child spans, and MCP tool call spans

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC
# MAGIC