# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab - Building Agents with DSPy on Databricks
# MAGIC #### Duration: 30 minutes
# MAGIC ## Overview
# MAGIC In this lab, you will build a tool-calling agent with **DSPy**, a framework where you declare *what* a language model should do with **signatures** and let **modules** decide *how* to prompt it. You will connect DSPy to a Databricks model, prototype one tool as a plain Python function, wrap a Unity Catalog function as a second tool, and combine them in a `dspy.ReAct` agent that you trace with MLflow. This notebook acts as a primer for the DSPy framework and is by no means comprehensive. 
# MAGIC
# MAGIC Each step has a **TODO** with `<FILL_IN>` placeholders. The instructions above each TODO tell you what to fill in, and you can reveal the answer if you get stuck.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lab, you will be able to:
# MAGIC 1. Configure DSPy to use a Databricks model serving endpoint
# MAGIC 2. Define DSPy signatures, in shorthand and as classes
# MAGIC 3. Prototype agent tools as plain Python functions, and wrap a Unity Catalog function as a DSPy tool
# MAGIC 4. Build a `dspy.ReAct` agent that reasons and calls tools
# MAGIC 5. Inspect an agent's trajectory and its MLflow trace

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
# MAGIC <li><strong>Serverless Compute, Version 6</strong>: How to select an environment version (<a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V6</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Classroom Setup
# MAGIC
# MAGIC Run the next cell to install the libraries for this lab, set up your catalog and schema, load the Airbnb dataset, and register the `avg_neigh_price` UC function. It also creates these variables: `catalog_name`, `schema_name`, `username`, and `mlflow_experiment_path`.

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-8

# COMMAND ----------

# MAGIC %md
# MAGIC Turn on MLflow tracing for DSPy. Every DSPy call after this cell is traced, which you will use to see how DSPy builds prompts and calls tools.

# COMMAND ----------

import mlflow

mlflow.dspy.autolog()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. Lab Scenario
# MAGIC
# MAGIC <div class="scn8"><style>.scn8{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#0b2026;width:100%;max-width:100%;box-sizing:border-box;margin:0;padding:8px}.scn8 .scen{background:#FFFFFF;color:#0b2026;padding:26px 30px;border-radius:12px;box-sizing:border-box;overflow:hidden;box-shadow:0 2px 16px rgba(27,49,57,0.10),0 1px 4px rgba(27,49,57,0.06);border:1px solid rgba(27,49,57,0.08)}.scn8 .kicker{font-size:14px;text-transform:uppercase;letter-spacing:1.2px;color:#5A6F77;margin-bottom:10px;font-weight:700;display:flex;align-items:center;gap:10px}.scn8 .body{font-size:15px;line-height:1.7;color:#0b2026}.scn8 .body p{margin:0 0 14px}.scn8 .body p:last-child{margin:0}.scn8 .tools{background:#F9F7F4;border:1px solid rgba(27,49,57,0.10);border-radius:8px;padding:12px 16px;margin:0 0 14px 0;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:14px;line-height:1.9}.scn8 .fn{color:#0B8F5F}.scn8 .arg{color:#5A6F77}.scn8 .dsc{color:#5A6F77;opacity:.8}.scn8 code{background:rgba(27,49,57,0.07);padding:2px 6px;border-radius:4px;font-size:14px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#0b2026}.scn8 .eq{background:rgba(0,169,114,.10);border-left:4px solid #00A972;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0 0;font-size:14px;font-style:italic;line-height:1.6;color:#0b2026}</style><div class="scen">
# MAGIC <div class="kicker"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#5A6F77" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect x="4" y="8" width="16" height="12" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M9 13v2"/><path d="M15 13v2"/></svg>Scenario</div>
# MAGIC <div class="body">
# MAGIC <p>You are building an <strong>Airbnb recommendation assistant</strong> for San Francisco. A user should be able to ask about a specific listing, compare it against neighborhood averages, and get a concrete recommendation &#x2014; all in one turn. The assistant needs to <em>reason</em> about which tools to call and in what order, then synthesize the results into a single answer.</p>
# MAGIC
# MAGIC <p>You will wire DSPy to a Databricks-hosted model, define two tools, and combine them in a <code>dspy.ReAct</code> agent that alternates between thinking and acting until it can answer:</p>
# MAGIC
# MAGIC <div class="tools"><div><span class="fn">get_listing_details</span><span class="arg">(listing_id)</span> <span class="dsc">&#x2014; fetches structured details for one listing from a Spark table</span></div><div><span class="fn">avg_neigh_price</span><span class="arg">(neighborhood)</span> <span class="dsc">&#x2014; calls a Unity Catalog function that returns the neighborhood&#x2019;s average nightly price</span></div></div>
# MAGIC
# MAGIC <p>The first tool is a plain Python function &#x2014; DSPy has no tool decorator, so any function can be a tool. The second wraps a UC function using a parameterized query, keeping the logic governed in Unity Catalog while DSPy sees an ordinary callable. By the end, the agent will handle questions like:</p>
# MAGIC
# MAGIC <div class="eq">"Tell me about listing 958. Is it more expensive than the average listing in the Mission?"</div>
# MAGIC </div>
# MAGIC </div></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Configure DSPy

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Connect DSPy to a Model Service via Unity Gateway
# MAGIC
# MAGIC The first step in getting started with DSPy is to create a language model client. `dspy.LM` creates the client, and there are two ways to bring in a model:
# MAGIC - `databricks/<llm_endpoint_name>`: calls a Databricks-provided model service from the `system.ai` catalog (available to all account users by default).
# MAGIC - `databricks/<catalog.schema.model_service>`: calls a custom model service registered elsewhere in Unity Catalog (with configurable routing, fallbacks, and access controls).
# MAGIC
# MAGIC Authentication is handled via your Databricks credentials (in our case by defining a notebook-scoped token for demonstration purposes), and authorization is governed by Unity Catalog permissions after we have resolved the Workspace URL. `dspy.configure` then sets it as the default model for every DSPy module.
# MAGIC
# MAGIC **TODO:** Fill out the fully-qualified name for bringing in the model service in your schema `agent_training` (inspect the name using the **Catalog** button to the left) and set the default model for all DSPy modules that will be used in this notebook.

# COMMAND ----------

import os
import dspy

token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().apiToken().get()
)

host = spark.conf.get("spark.databricks.workspaceUrl")

model_service = "<FILL_IN>"

lm = dspy.LM(
    model=model_service,  # fully qualified UC name
    api_base=f"{host}/ai-gateway/mlflow/v1",
    api_key=token
)

dspy.configure(lm=<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import os
# MAGIC import dspy
# MAGIC
# MAGIC token = (
# MAGIC     dbutils.notebook.entry_point.getDbutils()
# MAGIC     .notebook().getContext().apiToken().get()
# MAGIC )
# MAGIC
# MAGIC host = spark.conf.get("spark.databricks.workspaceUrl")
# MAGIC
# MAGIC model_service = f"{catalog_name}.{schema_name}.agent-training-model"
# MAGIC
# MAGIC lm = dspy.LM(
# MAGIC     model=model_service,  # fully qualified UC name
# MAGIC     api_base=f"{host}/ai-gateway/mlflow/v1",
# MAGIC     api_key=token
# MAGIC )
# MAGIC
# MAGIC dspy.configure(lm=lm)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Signatures and Modules
# MAGIC
# MAGIC A [**signature**](https://dspy.ai/current/api/signatures/) declares a task's inputs and outputs, and a [**module**](https://dspy.ai/current/api/modules/) decides how to prompt the model for it. The shorthand signature below has one input, `question`, and one output, `answer`. `dspy.Predict` is the simplest module.
# MAGIC
# MAGIC You never write the prompt yourself. Inside every module, an [**adapter**](https://dspy.ai/current/api/adapters/) writes the prompt from your signature, sends it to the model, and parses the reply back into your output fields. The adapter pattern allows DSPy to work with different LM interfaces while maintaining a consistent programming model for users. The diagram follows one call through `dspy.Predict`:
# MAGIC
# MAGIC <div class="dsp">
# MAGIC <style>
# MAGIC   .dsp{background:#ffffff;width:100%;max-width:none;margin:0;padding:8px 0 16px;box-sizing:border-box;}
# MAGIC   .dsp svg{display:block;width:100%;height:auto;}
# MAGIC   .dsp svg text{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;fill:#0b2026;}
# MAGIC   .dsp .h   {font-weight:800;}
# MAGIC   .dsp .b   {font-weight:700;}
# MAGIC   .dsp .m   {fill:#618794;font-weight:500;}
# MAGIC   .dsp .mono,.dsp .mono text{white-space:pre;font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-ligatures:none;font-feature-settings:"liga" 0,"calt" 0;}
# MAGIC </style>
# MAGIC <svg viewBox="0 0 1600 840" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Inside dspy.Predict: the Adapter round trip, following one example. The Signature 'question -> answer' promises an answer for a question. 1, you call qa(question='What is the capital of France?'). 2, the Adapter's format step writes a system message listing the input and output fields and a user message with the question under a [[ ## question ## ]] marker. 3, the LM only sees those chat messages and replies with plain text. 4, the Adapter's parse step finds the [[ ## answer ## ]] marker in the reply and extracts Paris. 5, you get back Prediction(answer='Paris'). ChatAdapter is the default; JSONAdapter or TwoStepAdapter can be swapped in with dspy.configure(adapter=...) without changing the program.">
# MAGIC   <defs>
# MAGIC     <marker id="ar-b" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4299E0"/></marker>
# MAGIC     <marker id="ar-g" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#00A972"/></marker>
# MAGIC     <marker id="ar-s" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#C4CCD6"/></marker>
# MAGIC     <filter id="sh" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="4" stdDeviation="9" flood-color="#1B3139" flood-opacity="0.10"/></filter>
# MAGIC     <filter id="glow" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="0" dy="6" stdDeviation="12" flood-color="#1B5162" flood-opacity="0.30"/></filter>
# MAGIC   </defs>
# MAGIC   <!-- ================= the signature, drawn as a promise ================= -->
# MAGIC   <path d="M280.0,85 L1312.0,85" fill="none" stroke="#C4CCD6" stroke-width="2.5" marker-end="url(#ar-s)"/>
# MAGIC   <rect id="pill-q" x="92.0" y="62" width="184" height="46" rx="23" fill="#4299E0"/>
# MAGIC   <text class="mono b" x="184.0" y="92" font-size="19" text-anchor="middle" style="fill:#ffffff" data-box="pill-q">question</text>
# MAGIC   <rect id="pill-a" x="1324.0" y="62" width="184" height="46" rx="23" fill="#00A972"/>
# MAGIC   <text class="mono b" x="1416.0" y="92" font-size="19" text-anchor="middle" style="fill:#ffffff" data-box="pill-a">answer</text>
# MAGIC   <rect id="sig" x="560" y="66" width="480" height="38" rx="19" fill="#ffffff"/>
# MAGIC   <text x="800" y="92" font-size="19" text-anchor="middle" data-box="sig"><tspan class="b">Signature</tspan><tspan class="m">  </tspan><tspan class="mono m" font-size="17">"question -&gt; answer"</tspan></text>
# MAGIC   <!-- guides from the promise down to where it's kept -->
# MAGIC   <g fill="none" stroke="#C4CCD6" stroke-width="2" stroke-dasharray="6 6">
# MAGIC     <path d="M184.0,110 L184.0,318"/>
# MAGIC     <path d="M1416.0,110 L1416.0,329"/>
# MAGIC   </g>
# MAGIC   <!-- ================= the adapter arc: one object, both sides ================= -->
# MAGIC   <path d="M532.0,250 C532.0,184 612.0,171 800,171 C988.0,171 1068.0,184 1068.0,250" fill="none" stroke="#1B5162" stroke-width="3"/>
# MAGIC   <rect id="adp" x="684" y="148" width="232" height="46" rx="23" fill="#1B5162"/>
# MAGIC   <text x="800" y="178" font-size="21" text-anchor="middle" style="fill:#ffffff" data-box="adp" font-weight="800">Adapter</text>
# MAGIC   <text class="m" x="800" y="228" font-size="17" text-anchor="middle">ChatAdapter by default</text>
# MAGIC   <!-- ================= 1 · your call ================= -->
# MAGIC   <rect id="call" x="28" y="320" width="312" height="204" rx="12" fill="#303F47" filter="url(#sh)"/>
# MAGIC   <text x="50" y="354" font-size="16" style="fill:#90A5B1" data-box="call" font-weight="700" letter-spacing="1.4">YOUR CALL</text>
# MAGIC   <text class="mono" x="50" y="390" font-size="17" style="fill:#C4CCD6" data-box="call">qa = dspy.Predict(</text>
# MAGIC   <text class="mono" x="50" y="414" font-size="17" style="fill:#C4CCD6" data-box="call">  <tspan style="fill:#FFCC66">"question -&gt; answer"</tspan>)</text>
# MAGIC   <text class="mono" x="50" y="452" font-size="17" style="fill:#C4CCD6" data-box="call">qa(question=</text>
# MAGIC   <text class="mono" x="50" y="476" font-size="17" style="fill:#FFCC66" data-box="call">  "What is the capital</text>
# MAGIC   <text class="mono" x="50" y="500" font-size="17" style="fill:#C4CCD6" data-box="call"><tspan style="fill:#FFCC66">   of France?"</tspan>)</text>
# MAGIC   <path d="M344,422 L368,422" fill="none" stroke="#4299E0" stroke-width="2.5" marker-end="url(#ar-b)"/>
# MAGIC   <!-- ================= 2 · format() ================= -->
# MAGIC   <path id="fmt" d="M376,250 H678 A10,10 0 0 1 688,260 V584 A10,10 0 0 1 678,594 H376 Z" fill="#F9F7F4" stroke="#EEEDE9" filter="url(#sh)"/>
# MAGIC   <rect x="376" y="250" width="5" height="344" fill="#1B5162"/>
# MAGIC   <text class="mono b" x="396" y="292" font-size="26" data-box="fmt">format()</text>
# MAGIC   <text class="m" x="396" y="318" font-size="17" data-box="fmt">signature + demos + inputs</text>
# MAGIC   <rect id="sys" x="396" y="332" width="278" height="128" rx="8" fill="#ffffff" stroke="#DCE0E2"/>
# MAGIC   <text x="410" y="356" font-size="16" style="fill:#618794" data-box="sys" font-weight="700" letter-spacing="1">SYSTEM</text>
# MAGIC   <text class="mono" x="410" y="380" font-size="17" data-box="sys">Your input fields are:</text>
# MAGIC   <text class="mono" x="410" y="402" font-size="17" data-box="sys">1. <tspan style="fill:#4299E0">`question`</tspan> (str)</text>
# MAGIC   <text class="mono" x="410" y="424" font-size="17" data-box="sys">Your output fields are:</text>
# MAGIC   <text class="mono" x="410" y="446" font-size="17" data-box="sys">1. <tspan style="fill:#00A972">`answer`</tspan> (str) …</text>
# MAGIC   <rect id="usr" x="396" y="472" width="278" height="106" rx="8" fill="#ffffff" stroke="#DCE0E2"/>
# MAGIC   <text x="410" y="496" font-size="16" style="fill:#618794" data-box="usr" font-weight="700" letter-spacing="1">USER</text>
# MAGIC   <text class="mono" x="410" y="520" font-size="17" style="fill:#4299E0" data-box="usr">[[ ## question ## ]]</text>
# MAGIC   <text class="mono" x="410" y="542" font-size="17" data-box="usr">What is the capital of</text>
# MAGIC   <text class="mono m" x="410" y="564" font-size="17" data-box="usr">France? Respond with …</text>
# MAGIC   <path d="M692,422 L728,422" fill="none" stroke="#4299E0" stroke-width="2.5" marker-end="url(#ar-b)"/>
# MAGIC   <!-- ================= 3 · the LM ================= -->
# MAGIC   <text class="mono b" x="800" y="334" font-size="19" text-anchor="middle">LM.__call__</text>
# MAGIC   <circle cx="800" cy="422" r="64" fill="#1B5162" filter="url(#glow)"/>
# MAGIC   <g stroke="#ffffff" stroke-width="3.2" fill="#ffffff">
# MAGIC     <line x1="782" y1="407" x2="816" y2="401"/>
# MAGIC     <line x1="782" y1="407" x2="820" y2="438"/>
# MAGIC     <line x1="816" y1="401" x2="820" y2="438"/>
# MAGIC     <line x1="782" y1="407" x2="786" y2="442"/>
# MAGIC     <line x1="820" y1="438" x2="786" y2="442"/>
# MAGIC     <circle cx="782" cy="407" r="6"/>
# MAGIC     <circle cx="816" cy="401" r="6"/>
# MAGIC     <circle cx="820" cy="438" r="6"/>
# MAGIC     <circle cx="786" cy="442" r="6"/>
# MAGIC   </g>
# MAGIC   <text class="m" x="800" y="520" font-size="17" text-anchor="middle">chat messages in,</text>
# MAGIC   <text class="m" x="800" y="543" font-size="17" text-anchor="middle">plain text out</text>
# MAGIC   <path d="M868,422 L904,422" fill="none" stroke="#00A972" stroke-width="2.5" marker-end="url(#ar-g)"/>
# MAGIC   <!-- ================= 4 · parse() ================= -->
# MAGIC   <path id="prs" d="M912,250 H1214 A10,10 0 0 1 1224,260 V584 A10,10 0 0 1 1214,594 H912 Z" fill="#F9F7F4" stroke="#EEEDE9" filter="url(#sh)"/>
# MAGIC   <rect x="912" y="250" width="5" height="344" fill="#1B5162"/>
# MAGIC   <!-- adapter stem-end dots: centered on the card top edge so the border bisects each circle -->
# MAGIC   <circle cx="532.0" cy="250" r="6" fill="#1B5162"/>
# MAGIC   <circle cx="1068.0" cy="250" r="6" fill="#1B5162"/>
# MAGIC   <text class="mono b" x="932" y="292" font-size="26" data-box="prs">parse()</text>
# MAGIC   <text class="m" x="932" y="318" font-size="17" data-box="prs">reply → typed output fields</text>
# MAGIC   <rect id="rep" x="932" y="332" width="278" height="126" rx="8" fill="#ffffff" stroke="#DCE0E2"/>
# MAGIC   <text x="946" y="356" font-size="16" style="fill:#618794" data-box="rep" font-weight="700" letter-spacing="1">LM REPLY</text>
# MAGIC   <text class="mono" x="946" y="382" font-size="17" style="fill:#00A972" data-box="rep">[[ ## answer ## ]]</text>
# MAGIC   <rect id="paris" x="940" y="389" width="72" height="35" rx="6" fill="rgba(0,169,114,0.14)"/>
# MAGIC   <text class="mono b" x="946" y="412" font-size="17" data-box="paris">Paris</text>
# MAGIC   <text class="mono m" x="946" y="444" font-size="17" data-box="rep">[[ ## completed ## ]]</text>
# MAGIC   <rect id="ans" x="932" y="474" width="278" height="46" rx="8" fill="#00A972"/>
# MAGIC   <text class="mono b" x="1071.0" y="503" font-size="17" text-anchor="middle" style="fill:#ffffff" data-box="ans">answer = "Paris"</text>
# MAGIC   <text class="m" x="1071.0" y="556" font-size="17" text-anchor="middle" data-box="prs">checked against the output type</text>
# MAGIC   <path d="M1228,422 L1252,422" fill="none" stroke="#00A972" stroke-width="2.5" marker-end="url(#ar-g)"/>
# MAGIC   <!-- ================= 5 · what you get back ================= -->
# MAGIC   <path id="res" d="M1260,331 H1562 A10,10 0 0 1 1572,341 V503 A10,10 0 0 1 1562,513 H1260 Z" fill="#ffffff" stroke="#DCE0E2" stroke-width="1.5" filter="url(#sh)"/>
# MAGIC   <rect x="1260" y="331" width="5" height="182" fill="#00A972"/>
# MAGIC   <text x="1282" y="365" font-size="16" style="fill:#00A972" data-box="res" font-weight="700" letter-spacing="1.4">YOU GET BACK</text>
# MAGIC   <text class="mono" x="1282" y="403" font-size="17" data-box="res">Prediction(</text>
# MAGIC   <text class="mono" x="1282" y="429" font-size="17" data-box="res">  answer=<tspan class="b" style="fill:#00A972">"Paris"</tspan>)</text>
# MAGIC   <line x1="1282" y1="453" x2="1550" y2="453" stroke="#EEEDE9" stroke-width="1.5"/>
# MAGIC   <text class="mono m" x="1282" y="485" font-size="17" data-box="res">pred.answer → "Paris"</text>
# MAGIC   <!-- ================= step captions ================= -->
# MAGIC   <g font-size="18" text-anchor="middle">
# MAGIC     <circle cx="184.0" cy="648" r="16" fill="#4299E0"/><text x="184.0" y="654" font-size="18" font-weight="700" style="fill:#ffffff">1</text>
# MAGIC     <text class="m" x="184.0" y="692" data-cap="1">Call with keyword args</text>
# MAGIC     <text class="m" x="184.0" y="716" data-cap="1">named by the signature</text>
# MAGIC     <circle cx="532.0" cy="648" r="16" fill="#1B5162"/><text x="532.0" y="654" font-size="18" font-weight="700" style="fill:#ffffff">2</text>
# MAGIC     <text class="m" x="532.0" y="692" data-cap="2">format() writes the prompt,</text>
# MAGIC     <text class="m" x="532.0" y="716" data-cap="2">marking each field by name</text>
# MAGIC     <circle cx="800" cy="648" r="16" fill="#1B5162"/><text x="800" y="654" font-size="18" font-weight="700" style="fill:#ffffff">3</text>
# MAGIC     <text class="m" x="800" y="692" data-cap="3">The LM never sees</text>
# MAGIC     <text class="m" x="800" y="716" data-cap="3">your Python types</text>
# MAGIC     <circle cx="1068.0" cy="648" r="16" fill="#1B5162"/><text x="1068.0" y="654" font-size="18" font-weight="700" style="fill:#ffffff">4</text>
# MAGIC     <text class="m" x="1068.0" y="692" data-cap="4">parse() finds the markers</text>
# MAGIC     <text class="m" x="1068.0" y="716" data-cap="4">and pulls out each field</text>
# MAGIC     <circle cx="1416.0" cy="648" r="16" fill="#00A972"/><text x="1416.0" y="654" font-size="18" font-weight="700" style="fill:#ffffff">5</text>
# MAGIC     <text class="m" x="1416.0" y="692" data-cap="5">Typed fields come back</text>
# MAGIC     <text class="m" x="1416.0" y="716" data-cap="5">as a Prediction</text>
# MAGIC   </g>
# MAGIC   <!-- ================= swap strip ================= -->
# MAGIC   <line x1="28" y1="748" x2="1572" y2="748" stroke="#EEEDE9" stroke-width="1.5"/>
# MAGIC   <text class="b" x="28" y="800" font-size="20" data-row="swap">Swap the adapter, keep the program</text>
# MAGIC   <g font-size="17">
# MAGIC     <rect id="p1" x="412" y="772" width="248" height="44" rx="22" fill="#1B5162" data-row="swap"/>
# MAGIC     <text x="536" y="800" text-anchor="middle" style="fill:#ffffff" data-box="p1"><tspan class="mono b">ChatAdapter</tspan> · default</text>
# MAGIC     <rect id="p2" x="672" y="772" width="160" height="44" rx="22" fill="#ffffff" stroke="#1B5162" stroke-width="1.5" data-row="swap"/>
# MAGIC     <text class="mono b" x="752" y="800" text-anchor="middle" data-box="p2">JSONAdapter</text>
# MAGIC     <rect id="p3" x="844" y="772" width="190" height="44" rx="22" fill="#ffffff" stroke="#1B5162" stroke-width="1.5" data-row="swap"/>
# MAGIC     <text class="mono b" x="939" y="800" text-anchor="middle" data-box="p3">TwoStepAdapter</text>
# MAGIC   </g>
# MAGIC   <text class="mono m" x="1572" y="800" font-size="17" text-anchor="end" data-row="swap">dspy.configure(adapter=JSONAdapter())</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC Run the next cell, then open the trace below the output. You will see the same round trip: `ChatAdapter.format`, then `LM.__call__`, then the parsed `answer`.

# COMMAND ----------

signature = "question -> answer: str"
question = "4 dice are tossed. What is the probability that the sum of the dice equals 2 or 4?"

qa = dspy.Predict(signature)
qa(question=question)

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO:** Run the same `signature` through `dspy.ChainOfThought`, which asks the model to reason step by step before answering. Compare the two traces: Chain of Thought adds a `reasoning` output field and an extra `Predict.forward` step, but the adapter round trip is the same.

# COMMAND ----------

qa_cot = dspy.<FILL_IN>(<FILL_IN>)
qa_cot(question=question)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC qa_cot = dspy.ChainOfThought(signature)
# MAGIC qa_cot(question=question)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Prepare the Data
# MAGIC
# MAGIC We will create an agent that answers questions about the San Francisco Airbnb listings created by our classroom setup script. Run the next cell to preview the table.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM sf_airbnb_listings LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC The next cell defines a [Pydantic model](https://pydantic.dev/), `ListingInfo`, that describes a listing. Every field on a Signature is a Pydantic `FieldInfo` and the base `Signature` class extends `pydantic.BaseModel`. Your first tool returns it, so the agent receives structured, typed fields instead of a raw row.

# COMMAND ----------

from typing import Optional
from pydantic import BaseModel

class ListingInfo(BaseModel):
    """Key details about an Airbnb listing."""
    id: int
    name: str
    neighbourhood: str
    neighbourhood_cleansed: str
    summary: Optional[str]
    price: float
    room_type: str
    listing_source_information: Optional[str]

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Define the Agent's Tools

# COMMAND ----------

# MAGIC %md
# MAGIC ### E1. Prototype a Tool as a Plain Function
# MAGIC
# MAGIC DSPy has no tool decorator: any Python function can be a tool. Instead, we can pass the tool directly to the agent without any additional configuration.
# MAGIC
# MAGIC **TODO:** Finish `get_listing_details` by adding:
# MAGIC - a type hint for `listing_id` (an integer) and the return type (`ListingInfo`)
# MAGIC - a docstring that says what the tool returns and when to use it
# MAGIC
# MAGIC The body is provided.

# COMMAND ----------

def get_listing_details(listing_id: <FILL_IN>) -> <FILL_IN>:
    """<FILL_IN>"""
    rows = spark.table("sf_airbnb_listings").filter(f"id = {int(listing_id)}").limit(1).collect()
    if not rows:
        raise ValueError(f"Listing with ID {listing_id} not found.")
    r = rows[0]
    return ListingInfo(
        id=r.id,
        name=r.name,
        neighbourhood=r.neighbourhood,
        neighbourhood_cleansed=r.neighbourhood_cleansed,
        summary=r.summary,
        price=r.price,
        room_type=r.room_type,
        listing_source_information=r.listing_source_information,
    )

get_listing_details(958)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC def get_listing_details(listing_id: int) -> ListingInfo:
# MAGIC     """Get details for one San Francisco Airbnb listing by its ID, including its
# MAGIC     neighborhood, room type, nightly price in USD, and summary.
# MAGIC
# MAGIC     Use this when the user asks about a specific listing ID."""
# MAGIC     rows = spark.table("sf_airbnb_listings").filter(f"id = {int(listing_id)}").limit(1).collect()
# MAGIC     if not rows:
# MAGIC         raise ValueError(f"Listing with ID {listing_id} not found.")
# MAGIC     r = rows[0]
# MAGIC     return ListingInfo(
# MAGIC         id=r.id,
# MAGIC         name=r.name,
# MAGIC         neighbourhood=r.neighbourhood,
# MAGIC         neighbourhood_cleansed=r.neighbourhood_cleansed,
# MAGIC         summary=r.summary,
# MAGIC         price=r.price,
# MAGIC         room_type=r.room_type,
# MAGIC         listing_source_information=r.listing_source_information,
# MAGIC     )
# MAGIC
# MAGIC get_listing_details(958)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Wrap a Unity Catalog Function as a Tool
# MAGIC
# MAGIC Setup registered `avg_neigh_price` to Unity Catalog. To give it to a DSPy agent, wrap a call to it in a plain function. The logic stays governed in Unity Catalog, and DSPy sees an ordinary tool.
# MAGIC
# MAGIC The neighborhood comes from the model, so never paste it into the SQL text. Use a **named parameter marker** such as `:neighborhood` and pass the value in `args`. Spark then treats it as data, so a value with a quote in it cannot break, or change, the query.
# MAGIC
# MAGIC **TODO:** Fill in the parameter marker and the `args` dictionary.

# COMMAND ----------

def avg_neigh_price(neighborhood: str) -> Optional[float]:
    """Get the average nightly price in USD of San Francisco Airbnb listings in a
    neighborhood, such as 'Mission' or 'Noe Valley'."""
    result = spark.sql(
        "SELECT avg_neigh_price(<FILL_IN>) AS average_price",
        args=<FILL_IN>,
    ).collect()
    return result[0]["average_price"] if result else None

avg_neigh_price("Mission")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC def avg_neigh_price(neighborhood: str) -> Optional[float]:
# MAGIC     """Get the average nightly price in USD of San Francisco Airbnb listings in a
# MAGIC     neighborhood, such as 'Mission' or 'Noe Valley'."""
# MAGIC     result = spark.sql(
# MAGIC         "SELECT avg_neigh_price(:neighborhood) AS average_price",
# MAGIC         args={"neighborhood": neighborhood},
# MAGIC     ).collect()
# MAGIC     return result[0]["average_price"] if result else None
# MAGIC
# MAGIC avg_neigh_price("Mission")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Build the ReAct Agent

# COMMAND ----------

# MAGIC %md
# MAGIC ### F1. Define the Agent's Signature
# MAGIC
# MAGIC For an agent, write the signature as a class. The docstring becomes the agent's instructions, and each field gets a description the model reads.
# MAGIC
# MAGIC **TODO:** Declare `user_request` as an input field and `response` as an output field.

# COMMAND ----------

class AirbnbRecommendationAgent(dspy.Signature):
    """You are an Airbnb recommendation assistant for San Francisco listings.
    Use your tools to look up listings and neighborhood prices, and answer with specific numbers."""

    user_request: str = dspy.<FILL_IN>(desc="The user's question about Airbnb listings")
    response: str = dspy.<FILL_IN>(desc="A helpful answer with specific listing details and prices")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC class AirbnbRecommendationAgent(dspy.Signature):
# MAGIC     """You are an Airbnb recommendation assistant for San Francisco listings.
# MAGIC     Use your tools to look up listings and neighborhood prices, and answer with specific numbers."""
# MAGIC
# MAGIC     user_request: str = dspy.InputField(desc="The user's question about Airbnb listings")
# MAGIC     response: str = dspy.OutputField(desc="A helpful answer with specific listing details and prices")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### F2. Create the Agent
# MAGIC
# MAGIC `dspy.ReAct` alternates between **reasoning** about what to do next and **acting** by calling a tool, until it can answer.
# MAGIC
# MAGIC **TODO:** Create the agent from your signature and both tools. The output lists three tools: your two, plus `finish`, which ReAct always adds so the agent can signal that it's done.

# COMMAND ----------

agent = dspy.ReAct(
    <FILL_IN>,
    tools=[<FILL_IN>, <FILL_IN>],
)

for name, tool in agent.tools.items():
    print(f"- {name}: {tool.desc[:90]}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC agent = dspy.ReAct(
# MAGIC     AirbnbRecommendationAgent,
# MAGIC     tools=[get_listing_details, avg_neigh_price],
# MAGIC )
# MAGIC
# MAGIC for name, tool in agent.tools.items():
# MAGIC     print(f"- {name}: {tool.desc[:90]}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## G. Run and Inspect the Agent

# COMMAND ----------

# MAGIC %md
# MAGIC ### G1. Ask a Question That Needs Both Tools
# MAGIC
# MAGIC **TODO:** Call the agent, passing the question as the signature's input field, and print the signature's output field. Then open the trace under the output: you should see a call to `get_listing_details` and a call to `avg_neigh_price`.

# COMMAND ----------

result = agent(<FILL_IN>="Tell me about listing 958. Is it more expensive than the average listing in the Mission?")

print(result.<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC result = agent(user_request="Tell me about listing 958. Is it more expensive than the average listing in the Mission?")
# MAGIC
# MAGIC print(result.response)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### G2. Inspect the Trajectory
# MAGIC
# MAGIC Besides the answer, the result holds the agent's **trajectory**: for each step, its thought, the tool it chose, the arguments, and the result it observed. Run the next cell to print it.

# COMMAND ----------

for key, value in result.trajectory.items():
    print(f"{key}: {str(value)[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC To see the exact prompts DSPy sent to the model, use `dspy.inspect_history`.

# COMMAND ----------

dspy.inspect_history(n=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you:
# MAGIC 1. Connected DSPy to a Databricks model with `dspy.LM` and `dspy.configure`
# MAGIC 2. Declared tasks with shorthand and class-based signatures, and let modules write the prompts
# MAGIC 3. Prototyped a tool as a plain Python function and wrapped a Unity Catalog function as a second tool, using a parameterized query
# MAGIC 4. Built a `dspy.ReAct` agent and inspected its trajectory and MLflow trace
# MAGIC
# MAGIC DSPy's biggest strength, **optimizing** prompts and modules against your own data, builds on exactly these signatures and modules. To go further, see the [DSPy documentation](https://dspy.ai/) and the Databricks guide to DSPy (<a href="https://docs.databricks.com/aws/en/generative-ai/dspy/" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/generative-ai/dspy/" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/generative-ai/dspy/" style="color: #1976d2; text-decoration: underline;">GCP</a>).

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>