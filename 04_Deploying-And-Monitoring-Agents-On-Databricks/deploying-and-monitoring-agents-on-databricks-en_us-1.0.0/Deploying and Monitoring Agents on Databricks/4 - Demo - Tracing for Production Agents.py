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
# MAGIC # Demo - Tracing for Production Agents
# MAGIC ## Overview
# MAGIC Building on basic automatic tracing concepts, this notebook focuses on advanced MLflow Tracing techniques for production agents. You will instrument an agent with manual span decorators and typed spans, attach production-relevant metadata to traces, group multi-turn conversations with session IDs, and search traces programmatically for debugging and analysis.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC 1. Create manual spans with `@mlflow.trace` and assign semantic span types
# MAGIC 2. Attach custom span attributes for production context (user ID, session ID, app version)
# MAGIC 3. Group multi-turn conversations using session-level trace attributes
# MAGIC 4. Search and filter traces programmatically with `mlflow.search_traces()`

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
# MAGIC ## A. Install Dependencies and Setup

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-4

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. Quick Recap and Client Setup
# MAGIC
# MAGIC Automatic tracing with `mlflow.openai.autolog()` captures LLM calls with zero additional code. In this demo, we go further: manual spans, custom attributes, session grouping, and programmatic trace search.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Recap: Automatic Tracing</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC <code>mlflow.openai.autolog()</code> automatically traces OpenAI-compatible LLM calls. Automatic tracing captures the LLM call itself but not custom logic around it. For tool calls, retrieval steps, or business logic, you need <strong>manual tracing</strong>, which is the focus of this demo.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

import mlflow
from openai import OpenAI

# Enable automatic tracing for OpenAI-compatible clients
mlflow.openai.autolog()

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

BASE_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()+"/serving-endpoints"

client = OpenAI(
    base_url=BASE_URL,
    api_key=DATABRICKS_TOKEN
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Manual Tracing with Decorators

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Adding Custom Spans
# MAGIC
# MAGIC The `span_type` parameter on `@mlflow.trace` is a semantic label that tells MLflow what kind of GenAI step a function represents. This enables structured trace trees, specialized UI rendering, and downstream evaluation scorers that key off span type.
# MAGIC
# MAGIC **NOTE:** `span_type` accepts both string literals (e.g., `"TOOL"`) and the `SpanType` enum (e.g., `SpanType.TOOL`).
# MAGIC
# MAGIC #### Common values for <code>span_type</code>:
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 1000px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .span-row { display: block; white-space: nowrap; font-size: 0; width: 100%; box-sizing: border-box; }
# MAGIC
# MAGIC .span-box {
# MAGIC   display: inline-block;
# MAGIC   width: 15.8%;
# MAGIC   margin-right: 1%;
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
# MAGIC .span-box:last-child { margin-right: 0; }
# MAGIC
# MAGIC .span-box:hover { transform: translateY(-2px); }
# MAGIC
# MAGIC .span-box.active {
# MAGIC   background: #fff;
# MAGIC   border-left-color: var(--pc);
# MAGIC   border-right-color: var(--pc);
# MAGIC   border-bottom-color: var(--pc);
# MAGIC }
# MAGIC
# MAGIC .span-title {
# MAGIC   display: block;
# MAGIC   font-size: 13pt;
# MAGIC   font-weight: 700;
# MAGIC   color: #0b2026;
# MAGIC   line-height: 1.3;
# MAGIC   white-space: normal;
# MAGIC   pointer-events: none;
# MAGIC }
# MAGIC
# MAGIC .span-detail-wrap {
# MAGIC   overflow: hidden;
# MAGIC   max-height: 0;
# MAGIC   opacity: 0;
# MAGIC   transition: max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease;
# MAGIC   margin-top: 0;
# MAGIC }
# MAGIC
# MAGIC .span-detail-wrap.open {
# MAGIC   max-height: 1200px;
# MAGIC   opacity: 1;
# MAGIC   margin-top: 12px;
# MAGIC }
# MAGIC
# MAGIC .span-detail-card {
# MAGIC   background: #F9F7F4;
# MAGIC   border-radius: 10px;
# MAGIC   padding: 20px;
# MAGIC   box-sizing: border-box;
# MAGIC   border-top: 7px solid #ccc;
# MAGIC }
# MAGIC
# MAGIC .span-can-block {
# MAGIC   border-left: 4px solid;
# MAGIC   border-radius: 0 6px 6px 0;
# MAGIC   padding: 14px 14px;
# MAGIC   margin-bottom: 14px;
# MAGIC   font-size: 14.5pt;
# MAGIC   line-height: 1.6;
# MAGIC }
# MAGIC
# MAGIC .span-can-block ul {
# MAGIC   margin: 8px 0 0 0;
# MAGIC   padding-left: 20px;
# MAGIC }
# MAGIC
# MAGIC .span-can-block li {
# MAGIC   margin-bottom: 8px;
# MAGIC }
# MAGIC
# MAGIC .span-example-block {
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
# MAGIC   <div class="span-row">
# MAGIC     <span class="span-box active" data-id="0" onclick="selectSpan(0)" style="--pc:#2574B5; border-top-color:#2574B5;">
# MAGIC       <span class="span-title">AGENT</span>
# MAGIC     </span><span class="span-box" data-id="1" onclick="selectSpan(1)" style="--pc:#02A36F; border-top-color:#02A36F;">
# MAGIC       <span class="span-title">TOOL</span>
# MAGIC     </span><span class="span-box" data-id="2" onclick="selectSpan(2)" style="--pc:#FE3722; border-top-color:#FE3722;">
# MAGIC       <span class="span-title">RETRIEVER</span>
# MAGIC     </span><span class="span-box" data-id="3" onclick="selectSpan(3)" style="--pc:#F8A805; border-top-color:#F8A805;">
# MAGIC       <span class="span-title">CHAT_MODEL</span>
# MAGIC     </span><span class="span-box" data-id="4" onclick="selectSpan(4)" style="--pc:#98182A; border-top-color:#98182A;">
# MAGIC       <span class="span-title">PARSER</span>
# MAGIC     </span><span class="span-box" data-id="5" onclick="selectSpan(5)" style="--pc:#1B5162; border-top-color:#1B5162;">
# MAGIC       <span class="span-title">CHAIN</span>
# MAGIC     </span>
# MAGIC   </div>
# MAGIC
# MAGIC   <div class="span-detail-wrap" id="span-detail-wrap">
# MAGIC     <div class="span-detail-card" id="span-detail-card"></div>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var SPAN_TYPES = [
# MAGIC   {
# MAGIC     title: "AGENT",
# MAGIC     color: "#2574B5",
# MAGIC     bg: "rgba(37,116,181,0.10)",
# MAGIC     exampleBg: "rgba(37,116,181,0.10)",
# MAGIC     content: `
# MAGIC       <ul>
# MAGIC         <li>The <strong>top-level orchestrator</strong> span that represents the overall agent logic</li>
# MAGIC         <li>Typically the root or near-root span in a trace tree, containing child spans for tools, LLM calls, and other steps</li>
# MAGIC         <li>Use this for the main entry point of your agent pipeline</li>
# MAGIC       </ul>
# MAGIC     `,
# MAGIC     example: '<code>@mlflow.trace(span_type="AGENT")<br>def travel_agent(user_query: str) -> str:</code>'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "TOOL",
# MAGIC     color: "#02A36F",
# MAGIC     bg: "rgba(2,163,111,0.10)",
# MAGIC     exampleBg: "rgba(2,163,111,0.10)",
# MAGIC     content: `
# MAGIC       <ul>
# MAGIC         <li>Marks a function as a <strong>tool call</strong>: UC functions, SQL queries, API calls, or any utility the agent invokes</li>
# MAGIC         <li>MLflow renders tool spans with dedicated input/output formatting in the trace UI</li>
# MAGIC         <li>Also available as <code>SpanType.TOOL</code></li>
# MAGIC       </ul>
# MAGIC     `,
# MAGIC     example: '<code>@mlflow.trace(span_type="TOOL")<br>def lookup_weather(city: str) -> dict:</code>'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "RETRIEVER",
# MAGIC     color: "#FE3722",
# MAGIC     bg: "rgba(254,55,34,0.10)",
# MAGIC     exampleBg: "rgba(254,55,34,0.10)",
# MAGIC     content: `
# MAGIC       <ul>
# MAGIC         <li>Represents a <strong>retrieval or AI search</strong> step in a RAG pipeline</li>
# MAGIC         <li>Required by evaluation scorers like <strong>RetrievalGroundedness</strong> and <strong>RetrievalRelevance</strong>. These scorers locate RETRIEVER spans to extract retrieved documents</li>
# MAGIC         <li>Use this for any function that fetches context from an AI search index or document retrieval system</li>
# MAGIC       </ul>
# MAGIC     `,
# MAGIC     example: '<code>@mlflow.trace(span_type="RETRIEVER")<br>def search_knowledge_base(query: str) -> list:</code>'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "CHAT_MODEL",
# MAGIC     color: "#F8A805",
# MAGIC     bg: "rgba(248,168,5,0.10)",
# MAGIC     exampleBg: "rgba(248,168,5,0.10)",
# MAGIC     content: `
# MAGIC       <ul>
# MAGIC         <li>Represents the underlying <strong>LLM / chat completion call</strong></li>
# MAGIC         <li>Autologging (e.g., <code>mlflow.openai.autolog()</code>) automatically creates CHAT_MODEL spans for supported providers</li>
# MAGIC         <li>Use this when wrapping a custom or direct LLM call that autologging does not capture</li>
# MAGIC       </ul>
# MAGIC     `,
# MAGIC     example: '<code>@mlflow.trace(span_type="CHAT_MODEL")<br>def call_llm(messages: list) -> str:</code>'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "PARSER",
# MAGIC     color: "#98182A",
# MAGIC     bg: "rgba(152,24,42,0.10)",
# MAGIC     exampleBg: "rgba(152,24,42,0.10)",
# MAGIC     content: `
# MAGIC       <ul>
# MAGIC         <li>Marks <strong>parsing, formatting, or pre/post-processing</strong> logic</li>
# MAGIC         <li>Use for functions that transform LLM output into structured data, extract fields from responses, or prepare inputs for downstream steps</li>
# MAGIC         <li>Helps distinguish data transformation from business logic in complex traces</li>
# MAGIC       </ul>
# MAGIC     `,
# MAGIC     example: '<code>@mlflow.trace(span_type="PARSER")<br>def extract_json_from_response(raw: str) -> dict:</code>'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "CHAIN / FUNCTION",
# MAGIC     color: "#1B5162",
# MAGIC     bg: "rgba(27,81,98,0.10)",
# MAGIC     exampleBg: "rgba(27,81,98,0.10)",
# MAGIC     content: `
# MAGIC       <ul>
# MAGIC         <li>General-purpose span types for <strong>higher-level chains</strong> or <strong>generic helper functions</strong> in a pipeline</li>
# MAGIC         <li><code>CHAIN</code> is useful for grouping a sequence of steps (e.g., retrieve → process → respond) under a single logical unit</li>
# MAGIC         <li><code>FUNCTION</code> is the most generic type. Use it when none of the other span types apply</li>
# MAGIC       </ul>
# MAGIC     `,
# MAGIC     example: '<code>@mlflow.trace(span_type="CHAIN")<br>def rag_pipeline(question: str) -> str:</code>'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var spanCurrent = null;
# MAGIC
# MAGIC function selectSpan(id) {
# MAGIC   var wrap = document.getElementById('span-detail-wrap');
# MAGIC   var card = document.getElementById('span-detail-card');
# MAGIC   var c = SPAN_TYPES[id];
# MAGIC
# MAGIC   document.querySelectorAll('.span-box').forEach(function(b) {
# MAGIC     var isActive = parseInt(b.dataset.id, 10) === id;
# MAGIC     b.classList.toggle('active', isActive);
# MAGIC   });
# MAGIC
# MAGIC   if (spanCurrent === id) {
# MAGIC     wrap.classList.remove('open');
# MAGIC     document.querySelectorAll('.span-box').forEach(function(b) {
# MAGIC       b.classList.remove('active');
# MAGIC     });
# MAGIC     spanCurrent = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   spanCurrent = id;
# MAGIC
# MAGIC   card.style.borderTopColor = c.color;
# MAGIC   card.innerHTML =
# MAGIC     '<div style="font-size:18pt;font-weight:700;margin-bottom:12px;color:#0b2026;">' + c.title + '</div>'
# MAGIC     + '<div class="span-can-block" style="background:' + c.bg + ';border-color:' + c.color + ';">'
# MAGIC     + c.content
# MAGIC     + '</div>'
# MAGIC     + '<div class="span-example-block" style="background:' + c.exampleBg + ';border-color:' + c.color + ';">'
# MAGIC     + '<strong>Usage:</strong> ' + c.example
# MAGIC     + '</div>';
# MAGIC
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC
# MAGIC selectSpan(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC #### Select each concept to highlight it in the code
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 1200px; margin: 0 auto; font-family: sans-serif; color: #0b2026;">
# MAGIC
# MAGIC <style>
# MAGIC .c1c-cards { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
# MAGIC .c1c-card {
# MAGIC   flex: 1; background: #F9F7F4; border-top: 6px solid transparent;
# MAGIC   border-left: 2px solid transparent; border-right: 2px solid transparent; border-bottom: 2px solid transparent;
# MAGIC   border-radius: 8px; padding: 12px 10px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, background 0.15s;
# MAGIC }
# MAGIC .c1c-card:hover { transform: translateY(-2px); }
# MAGIC .c1c-card.active { background: #fff; border-left-color: var(--dc); border-right-color: var(--dc); border-bottom-color: var(--dc); }
# MAGIC .c1c-card-label { display: block; font-size: 12.5pt; font-weight: 700; color: #0b2026; line-height: 1.3; pointer-events: none; }
# MAGIC .c1c-layout { display: flex; gap: 22px; align-items: stretch; }
# MAGIC .c1c-code-wrap { flex: 1; position: relative; }
# MAGIC .c1c-code {
# MAGIC   border: 1px solid #000; border-radius: 10px; padding: 20px 22px; font-family: 'Menlo','Consolas',monospace;
# MAGIC   font-size: 12.5pt; line-height: 1.75; overflow-x: auto; background: #fafafa; color: #383a42;
# MAGIC   transition: background 0.3s, color 0.3s;
# MAGIC }
# MAGIC .c1c-code .tk-kw { color: #a626a4; } .c1c-code .tk-fn { color: #4078f2; }
# MAGIC .c1c-code .tk-str { color: #50a14f; } .c1c-code .tk-num { color: #986801; }
# MAGIC .c1c-code .tk-cmt { color: #a0a1a7; } .c1c-code .tk-dim { color: #696c77; }
# MAGIC .c1c-code .line { display: block; padding: 1px 6px; border-radius: 3px; transition: background 0.25s, opacity 0.25s; }
# MAGIC .c1c-code.has-highlight .line { opacity: 0.35; }
# MAGIC .c1c-code.has-highlight .line.hl { opacity: 1; background: rgba(0,0,0,0.06); }
# MAGIC .c1c-explain { flex: 0 0 260px; display: flex; flex-direction: column; justify-content: center; }
# MAGIC .c1c-explain-card { background: #F9F7F4; border-radius: 10px; border-top: 6px solid #ccc; padding: 20px; font-size: 14pt; line-height: 1.6; opacity: 0; transition: opacity 0.3s; }
# MAGIC .c1c-explain-card.visible { opacity: 1; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="c1c-cards">
# MAGIC   <div class="c1c-card" data-id="0" onclick="c1cSelect(0)" style="--dc:#02A36F; border-top-color:#02A36F;">
# MAGIC     <span class="c1c-card-label">TOOL Span</span>
# MAGIC   </div>
# MAGIC   <div class="c1c-card" data-id="1" onclick="c1cSelect(1)" style="--dc:#9B59B6; border-top-color:#9B59B6;">
# MAGIC     <span class="c1c-card-label">CHAIN Span</span>
# MAGIC   </div>
# MAGIC   <div class="c1c-card" data-id="2" onclick="c1cSelect(2)" style="--dc:#F8A805; border-top-color:#F8A805;">
# MAGIC     <span class="c1c-card-label">Agent Orchestration</span>
# MAGIC   </div>
# MAGIC   <div class="c1c-card" data-id="3" onclick="c1cSelect(3)" style="--dc:#4299E0; border-top-color:#4299E0;">
# MAGIC     <span class="c1c-card-label">Invocation</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="c1c-layout">
# MAGIC   <div class="c1c-code-wrap" id="c1c-code-wrap">
# MAGIC     <div class="c1c-code" id="c1c-code">
# MAGIC       <span class="line" data-g="0"><span class="tk-fn">@mlflow.trace</span>(<span class="tk-dim">span_type=</span><span class="tk-str">"TOOL"</span>)</span>
# MAGIC       <span class="line" data-g="0"><span class="tk-kw">def</span> <span class="tk-fn">lookup_weather</span>(city: <span class="tk-dim">str</span>) -> <span class="tk-dim">dict</span>:</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-cmt">"""Simulated weather tool for demonstration."""</span></span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;weather_data = {</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"tokyo"</span>: {<span class="tk-str">"temp_f"</span>: <span class="tk-num">72</span>, ...},</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"london"</span>: {<span class="tk-str">"temp_f"</span>: <span class="tk-num">58</span>, ...},</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;}</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-kw">return</span> weather_data.get(city.lower(), ...)</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="1"><span class="tk-fn">@mlflow.trace</span>(<span class="tk-dim">span_type=</span><span class="tk-str">"CHAIN"</span>)</span>
# MAGIC       <span class="line" data-g="1"><span class="tk-kw">def</span> <span class="tk-fn">travel_agent</span>(user_query: <span class="tk-dim">str</span>) -> <span class="tk-dim">str</span>:</span>
# MAGIC       <span class="line" data-g="1">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-cmt">"""A simple agent that looks up weather before responding."""</span></span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-cmt"># Step 1: Determine if weather is needed</span></span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;weather_info = <span class="tk-fn">lookup_weather</span>(<span class="tk-str">"tokyo"</span>)</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-cmt"># Step 2: Call the LLM with context</span></span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;response = client.chat.completions.<span class="tk-fn">create</span>(</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model=<span class="tk-str">"databricks-gpt-5-4"</span>,</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;messages=[...],</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_tokens=<span class="tk-num">500</span></span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;)</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-kw">return</span> response.choices[<span class="tk-num">0</span>].message.content</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="3">result = <span class="tk-fn">travel_agent</span>(<span class="tk-str">"Should I pack an umbrella for Tokyo?"</span>)</span>
# MAGIC       <span class="line" data-g="3"><span class="tk-fn">print</span>(result)</span>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <div class="c1c-explain">
# MAGIC     <div class="c1c-explain-card" id="c1c-explain-card"></div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var C1C_DATA = [
# MAGIC   { color:'#02A36F', title:'@mlflow.trace(span_type="TOOL")',
# MAGIC     text:'Marks <code>lookup_weather</code> as a <strong>TOOL</strong> span. When the agent calls this function, MLflow records it as a distinct tool invocation in the trace tree with its own inputs (city) and outputs (weather dict).' },
# MAGIC   { color:'#9B59B6', title:'@mlflow.trace(span_type="CHAIN")',
# MAGIC     text:'Wraps <code>travel_agent</code> as a <strong>CHAIN</strong> span, grouping a sequence of steps (tool call + LLM call) under a single logical unit. The TOOL span from <code>lookup_weather</code> and the auto-traced CHAT_MODEL span from the LLM call both nest as children.' },
# MAGIC   { color:'#F8A805', title:'Agent Orchestration Logic',
# MAGIC     text:'The body of <code>travel_agent</code> is the orchestration layer: it calls the weather <strong>tool</strong>, injects the result into the prompt, and sends the enriched prompt to the <strong>LLM</strong>. Each step becomes a visible child span.' },
# MAGIC   { color:'#4299E0', title:'Invoking the Agent',
# MAGIC     text:'Calling <code>travel_agent()</code> triggers the full traced execution. The result is a hierarchical trace: <strong>CHAIN</strong> (travel_agent) &rarr; <strong>TOOL</strong> (lookup_weather) + <strong>CHAT_MODEL</strong> (LLM call).' }
# MAGIC ];
# MAGIC var c1cCurrent = null, c1cLight = false;
# MAGIC function c1cToggle() {
# MAGIC   c1cLight = !c1cLight;
# MAGIC }
# MAGIC function c1cSelect(id) {
# MAGIC   var code = document.getElementById('c1c-code'), card = document.getElementById('c1c-explain-card');
# MAGIC   document.querySelectorAll('.c1c-card').forEach(function(b){ b.classList.toggle('active', parseInt(b.dataset.id)===id); });
# MAGIC   if (c1cCurrent === id) { code.classList.remove('has-highlight'); card.classList.remove('visible'); document.querySelectorAll('.c1c-card').forEach(function(b){b.classList.remove('active');}); c1cCurrent=null; return; }
# MAGIC   c1cCurrent = id; var c = C1C_DATA[id];
# MAGIC   code.classList.add('has-highlight');
# MAGIC   code.querySelectorAll('.line').forEach(function(ln){ ln.classList.toggle('hl', ln.dataset.g===String(id)); });
# MAGIC   card.style.borderTopColor = c.color;
# MAGIC   card.innerHTML = '<div style="font-size:15pt;font-weight:700;margin-bottom:10px;color:'+c.color+';">'+c.title+'</div><div>'+c.text+'</div>';
# MAGIC   card.classList.add('visible');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

import mlflow
import json

@mlflow.trace(span_type="TOOL")
def lookup_weather(city: str) -> dict:
    """Simulated weather tool for demonstration."""
    weather_data = {
        "tokyo": {"temp_f": 72, "condition": "partly cloudy", "humidity": 65},
        "london": {"temp_f": 58, "condition": "rainy", "humidity": 80},
        "new york": {"temp_f": 68, "condition": "sunny", "humidity": 45},
    }
    return weather_data.get(city.lower(), {"temp_f": 0, "condition": "unknown", "humidity": 0})


@mlflow.trace(span_type="CHAIN")
def travel_agent(user_query: str) -> str:
    """A simple agent that looks up weather before responding."""
    # Step 1: Determine if weather is needed
    weather_info = lookup_weather("tokyo")

    # Step 2: Call the LLM with context
    response = client.chat.completions.create(
        model="databricks-gpt-5-4",
        messages=[
            {"role": "system", "content": "You are a travel assistant. Use the provided weather data in your response."},
            {"role": "user", "content": f"{user_query}\n\nCurrent weather: {json.dumps(weather_info)}"}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


result = travel_agent("Should I pack an umbrella for Tokyo this weekend?")
print(result)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Reading the Span Tree
# MAGIC
# MAGIC The trace now shows a hierarchy. Each span records its start time, end time, inputs, outputs, and status.
# MAGIC
# MAGIC <div style="max-width:680px;margin:16px 0;font-family:'Menlo','Consolas',monospace;font-size:13pt;line-height:2;background:#f7f8fc;border:1px solid #dde0e8;border-radius:10px;padding:18px 22px;">
# MAGIC <div style="color:#3b82f6;font-weight:700;">CHAIN <span style="color:#999;font-weight:400;">travel_agent</span></div>
# MAGIC <div style="margin-left:24px;border-left:2px solid #dde0e8;padding-left:14px;">
# MAGIC <div style="color:#16a34a;font-weight:700;">TOOL <span style="color:#999;font-weight:400;">lookup_weather</span> <span style="color:#bbb;font-size:11pt;">inputs: "tokyo" &rarr; outputs: {temp_f: 72, ...}</span></div>
# MAGIC <div style="color:#a855f7;font-weight:700;">CHAT_MODEL <span style="color:#999;font-weight:400;">auto-captured</span> <span style="color:#bbb;font-size:11pt;">prompt + response</span></div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC This is the execution graph you will use for debugging in Module 3.

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Span Attributes for Production Context

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Adding Metadata to Spans
# MAGIC
# MAGIC In production, you need to correlate traces with business context: which user made the request, what session it belongs to, what application version is running. Span attributes let you attach arbitrary key-value metadata to any span.
# MAGIC
# MAGIC For example, after running the cell below, click on **Attributes** to view the different attributes that were logged with `span.set_attributes()` in the definition of `travel_agent_with_context()`.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC #### Select each concept to highlight it in the code
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif; color: #0b2026;">
# MAGIC
# MAGIC <style>
# MAGIC .d1-cards {
# MAGIC   display: flex;
# MAGIC   flex-wrap: wrap;
# MAGIC   gap: 8px;
# MAGIC   margin-bottom: 18px;
# MAGIC }
# MAGIC
# MAGIC .d1-card {
# MAGIC   flex: 1;
# MAGIC   background: #F9F7F4;
# MAGIC   border-top: 6px solid transparent;
# MAGIC   border-left: 2px solid transparent;
# MAGIC   border-right: 2px solid transparent;
# MAGIC   border-bottom: 2px solid transparent;
# MAGIC   border-radius: 8px;
# MAGIC   padding: 12px 10px;
# MAGIC   text-align: center;
# MAGIC   cursor: pointer;
# MAGIC   user-select: none;
# MAGIC   transition: transform 0.12s, background 0.15s;
# MAGIC }
# MAGIC
# MAGIC .d1-card:hover { transform: translateY(-2px); }
# MAGIC
# MAGIC .d1-card.active {
# MAGIC   background: #fff;
# MAGIC   border-left-color: var(--dc);
# MAGIC   border-right-color: var(--dc);
# MAGIC   border-bottom-color: var(--dc);
# MAGIC }
# MAGIC
# MAGIC .d1-card-label {
# MAGIC   display: block;
# MAGIC   font-size: 12.5pt;
# MAGIC   font-weight: 700;
# MAGIC   color: #0b2026;
# MAGIC   line-height: 1.3;
# MAGIC   pointer-events: none;
# MAGIC }
# MAGIC
# MAGIC .d1-layout {
# MAGIC   display: flex;
# MAGIC   gap: 22px;
# MAGIC   align-items: stretch;
# MAGIC }
# MAGIC
# MAGIC .d1-code-wrap {
# MAGIC   flex: 1;
# MAGIC   position: relative;
# MAGIC }
# MAGIC
# MAGIC .d1-code {
# MAGIC   border: 1px solid #000; border-radius: 10px;
# MAGIC   padding: 20px 22px;
# MAGIC   font-family: 'Menlo', 'Consolas', monospace;
# MAGIC   font-size: 12.5pt;
# MAGIC   line-height: 1.75;
# MAGIC   overflow-x: auto;
# MAGIC   background: #fafafa;
# MAGIC   color: #383a42;
# MAGIC }
# MAGIC
# MAGIC .d1-code .tk-kw { color: #a626a4; }
# MAGIC .d1-code .tk-fn { color: #4078f2; }
# MAGIC .d1-code .tk-str { color: #50a14f; }
# MAGIC .d1-code .tk-num { color: #986801; }
# MAGIC .d1-code .tk-cmt { color: #a0a1a7; }
# MAGIC .d1-code .tk-dim { color: #696c77; }
# MAGIC
# MAGIC .d1-code .line {
# MAGIC   display: block;
# MAGIC   padding: 1px 6px;
# MAGIC   border-radius: 3px;
# MAGIC   transition: background 0.25s, opacity 0.25s;
# MAGIC }
# MAGIC
# MAGIC .d1-code.has-highlight .line { opacity: 0.35; }
# MAGIC .d1-code.has-highlight .line.hl { opacity: 1; background: rgba(0,0,0,0.06); }
# MAGIC
# MAGIC .d1-explain {
# MAGIC   flex: 0 0 300px;
# MAGIC   display: flex;
# MAGIC   flex-direction: column;
# MAGIC   justify-content: center;
# MAGIC }
# MAGIC
# MAGIC .d1-explain-card {
# MAGIC   background: #F9F7F4;
# MAGIC   border-radius: 10px;
# MAGIC   border-top: 6px solid #ccc;
# MAGIC   padding: 20px;
# MAGIC   font-size: 14pt;
# MAGIC   line-height: 1.6;
# MAGIC   opacity: 0;
# MAGIC   transition: opacity 0.3s;
# MAGIC }
# MAGIC
# MAGIC .d1-explain-card.visible { opacity: 1; }
# MAGIC </style>
# MAGIC
# MAGIC <!-- Card selectors -->
# MAGIC <div class="d1-cards">
# MAGIC   <div class="d1-card" data-id="0" onclick="d1Select(0)" style="--dc:#9B59B6; border-top-color:#9B59B6;">
# MAGIC     <span class="d1-card-label">@mlflow.trace</span>
# MAGIC   </div>
# MAGIC   <div class="d1-card" data-id="1" onclick="d1Select(1)" style="--dc:#4299E0; border-top-color:#4299E0;">
# MAGIC     <span class="d1-card-label">Get Active Span</span>
# MAGIC   </div>
# MAGIC   <div class="d1-card" data-id="2" onclick="d1Select(2)" style="--dc:#02A36F; border-top-color:#02A36F;">
# MAGIC     <span class="d1-card-label">Set Attributes</span>
# MAGIC   </div>
# MAGIC   <div class="d1-card" data-id="3" onclick="d1Select(3)" style="--dc:#F8A805; border-top-color:#F8A805;">
# MAGIC     <span class="d1-card-label">Tool + LLM</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <!-- Code + explanation side by side -->
# MAGIC <div class="d1-layout">
# MAGIC
# MAGIC   <div class="d1-code-wrap" id="d1-code-wrap">
# MAGIC     <div class="d1-code" id="d1-code">
# MAGIC       <span class="line" data-g="0"><span class="tk-fn">@mlflow.trace</span>(<span class="tk-dim">span_type=</span><span class="tk-str">"CHAIN"</span>)</span>
# MAGIC       <span class="line" data-g="0"><span class="tk-kw">def</span> <span class="tk-fn">travel_agent_with_context</span>(user_query, user_id, session_id):</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-cmt">"""Agent with production context attached to the trace."""</span></span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="1">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-cmt"># Get the current span and add production metadata</span></span>
# MAGIC       <span class="line" data-g="1">&nbsp;&nbsp;&nbsp;&nbsp;span = mlflow.<span class="tk-fn">get_current_active_span</span>()</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;span.<span class="tk-fn">set_attributes</span>({</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"user_id"</span>: user_id,</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"session_id"</span>: session_id,</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"app_version"</span>: <span class="tk-str">"1.2.0"</span>,</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"request_source"</span>: <span class="tk-str">"web_ui"</span></span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;})</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;weather_info = <span class="tk-fn">lookup_weather</span>(<span class="tk-str">"london"</span>)</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;response = client.chat.completions.<span class="tk-fn">create</span>(</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model=<span class="tk-str">"databricks-gpt-5-4"</span>,</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;messages=[...],</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_tokens=<span class="tk-num">500</span></span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;)</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-kw">return</span> response.choices[<span class="tk-num">0</span>].message.content</span>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC
# MAGIC   <div class="d1-explain">
# MAGIC     <div class="d1-explain-card" id="d1-explain-card"></div>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var D1_DATA = [
# MAGIC   {
# MAGIC     color: '#9B59B6',
# MAGIC     title: '@mlflow.trace(span_type="CHAIN")',
# MAGIC     text: 'The decorator wraps the entire function as a <strong>CHAIN</strong> span. Everything that executes inside this function becomes a child in the trace tree. The <code>span_type</code> label tells MLflow this is a higher-level orchestration step, not a tool or LLM call.'
# MAGIC   },
# MAGIC   {
# MAGIC     color: '#4299E0',
# MAGIC     title: 'get_current_active_span()',
# MAGIC     text: 'Retrieves a reference to the span that <code>@mlflow.trace</code> just created. This gives you a handle to enrich the span from <em>inside</em> the function body, rather than only relying on what the decorator captures automatically (inputs and outputs).'
# MAGIC   },
# MAGIC   {
# MAGIC     color: '#02A36F',
# MAGIC     title: 'set_attributes({...})',
# MAGIC     text: 'Attaches arbitrary key-value pairs to the span. These attributes (<code>user_id</code>, <code>session_id</code>, <code>app_version</code>, <code>request_source</code>) become <strong>searchable and filterable</strong> in the MLflow UI and via <code>mlflow.search_traces()</code>.'
# MAGIC   },
# MAGIC   {
# MAGIC     color: '#F8A805',
# MAGIC     title: 'Tool Call + LLM Request',
# MAGIC     text: 'The <code>lookup_weather</code> tool call and <code>chat.completions.create</code> LLM call execute as normal Python. Because autologging is enabled, MLflow automatically creates a <strong>TOOL</strong> child span and a <strong>CHAT_MODEL</strong> child span nested under this CHAIN.'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var d1Current = null;
# MAGIC
# MAGIC function d1Select(id) {
# MAGIC   var codeEl = document.getElementById('d1-code');
# MAGIC   var explainCard = document.getElementById('d1-explain-card');
# MAGIC
# MAGIC   // Toggle cards
# MAGIC   document.querySelectorAll('.d1-card').forEach(function(b) {
# MAGIC     b.classList.toggle('active', parseInt(b.dataset.id) === id);
# MAGIC   });
# MAGIC
# MAGIC   // Deselect if same card clicked
# MAGIC   if (d1Current === id) {
# MAGIC     codeEl.classList.remove('has-highlight');
# MAGIC     explainCard.classList.remove('visible');
# MAGIC     document.querySelectorAll('.d1-card').forEach(function(b) { b.classList.remove('active'); });
# MAGIC     d1Current = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   d1Current = id;
# MAGIC   var c = D1_DATA[id];
# MAGIC
# MAGIC   // Highlight matching code lines
# MAGIC   codeEl.classList.add('has-highlight');
# MAGIC   codeEl.querySelectorAll('.line').forEach(function(ln) {
# MAGIC     ln.classList.toggle('hl', ln.dataset.g === String(id));
# MAGIC   });
# MAGIC
# MAGIC   // Show explanation
# MAGIC   explainCard.style.borderTopColor = c.color;
# MAGIC   explainCard.innerHTML =
# MAGIC     '<div style="font-size:15pt;font-weight:700;margin-bottom:10px;color:' + c.color + ';">' + c.title + '</div>'
# MAGIC     + '<div>' + c.text + '</div>';
# MAGIC   explainCard.classList.add('visible');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

@mlflow.trace(span_type="CHAIN")
def travel_agent_with_context(user_query: str, user_id: str, session_id: str) -> str:
    """Agent with production context attached to the trace."""
    # Get the current span and add production metadata
    span = mlflow.get_current_active_span()
    span.set_attributes({
        "user_id": user_id,
        "session_id": session_id,
        "app_version": "1.2.0",
        "request_source": "web_ui"
    })

    weather_info = lookup_weather("london")

    response = client.chat.completions.create(
        model="databricks-gpt-5-4",
        messages=[
            {"role": "system", "content": "You are a travel assistant."},
            {"role": "user", "content": f"{user_query}\n\nWeather: {json.dumps(weather_info)}"}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


result = travel_agent_with_context(
    user_query="What should I wear in London tomorrow?",
    user_id="user_42",
    session_id="sess_abc123"
)
print(result)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C2. Why Attributes Matter
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Span attributes are searchable and filterable. You can use attributes to slice traces by user, session, or app version when debugging production issues. Attach them early as you cannot add them retroactively.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Multi-Turn Conversation Tracing

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Session-Level Grouping
# MAGIC
# MAGIC Production agents often handle multi-turn conversations. Each turn produces its own trace, but you need to group them into sessions for analysis. Use a consistent `session_id` attribute across turns to enable session-level queries.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC #### Select each concept to highlight it in the code
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif; color: #0b2026;">
# MAGIC
# MAGIC <style>
# MAGIC .e1-cards { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
# MAGIC .e1-card {
# MAGIC   flex: 1; background: #F9F7F4; border-top: 6px solid transparent;
# MAGIC   border-left: 2px solid transparent; border-right: 2px solid transparent; border-bottom: 2px solid transparent;
# MAGIC   border-radius: 8px; padding: 12px 10px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, background 0.15s;
# MAGIC }
# MAGIC .e1-card:hover { transform: translateY(-2px); }
# MAGIC .e1-card.active { background: #fff; border-left-color: var(--dc); border-right-color: var(--dc); border-bottom-color: var(--dc); }
# MAGIC .e1-card-label { display: block; font-size: 12.5pt; font-weight: 700; color: #0b2026; line-height: 1.3; pointer-events: none; }
# MAGIC .e1-layout { display: flex; gap: 22px; align-items: stretch; }
# MAGIC .e1-code-wrap { flex: 1; position: relative; }
# MAGIC .e1-code {
# MAGIC   border: 1px solid #000; border-radius: 10px; padding: 20px 22px; font-family: 'Menlo','Consolas',monospace;
# MAGIC   font-size: 12.5pt; line-height: 1.75; overflow-x: auto; background: #fafafa; color: #383a42;
# MAGIC   transition: background 0.3s, color 0.3s;
# MAGIC }
# MAGIC .e1-code .tk-kw { color: #a626a4; } .e1-code .tk-fn { color: #4078f2; }
# MAGIC .e1-code .tk-str { color: #50a14f; } .e1-code .tk-num { color: #986801; }
# MAGIC .e1-code .tk-cmt { color: #a0a1a7; } .e1-code .tk-dim { color: #696c77; }
# MAGIC .e1-code .line { display: block; padding: 1px 6px; border-radius: 3px; transition: background 0.25s, opacity 0.25s; }
# MAGIC .e1-code.has-highlight .line { opacity: 0.35; }
# MAGIC .e1-code.has-highlight .line.hl { opacity: 1; background: rgba(0,0,0,0.06); }
# MAGIC .e1-explain { flex: 0 0 300px; display: flex; flex-direction: column; justify-content: center; }
# MAGIC .e1-explain-card { background: #F9F7F4; border-radius: 10px; border-top: 6px solid #ccc; padding: 20px; font-size: 14pt; line-height: 1.6; opacity: 0; transition: opacity 0.3s; }
# MAGIC .e1-explain-card.visible { opacity: 1; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="e1-cards">
# MAGIC   <div class="e1-card" data-id="0" onclick="e1Select(0)" style="--dc:#9B59B6; border-top-color:#9B59B6;">
# MAGIC     <span class="e1-card-label">Session Setup</span>
# MAGIC   </div>
# MAGIC   <div class="e1-card" data-id="1" onclick="e1Select(1)" style="--dc:#4299E0; border-top-color:#4299E0;">
# MAGIC     <span class="e1-card-label">Turn Decorator</span>
# MAGIC   </div>
# MAGIC   <div class="e1-card" data-id="2" onclick="e1Select(2)" style="--dc:#02A36F; border-top-color:#02A36F;">
# MAGIC     <span class="e1-card-label">Session Attributes</span>
# MAGIC   </div>
# MAGIC   <div class="e1-card" data-id="3" onclick="e1Select(3)" style="--dc:#F8A805; border-top-color:#F8A805;">
# MAGIC     <span class="e1-card-label">Conversation Memory</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="e1-layout">
# MAGIC   <div class="e1-code-wrap" id="e1-code-wrap">
# MAGIC     <div class="e1-code" id="e1-code">
# MAGIC       <span class="line" data-g="0"><span class="tk-kw">import</span> uuid</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="0">session_id = <span class="tk-fn">str</span>(<span class="tk-fn">uuid.uuid4</span>())</span>
# MAGIC       <span class="line" data-g="0">conversation_history = [</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;{<span class="tk-str">"role"</span>: <span class="tk-str">"system"</span>, <span class="tk-str">"content"</span>: <span class="tk-str">"You are a travel assistant. Be concise."</span>}</span>
# MAGIC       <span class="line" data-g="0">]</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="1"><span class="tk-fn">@mlflow.trace</span>(<span class="tk-dim">span_type=</span><span class="tk-str">"CHAIN"</span>, <span class="tk-dim">name=</span><span class="tk-str">"conversation_turn"</span>)</span>
# MAGIC       <span class="line" data-g="1"><span class="tk-kw">def</span> <span class="tk-fn">chat_turn</span>(user_message: <span class="tk-dim">str</span>, turn_number: <span class="tk-dim">int</span>) -> <span class="tk-dim">str</span>:</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;span = mlflow.<span class="tk-fn">get_current_active_span</span>()</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;span.<span class="tk-fn">set_attributes</span>({</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"session_id"</span>: session_id,</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"turn_number"</span>: turn_number</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;})</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;conversation_history.<span class="tk-fn">append</span>({<span class="tk-str">"role"</span>: <span class="tk-str">"user"</span>, <span class="tk-str">"content"</span>: user_message})</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;response = client.chat.completions.<span class="tk-fn">create</span>(</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model=<span class="tk-str">"databricks-gpt-5-4"</span>,</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;messages=conversation_history,</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_tokens=<span class="tk-num">300</span></span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;)</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;assistant_msg = response.choices[<span class="tk-num">0</span>].message.content</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;conversation_history.<span class="tk-fn">append</span>({<span class="tk-str">"role"</span>: <span class="tk-str">"assistant"</span>, <span class="tk-str">"content"</span>: assistant_msg})</span>
# MAGIC       <span class="line" data-g="3">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-kw">return</span> assistant_msg</span>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <div class="e1-explain">
# MAGIC     <div class="e1-explain-card" id="e1-explain-card"></div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var E1_DATA = [
# MAGIC   { color:'#9B59B6', title:'Session Setup',
# MAGIC     text:'A single <code>session_id</code> (UUID) is generated once and shared across all turns. The <code>conversation_history</code> list accumulates messages, giving the LLM context from prior turns.' },
# MAGIC   { color:'#4299E0', title:'Decorator with name= Parameter',
# MAGIC     text:'<code>@mlflow.trace(name="conversation_turn")</code> gives the span a <strong>custom name</strong> instead of defaulting to the function name. This makes traces easier to identify when browsing the UI.' },
# MAGIC   { color:'#02A36F', title:'Session & Turn Attributes',
# MAGIC     text:'Each turn attaches <code>session_id</code> and <code>turn_number</code> as span attributes. This lets you <strong>group all turns in a session</strong> with <code>mlflow.search_traces()</code> and filter by turn order.' },
# MAGIC   { color:'#F8A805', title:'Conversation Memory & LLM Call',
# MAGIC     text:'The user message is appended to history, the full history is sent to the LLM, and the assistant response is appended back. This accumulating pattern is how multi-turn context is maintained. The LLM call is auto-traced as a <strong>CHAT_MODEL</strong> child span.' }
# MAGIC ];
# MAGIC var e1Current = null, e1Light = false;
# MAGIC function e1Toggle() {
# MAGIC   e1Light = !e1Light;
# MAGIC }
# MAGIC function e1Select(id) {
# MAGIC   var code = document.getElementById('e1-code'), card = document.getElementById('e1-explain-card');
# MAGIC   document.querySelectorAll('.e1-card').forEach(function(b){ b.classList.toggle('active', parseInt(b.dataset.id)===id); });
# MAGIC   if (e1Current === id) { code.classList.remove('has-highlight'); card.classList.remove('visible'); document.querySelectorAll('.e1-card').forEach(function(b){b.classList.remove('active');}); e1Current=null; return; }
# MAGIC   e1Current = id; var c = E1_DATA[id];
# MAGIC   code.classList.add('has-highlight');
# MAGIC   code.querySelectorAll('.line').forEach(function(ln){ ln.classList.toggle('hl', ln.dataset.g===String(id)); });
# MAGIC   card.style.borderTopColor = c.color;
# MAGIC   card.innerHTML = '<div style="font-size:15pt;font-weight:700;margin-bottom:10px;color:'+c.color+';">'+c.title+'</div><div>'+c.text+'</div>';
# MAGIC   card.classList.add('visible');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

import uuid

session_id = str(uuid.uuid4())
conversation_history = [
    {"role": "system", "content": "You are a travel assistant. Be concise."}
]


@mlflow.trace(span_type="CHAIN", name="conversation_turn")
def chat_turn(user_message: str, turn_number: int) -> str:
    """Process a single turn in a multi-turn conversation."""
    span = mlflow.get_current_active_span()
    span.set_attributes({
        "session_id": session_id,
        "turn_number": turn_number
    })

    conversation_history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="databricks-gpt-5-4",
        messages=conversation_history,
        max_tokens=300
    )

    assistant_msg = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg


# Simulate a multi-turn conversation
print("Turn 1:", chat_turn("I'm planning a trip to Japan in April.", turn_number=1))

# COMMAND ----------

print("Turn 2:", chat_turn("What about the cherry blossoms, will I catch them?", turn_number=2))

# COMMAND ----------

print("Turn 3:", chat_turn("Do you recommend anything else to do while in Kyoto?", turn_number=3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Viewing Session Traces
# MAGIC
# MAGIC Each turn above produced a separate trace, but they all share the same `session_id` attribute. This lets you reconstruct the full conversation during debugging or analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Searching and Filtering Traces

# COMMAND ----------

# MAGIC %md
# MAGIC ### E1. Programmatic Trace Search
# MAGIC
# MAGIC In production, you will not browse traces one at a time in the UI. `mlflow.search_traces()` lets you query traces programmatically. Filter by time range, status, attributes, or any combination.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC #### Select each concept to highlight it in the code
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif; color: #0b2026;">
# MAGIC
# MAGIC <style>
# MAGIC .f1-cards { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
# MAGIC .f1-card {
# MAGIC   flex: 1; background: #F9F7F4; border-top: 6px solid transparent;
# MAGIC   border-left: 2px solid transparent; border-right: 2px solid transparent; border-bottom: 2px solid transparent;
# MAGIC   border-radius: 8px; padding: 12px 10px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, background 0.15s;
# MAGIC }
# MAGIC .f1-card:hover { transform: translateY(-2px); }
# MAGIC .f1-card.active { background: #fff; border-left-color: var(--dc); border-right-color: var(--dc); border-bottom-color: var(--dc); }
# MAGIC .f1-card-label { display: block; font-size: 12.5pt; font-weight: 700; color: #0b2026; line-height: 1.3; pointer-events: none; }
# MAGIC .f1-layout { display: flex; gap: 22px; align-items: stretch; }
# MAGIC .f1-code-wrap { flex: 1; position: relative; }
# MAGIC .f1-code {
# MAGIC   border: 1px solid #000; border-radius: 10px; padding: 20px 22px; font-family: 'Menlo','Consolas',monospace;
# MAGIC   font-size: 12.5pt; line-height: 1.75; overflow-x: auto; background: #fafafa; color: #383a42;
# MAGIC   transition: background 0.3s, color 0.3s;
# MAGIC }
# MAGIC .f1-code .tk-kw { color: #a626a4; } .f1-code .tk-fn { color: #4078f2; }
# MAGIC .f1-code .tk-str { color: #50a14f; } .f1-code .tk-num { color: #986801; }
# MAGIC .f1-code .tk-cmt { color: #a0a1a7; } .f1-code .tk-dim { color: #696c77; }
# MAGIC .f1-code .line { display: block; padding: 1px 6px; border-radius: 3px; transition: background 0.25s, opacity 0.25s; }
# MAGIC .f1-code.has-highlight .line { opacity: 0.35; }
# MAGIC .f1-code.has-highlight .line.hl { opacity: 1; background: rgba(0,0,0,0.06); }
# MAGIC .f1-explain { flex: 0 0 300px; display: flex; flex-direction: column; justify-content: center; }
# MAGIC .f1-explain-card { background: #F9F7F4; border-radius: 10px; border-top: 6px solid #ccc; padding: 20px; font-size: 14pt; line-height: 1.6; opacity: 0; transition: opacity 0.3s; }
# MAGIC .f1-explain-card.visible { opacity: 1; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="f1-cards">
# MAGIC   <div class="f1-card" data-id="0" onclick="f1Select(0)" style="--dc:#9B59B6; border-top-color:#9B59B6;">
# MAGIC     <span class="f1-card-label">Get Experiment</span>
# MAGIC   </div>
# MAGIC   <div class="f1-card" data-id="1" onclick="f1Select(1)" style="--dc:#02A36F; border-top-color:#02A36F;">
# MAGIC     <span class="f1-card-label">Search Traces</span>
# MAGIC   </div>
# MAGIC   <div class="f1-card" data-id="2" onclick="f1Select(2)" style="--dc:#4299E0; border-top-color:#4299E0;">
# MAGIC     <span class="f1-card-label">Display Results</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="f1-layout">
# MAGIC   <div class="f1-code-wrap" id="f1-code-wrap">
# MAGIC     <div class="f1-code" id="f1-code">
# MAGIC       <span class="line" data-g="0"><span class="tk-cmt"># Get the experiment associated with this notebook</span></span>
# MAGIC       <span class="line" data-g="0">experiment = mlflow.<span class="tk-fn">get_experiment_by_name</span>(</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;mlflow_experiment_path</span>
# MAGIC       <span class="line" data-g="0">)</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="0"><span class="tk-cmt"># If no experiment found, use the active one</span></span>
# MAGIC       <span class="line" data-g="0">experiment_id = experiment.experiment_id \</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-kw">if</span> experiment <span class="tk-kw">else</span> mlflow.<span class="tk-fn">get_experiment</span>(<span class="tk-num">0</span>).experiment_id</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="1"><span class="tk-cmt"># Search for recent traces</span></span>
# MAGIC       <span class="line" data-g="1">traces_df = mlflow.<span class="tk-fn">search_traces</span>(</span>
# MAGIC       <span class="line" data-g="1">&nbsp;&nbsp;&nbsp;&nbsp;locations=[experiment_id],</span>
# MAGIC       <span class="line" data-g="1">&nbsp;&nbsp;&nbsp;&nbsp;max_results=<span class="tk-num">10</span></span>
# MAGIC       <span class="line" data-g="1">)</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="2"><span class="tk-fn">display</span>(traces_df[[</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"client_request_id"</span>, <span class="tk-str">"trace_id"</span>,</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"request_time"</span>, <span class="tk-str">"state"</span>,</span>
# MAGIC       <span class="line" data-g="2">&nbsp;&nbsp;&nbsp;&nbsp;<span class="tk-str">"execution_duration"</span></span>
# MAGIC       <span class="line" data-g="2">]])</span>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <div class="f1-explain">
# MAGIC     <div class="f1-explain-card" id="f1-explain-card"></div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var F1_DATA = [
# MAGIC   { color:'#9B59B6', title:'Resolve the Experiment',
# MAGIC     text:'Looks up the MLflow experiment by its workspace path. Every notebook has an associated experiment where traces and runs are logged. The fallback ensures the code works even if the path variable is not set.' },
# MAGIC   { color:'#02A36F', title:'mlflow.search_traces()',
# MAGIC     text:'The core query API. Takes a <code>locations</code> list (experiment IDs) and optional <code>filter_string</code>, <code>max_results</code>, and <code>order_by</code> parameters. Filter on <code>trace.*</code> (status, timing), <code>metadata.*</code>, <code>tag.*</code>, or <code>span.attributes.*</code>. Returns a <strong>Pandas DataFrame</strong> with one row per trace.' },
# MAGIC   { color:'#4299E0', title:'Display Selected Columns',
# MAGIC     text:'Selects the most useful columns for a quick overview: <code>client_request_id</code>, <code>trace_id</code>, <code>request_time</code>, <code>state</code> (OK or ERROR), and <code>execution_duration</code>. The Databricks <code>display()</code> renders them as an interactive table.' }
# MAGIC ];
# MAGIC var f1Current = null, f1Light = false;
# MAGIC function f1Toggle() {
# MAGIC   f1Light = !f1Light;
# MAGIC }
# MAGIC function f1Select(id) {
# MAGIC   var code = document.getElementById('f1-code'), card = document.getElementById('f1-explain-card');
# MAGIC   document.querySelectorAll('.f1-card').forEach(function(b){ b.classList.toggle('active', parseInt(b.dataset.id)===id); });
# MAGIC   if (f1Current === id) { code.classList.remove('has-highlight'); card.classList.remove('visible'); document.querySelectorAll('.f1-card').forEach(function(b){b.classList.remove('active');}); f1Current=null; return; }
# MAGIC   f1Current = id; var c = F1_DATA[id];
# MAGIC   code.classList.add('has-highlight');
# MAGIC   code.querySelectorAll('.line').forEach(function(ln){ ln.classList.toggle('hl', ln.dataset.g===String(id)); });
# MAGIC   card.style.borderTopColor = c.color;
# MAGIC   card.innerHTML = '<div style="font-size:15pt;font-weight:700;margin-bottom:10px;color:'+c.color+';">'+c.title+'</div><div>'+c.text+'</div>';
# MAGIC   card.classList.add('visible');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# Get the experiment associated with this notebook
experiment = mlflow.get_experiment_by_name(
    mlflow_experiment_path
)

# If no experiment found, use the active one
experiment_id = experiment.experiment_id if experiment else mlflow.get_experiment(0).experiment_id

# Search for recent traces. Adjust filter as needed
traces_df = mlflow.search_traces(
    locations=[experiment_id],
    max_results=10
)

display(traces_df[["client_request_id", "trace_id", "request_time", "state", "execution_duration"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Filtering by Status and Timing
# MAGIC
# MAGIC Common production queries include finding error traces and slow traces.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC #### Select each concept to highlight it in the code
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 1060px; margin: 0 auto; font-family: sans-serif; color: #0b2026;">
# MAGIC
# MAGIC <style>
# MAGIC .f2-cards { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
# MAGIC .f2-card {
# MAGIC   flex: 1; background: #F9F7F4; border-top: 6px solid transparent;
# MAGIC   border-left: 2px solid transparent; border-right: 2px solid transparent; border-bottom: 2px solid transparent;
# MAGIC   border-radius: 8px; padding: 12px 10px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, background 0.15s;
# MAGIC }
# MAGIC .f2-card:hover { transform: translateY(-2px); }
# MAGIC .f2-card.active { background: #fff; border-left-color: var(--dc); border-right-color: var(--dc); border-bottom-color: var(--dc); }
# MAGIC .f2-card-label { display: block; font-size: 12.5pt; font-weight: 700; color: #0b2026; line-height: 1.3; pointer-events: none; }
# MAGIC .f2-layout { display: flex; gap: 22px; align-items: stretch; }
# MAGIC .f2-code-wrap { flex: 1; position: relative; }
# MAGIC .f2-code {
# MAGIC   border: 1px solid #000; border-radius: 10px; padding: 20px 22px; font-family: 'Menlo','Consolas',monospace;
# MAGIC   font-size: 12.5pt; line-height: 1.75; overflow-x: auto; background: #fafafa; color: #383a42;
# MAGIC   transition: background 0.3s, color 0.3s;
# MAGIC }
# MAGIC .f2-code .tk-kw { color: #a626a4; } .f2-code .tk-fn { color: #4078f2; }
# MAGIC .f2-code .tk-str { color: #50a14f; } .f2-code .tk-num { color: #986801; }
# MAGIC .f2-code .tk-cmt { color: #a0a1a7; } .f2-code .tk-dim { color: #696c77; }
# MAGIC .f2-code .line { display: block; padding: 1px 6px; border-radius: 3px; transition: background 0.25s, opacity 0.25s; }
# MAGIC .f2-code.has-highlight .line { opacity: 0.35; }
# MAGIC .f2-code.has-highlight .line.hl { opacity: 1; background: rgba(0,0,0,0.06); }
# MAGIC .f2-explain { flex: 0 0 300px; display: flex; flex-direction: column; justify-content: center; }
# MAGIC .f2-explain-card { background: #F9F7F4; border-radius: 10px; border-top: 6px solid #ccc; padding: 20px; font-size: 14pt; line-height: 1.6; opacity: 0; transition: opacity 0.3s; }
# MAGIC .f2-explain-card.visible { opacity: 1; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="f2-cards">
# MAGIC   <div class="f2-card" data-id="0" onclick="f2Select(0)" style="--dc:#FE3722; border-top-color:#FE3722;">
# MAGIC     <span class="f2-card-label">Error Traces</span>
# MAGIC   </div>
# MAGIC   <div class="f2-card" data-id="1" onclick="f2Select(1)" style="--dc:#F8A805; border-top-color:#F8A805;">
# MAGIC     <span class="f2-card-label">Slow Traces</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="f2-layout">
# MAGIC   <div class="f2-code-wrap" id="f2-code-wrap">
# MAGIC     <div class="f2-code" id="f2-code">
# MAGIC       <span class="line" data-g="0"><span class="tk-cmt"># Find error traces</span></span>
# MAGIC       <span class="line" data-g="0">error_traces = mlflow.<span class="tk-fn">search_traces</span>(</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;locations=[experiment_id],</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;filter_string=<span class="tk-str">"trace.status = 'ERROR'"</span>,</span>
# MAGIC       <span class="line" data-g="0">&nbsp;&nbsp;&nbsp;&nbsp;max_results=<span class="tk-num">100</span></span>
# MAGIC       <span class="line" data-g="0">)</span>
# MAGIC       <span class="line" data-g="0"><span class="tk-fn">print</span>(<span class="tk-str">f"Error traces found: {</span><span class="tk-fn">len</span>(error_traces)<span class="tk-str">}"</span>)</span>
# MAGIC       <span class="line">&nbsp;</span>
# MAGIC       <span class="line" data-g="1"><span class="tk-cmt"># Find slow traces (over 5 seconds)</span></span>
# MAGIC       <span class="line" data-g="1">slow_traces = mlflow.<span class="tk-fn">search_traces</span>(</span>
# MAGIC       <span class="line" data-g="1">&nbsp;&nbsp;&nbsp;&nbsp;locations=[experiment_id],</span>
# MAGIC       <span class="line" data-g="1">&nbsp;&nbsp;&nbsp;&nbsp;filter_string=<span class="tk-str">"trace.execution_time_ms > 5000"</span>,</span>
# MAGIC       <span class="line" data-g="1">&nbsp;&nbsp;&nbsp;&nbsp;max_results=<span class="tk-num">100</span></span>
# MAGIC       <span class="line" data-g="1">)</span>
# MAGIC       <span class="line" data-g="1"><span class="tk-fn">print</span>(<span class="tk-str">f"Slow traces (>5s): {</span><span class="tk-fn">len</span>(slow_traces)<span class="tk-str">}"</span>)</span>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <div class="f2-explain">
# MAGIC     <div class="f2-explain-card" id="f2-explain-card"></div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var F2_DATA = [
# MAGIC   { color:'#FE3722', title:'Filter by Error Status',
# MAGIC     text:'Uses <code>filter_string="trace.status = \'ERROR\'"</code> to return only failed traces. The <code>trace.*</code> prefix is required for trace-level fields like <code>trace.status</code>, <code>trace.timestamp_ms</code>, and <code>trace.execution_time_ms</code>. You can also filter on <code>metadata.*</code>, <code>tag.*</code>, and <code>span.attributes.*</code> using <code>=</code>, <code>!=</code>, <code>&lt;</code>, <code>&gt;</code>, <code>LIKE</code>, and <code>AND</code>.' },
# MAGIC   { color:'#F8A805', title:'Filter by Execution Time',
# MAGIC     text:'Uses <code>filter_string="trace.execution_time_ms > 5000"</code> to find traces taking over 5 seconds. You can combine filters with <code>AND</code>, e.g. <code>trace.status = \'OK\' AND trace.execution_time_ms > 2500</code>.' }
# MAGIC ];
# MAGIC var f2Current = null, f2Light = false;
# MAGIC function f2Toggle() {
# MAGIC   f2Light = !f2Light;
# MAGIC }
# MAGIC function f2Select(id) {
# MAGIC   var code = document.getElementById('f2-code'), card = document.getElementById('f2-explain-card');
# MAGIC   document.querySelectorAll('.f2-card').forEach(function(b){ b.classList.toggle('active', parseInt(b.dataset.id)===id); });
# MAGIC   if (f2Current === id) { code.classList.remove('has-highlight'); card.classList.remove('visible'); document.querySelectorAll('.f2-card').forEach(function(b){b.classList.remove('active');}); f2Current=null; return; }
# MAGIC   f2Current = id; var c = F2_DATA[id];
# MAGIC   code.classList.add('has-highlight');
# MAGIC   code.querySelectorAll('.line').forEach(function(ln){ ln.classList.toggle('hl', ln.dataset.g===String(id)); });
# MAGIC   card.style.borderTopColor = c.color;
# MAGIC   card.innerHTML = '<div style="font-size:15pt;font-weight:700;margin-bottom:10px;color:'+c.color+';">'+c.title+'</div><div>'+c.text+'</div>';
# MAGIC   card.classList.add('visible');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# Find error traces
error_traces = mlflow.search_traces(
    locations=[experiment_id],
    filter_string="trace.status = 'ERROR'",
    max_results=100
)
print(f"Error traces found: {len(error_traces)}")

# Find slow traces (over 5 seconds)
slow_traces = mlflow.search_traces(
    locations=[experiment_id],
    filter_string="trace.execution_time_ms > 5000",
    max_results=100
)
print(f"Slow traces (>5s): {len(slow_traces)}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E3. SQL-Based Trace Queries
# MAGIC
# MAGIC For dashboard-scale analysis, you can also query traces stored in Unity Catalog tables using Databricks SQL (through the MLflow trace views).
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><code>mlflow.search_traces()</code>  is best for interactive, programmatic analysis and recent data, and it can optionally use a Databricks SQL warehouse for performance on large trace datasets. For BI dashboards and scheduled reporting, build queries on top of the Unity Catalog trace tables/views and consider materializing or optimizing them for large historical windows.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC 1. **Manual tracing** (`@mlflow.trace`) with semantic `span_type` labels adds visibility into custom logic: tool calls, retrieval, business logic
# MAGIC 2. **Span attributes** attach production context (user ID, session ID, app version) that is critical for debugging
# MAGIC 3. **Multi-turn tracing** uses a shared `session_id` attribute to group conversation turns
# MAGIC 4. **Trace search** (`mlflow.search_traces()`) enables programmatic filtering by status, timing, and attributes
# MAGIC 5. **SQL-based queries** on Unity Catalog trace views support dashboard-scale analysis for production monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC