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
# MAGIC %md
# MAGIC # Lab - Assessing and Fixing Agent Tools
# MAGIC #### Duration: 30 minutes
# MAGIC ## Overview
# MAGIC In this lab, you will take a Unity Catalog function that works correctly but is poorly documented, see how it confuses an agent, and fix it. An agent decides when to call a tool, and what to pass it, based only on the tool's name, docstring, and parameter descriptions, so documentation is the interface between an agent and its tools. You will rewrite the function's documentation, re-register it to Unity Catalog, and verify the improvement through MCP and by testing it with a model.
# MAGIC
# MAGIC Each step has a **TODO** with `<FILL_IN>` placeholders. The instructions above each TODO tell you what to fill in, and you can reveal the answer if you get stuck.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lab, you will be able to:
# MAGIC 1. Identify documentation problems that keep an agent from using a UC function correctly
# MAGIC 2. Rewrite a Python function with descriptive parameter names, type hints, and a structured docstring
# MAGIC 3. Re-register an updated Python function to Unity Catalog with `DatabricksFunctionClient`
# MAGIC 4. Discover and call a UC function through a Managed MCP server
# MAGIC 5. Compare the arguments a model passes to a tool before and after improving the tool's documentation

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
# MAGIC ## A. Environment Setup
# MAGIC
# MAGIC Run the next cell to install the libraries for this lab and set up your catalog, schema, data, and the poorly documented `est_listing_value` function you will fix.

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-3

# COMMAND ----------

# DBTITLE 1,A1. Lab Scenario
# MAGIC %md-sandbox
# MAGIC ### A1. Lab Scenario
# MAGIC
# MAGIC <div class="scn3"><style>.scn3{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#0b2026;width:100%;max-width:100%;box-sizing:border-box;margin:0;padding:8px}.scn3 .scen{background:#FFFFFF;color:#0b2026;padding:26px 30px;border-radius:12px;box-sizing:border-box;overflow:hidden;box-shadow:0 2px 16px rgba(27,49,57,0.10),0 1px 4px rgba(27,49,57,0.06);border:1px solid rgba(27,49,57,0.08)}.scn3 .kicker{font-size:14px;text-transform:uppercase;letter-spacing:1.2px;color:#5A6F77;margin-bottom:10px;font-weight:700;display:flex;align-items:center;gap:10px}.scn3 .body{font-size:15px;line-height:1.7;color:#0b2026}.scn3 .body p{margin:0 0 14px}.scn3 .body p:last-child{margin:0}.scn3 .tools{background:#F9F7F4;border:1px solid rgba(27,49,57,0.10);border-radius:8px;padding:12px 16px;margin:0 0 14px 0;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:14px;line-height:1.9}.scn3 .fn{color:#0B8F5F}.scn3 .arg{color:#5A6F77}.scn3 .dsc{color:#5A6F77;opacity:.8}.scn3 code{background:rgba(27,49,57,0.07);padding:2px 6px;border-radius:4px;font-size:14px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#0b2026}.scn3 .eq{background:rgba(0,169,114,.10);border-left:4px solid #00A972;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0 14px;font-size:14px;font-style:italic;line-height:1.6;color:#0b2026}.scn3 .eq-q{background:#FFF6F4;border-left:4px solid #FF5F46;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0 14px;font-size:14px;line-height:1.6;color:#0b2026}</style><div class="scen">
# MAGIC <div class="kicker"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#5A6F77" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect x="4" y="8" width="16" height="12" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M9 13v2"/><path d="M15 13v2"/></svg>Scenario</div>
# MAGIC <div class="body">
# MAGIC <p>Your team recently registered a Unity Catalog function called <code>est_listing_value</code> to help an agent estimate the annual revenue of an Airbnb listing. The logic is correct &#x2014; but when you give it to a model as a tool, the model consistently misuses it: passing values to the wrong parameters, inventing values the tool can't use, and returning a confident but wrong answer.</p>
# MAGIC
# MAGIC <div class="tools"><div><span class="fn">est_listing_value</span><span class="arg">(p1, p2, p3)</span> <span class="dsc">&#x2014; "Calculates value."</span></div></div>
# MAGIC
# MAGIC <p>The problem isn't the code &#x2014; it's the documentation. The function has meaningless parameter names (<code>p1</code>, <code>p2</code>, <code>p3</code>), a two-word description, and a docstring that is out of date: it says <code>p1</code> is the price, but the code uses <code>p1</code> as the number of bedrooms. An agent decides <strong>when</strong> to call a tool, <strong>what</strong> to pass it, and <strong>how</strong> to interpret the result based entirely on its name, docstring, and parameter descriptions. When those are unclear, the agent guesses. When they are wrong, the agent confidently follows them.</p>
# MAGIC
# MAGIC <p>A user sends a perfectly reasonable request:</p>
# MAGIC <div class="eq">"I have a 2-bedroom entire-home Airbnb in SF priced at $250/night. What could I expect to make per year?"</div>
# MAGIC
# MAGIC <p>But the agent sees only <code>p1: price</code>, <code>p2: count</code>, <code>p3: market</code> &#x2014; and produces something like:</p>
# MAGIC <div class="eq-q"><strong>Agent:</strong> Calls <code>est_listing_value(p1=250, p2=2, p3="San Francisco")</code> &#x2014; passing the price where the code expects the bedroom count, and a city where it expects a room type. Nothing fails: the tool reports about $9,798 a year instead of about $88,148.</div>
# MAGIC
# MAGIC <p>You will diagnose the documentation problems, rewrite the function with descriptive names, type hints, and a structured docstring, re-register it to Unity Catalog, and verify the improvement through MCP and by testing it with a model.</p>
# MAGIC </div>
# MAGIC </div></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Inspect the Airbnb Dataset
# MAGIC
# MAGIC As part of the classroom setup, the Airbnb dataset has been processed and stored as a Delta table within Unity Catalog. Run the next cell to preview the first few rows.

# COMMAND ----------

df = spark.read.table('sf_airbnb_listings')
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### A3. Initialize the Databricks Function Client
# MAGIC
# MAGIC **TODO:** Initialize the `DatabricksFunctionClient` for serverless compute. This client lets you register and execute UC functions programmatically.

# COMMAND ----------

## Import and initialize the DatabricksFunctionClient for serverless compute
from unitycatalog.ai.core.databricks import <FILL_IN>

client = <FILL_IN>

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
# MAGIC from unitycatalog.ai.core.databricks import DatabricksFunctionClient
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# DBTITLE 1,Section B Intro
# MAGIC %md
# MAGIC ## B. Examine a Poorly Documented Agent Tool
# MAGIC
# MAGIC As part of the classroom setup, a Python function called `est_listing_value` has been registered to Unity Catalog. This function is intended to estimate the annual revenue of an Airbnb listing based on its characteristics, but its documentation is **vague and out of date**.
# MAGIC
# MAGIC Let's look at the function as it currently exists.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Review the Bad Function
# MAGIC
# MAGIC The function defined below has already been registered to Unity Catalog. Notice the problems: meaningless parameter names, a description that says almost nothing, and parameter descriptions that no longer match the code. Click each pill at the top to highlight a part of the code and read what's wrong with it.
# MAGIC
# MAGIC <div id="py-code-explorer"></div>
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
# MAGIC     mountId: "py-code-explorer",
# MAGIC     title: "PYTHON",
# MAGIC     hotspots: [
# MAGIC       { title: "function name", startLine: 1, endLine: 1,
# MAGIC         description: "The tool name is the only helpful clue the agent gets. It hints at an estimate, but not of what, in what units, or for whom." },
# MAGIC       { title: "meaningless parameters", startLine: 2, endLine: 4,
# MAGIC         description: "<code>p1</code>, <code>p2</code>, and <code>p3</code> tell the agent nothing. The types help only a little: <code>p3</code> must be text, but the two floats are interchangeable." },
# MAGIC       { title: "return type", startLine: 5, endLine: 5,
# MAGIC         description: "<code>-> str</code> says the tool returns text, but not what the text contains." },
# MAGIC       { title: "two-word description", startLine: 6, endLine: 8,
# MAGIC         description: "<code>Calculates value.</code> is the tool description the model reads. It doesn't say what is calculated, or when to use the tool." },
# MAGIC       { title: "out-of-date Args", startLine: 9, endLine: 12,
# MAGIC         description: "These descriptions no longer match the code: <code>p1</code> is described as the price, but the code uses it as the bedroom count. <code>count</code> and <code>market</code> invite guesses, and a model will likely put a city in <code>p3</code>." },
# MAGIC       { title: "useless Returns", startLine: 13, endLine: 16,
# MAGIC         description: "<code>str: result</code> repeats the type and adds nothing about the value or its units." },
# MAGIC       { title: "where the docs bite", startLine: 18, endLine: 19,
# MAGIC         description: "The code multiplies <code>p2</code> as a nightly price and treats <code>p1</code> as a bedroom count, the reverse of what the docstring says." },
# MAGIC       { title: "silent fallback", startLine: 20, endLine: 25,
# MAGIC         description: "Any <code>p3</code> that isn't a known room type, such as <code>'San Francisco'</code>, silently falls through to the lowest multiplier instead of raising an error." },
# MAGIC       { title: "confident result", startLine: 26, endLine: 27,
# MAGIC         description: "The tool always returns a dollar amount, so a wrong call produces a believable wrong answer instead of an error." }
# MAGIC     ],
# MAGIC     source: "def est_listing_value(\n    p1: float,\n    p2: float,\n    p3: str\n) -> str:\n    \"\"\"\n    Calculates value.\n\n    Args:\n        p1: price\n        p2: count\n        p3: market\n\n    Returns:\n        str: result\n    \"\"\"\n\n    base_value = p2 * 365 * 0.7  # assume 70% occupancy\n    bedroom_factor = 1 + (p1 - 1) * 0.15\n    if p3.lower() == 'entire home/apt':\n        multiplier = 1.2\n    elif p3.lower() == 'private room':\n        multiplier = 0.8\n    else:\n        multiplier = 0.5\n    estimated = base_value * bedroom_factor * multiplier\n    return f\"Estimated annual revenue: ${estimated:,.2f}\""
# MAGIC   };
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

# MAGIC %md-sandbox
# MAGIC ### B2. See How a Model Uses the Tool
# MAGIC
# MAGIC Give the poorly documented tool to a model and see what it does with a realistic request. Run the next cell to connect to the model and define `ask_model`, the same helper as in the *Building Agent Tools on Databricks* demo, with one change: when you offer tools, it requires the model to call one, so you always see the arguments it chooses. It sends a question to the model, offers it the tools you name, and prints the tool calls the model wants to make without running them.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC We won't dive into this code much. The <em>Introduction to OpenAI Agents SDK on Databricks</em> demo, later in the course, covers connecting to a model and building agents in detail. For now, focus on the output.<br/><br/>The code calls a <strong>model service</strong>, <code>agent-training-model</code>, that the classroom setup created in your catalog and schema. A model service is our way of connecting to a model endpoint. We'll cover the details later in the course.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

import json
import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from databricks_openai import DatabricksOpenAI

nest_asyncio.apply()
ws = WorkspaceClient()
test_mcp_client = DatabricksMCPClient(
    server_url=f"{ws.config.host}/api/2.0/mcp/functions/{catalog_name}/{schema_name}",
    workspace_client=ws,
)

# A model service is our way of connecting to a model endpoint (covered later in the course)
model_service = f"{catalog_name}.{schema_name}.agent-training-model"
llm_client = DatabricksOpenAI(workspace_client=ws, use_ai_gateway=True)

def ask_model(question, tool_names=None):
    """Send a question to the model, optionally offering UC functions as tools.

    Prints the model's answer, or the tool calls it wants to make and the
    arguments it would pass. The tools are not run.
    """
    offered = [t for t in test_mcp_client.list_tools() if t.name.split("__")[-1] in (tool_names or [])]
    tool_specs = [
        {"type": "function",
         "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
        for t in offered
    ]
    # Require a tool call so you always see the arguments the model picks
    extra = {"tools": tool_specs, "tool_choice": "required"} if tool_specs else {}
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
# MAGIC Now ask the question from the scenario. Look closely at the arguments. The model reads `p1: price`, `p2: count`, and `p3: market`, so it will likely pass the $250 price as `p1`, which the code treats as the bedroom count, and a city such as `"San Francisco"` as `p3`. Nothing would fail if the tool ran: a call like that returns about $9,798 a year, when the right answer is about $88,148. The model can't tell, because it trusts the documentation. The exact call can vary from run to run.

# COMMAND ----------

question = "I have a 2-bedroom entire-home Airbnb listing in San Francisco priced at $250. What could I expect to make from it per year?"

ask_model(question, tool_names=["est_listing_value"])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Expected output</strong>
# MAGIC <pre style="margin: 10px 0; padding: 10px 12px; background: #ffffff; border: 1px solid #C6DCEA; border-radius: 6px; font-size: 12px; white-space: pre-wrap;">The model wants to call:
# MAGIC   - est_listing_value({'p1': 250, 'p2': 2, 'p3': 'San Francisco'})</pre>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC The values are in the wrong parameters. The $250 price went into <code>p1</code>, which the code uses as the bedroom count, and a city went into <code>p3</code>, where the code expects a room type. If this call ran, the tool would return <strong>$9,798.42</strong>, a believable but wrong estimate. Your exact output can vary, for example <code>250.0</code> instead of <code>250</code>, or a different value for <code>p3</code>, but expect the price in <code>p1</code> and something other than a room type in <code>p3</code>.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Fix the Function Description
# MAGIC
# MAGIC The function logic is correct -- the problem is entirely in the documentation. An agent reads the function name, docstring, and parameter descriptions to decide:
# MAGIC 1. **When** to call the function
# MAGIC 2. **What** values to pass for each parameter
# MAGIC 3. **How** to interpret the result
# MAGIC
# MAGIC Without clear documentation, the agent is guessing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Rewrite the Function with Proper Documentation
# MAGIC
# MAGIC **TODO:** Redefine the function with:
# MAGIC - Descriptive, typed parameters: rename `p1`, `p2`, and `p3` to `num_bedrooms`, `nightly_price_usd`, and `room_type`, and give each a type hint. The function body already uses these names, and the test and MCP call below pass them by name.
# MAGIC - A clear docstring explaining the function's purpose and when to use it, and that matches what the code actually does
# MAGIC - Detailed parameter descriptions with types, units, and examples
# MAGIC - A clear return value description

# COMMAND ----------

## Redefine the function with an improved docstring
def est_listing_value(
    <FILL_IN>,
    <FILL_IN>,
    <FILL_IN>
) -> str:
    """
    <FILL_IN>
    """
    base_value = nightly_price_usd * 365 * 0.7  # assume 70% occupancy
    bedroom_factor = 1 + (num_bedrooms - 1) * 0.15
    if room_type.lower() == 'entire home/apt':
        multiplier = 1.2
    elif room_type.lower() == 'private room':
        multiplier = 0.8
    else:
        multiplier = 0.5
    estimated = base_value * bedroom_factor * multiplier
    return f"Estimated annual revenue: ${estimated:,.2f}"

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
# MAGIC def est_listing_value(
# MAGIC     nightly_price_usd: float,
# MAGIC     num_bedrooms: float,
# MAGIC     room_type: str
# MAGIC ) -> str:
# MAGIC     """
# MAGIC     Estimate the annual revenue for a San Francisco Airbnb listing based on
# MAGIC     its nightly price, number of bedrooms, and room type. Assumes 70%
# MAGIC     occupancy year-round.
# MAGIC
# MAGIC     Use this tool when a user asks how much a listing could earn per year.
# MAGIC
# MAGIC     Args:
# MAGIC         nightly_price_usd: The nightly price of the listing in US dollars (e.g. 250.0).
# MAGIC         num_bedrooms: The number of bedrooms in the listing (e.g. 1, 2, 3). A
# MAGIC             studio should be passed as 1. Each additional bedroom beyond
# MAGIC             the first increases the estimate by 15%.
# MAGIC         room_type: The Airbnb room type. Must be exactly one of:
# MAGIC             "Entire home/apt", "Private room", or "Shared room".
# MAGIC
# MAGIC     Returns:
# MAGIC         A string describing the estimated annual revenue in USD.
# MAGIC     """
# MAGIC     base_value = nightly_price_usd * 365 * 0.7  # assume 70% occupancy
# MAGIC     bedroom_factor = 1 + (num_bedrooms - 1) * 0.15
# MAGIC     if room_type.lower() == 'entire home/apt':
# MAGIC         multiplier = 1.2
# MAGIC     elif room_type.lower() == 'private room':
# MAGIC         multiplier = 0.8
# MAGIC     else:
# MAGIC         multiplier = 0.5
# MAGIC     estimated = base_value * bedroom_factor * multiplier
# MAGIC     return f"Estimated annual revenue: ${estimated:,.2f}"
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Test the Improved Function Locally
# MAGIC
# MAGIC Before re-registering, verify the function works correctly at the notebook level.

# COMMAND ----------

result = est_listing_value(nightly_price_usd=250.0, num_bedrooms=2, room_type="entire home/apt")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Re-register the Function to Unity Catalog
# MAGIC
# MAGIC **TODO:** Use `DatabricksFunctionClient` to re-register the improved function with `replace=True`.

# COMMAND ----------

function_info = client.<FILL_IN>(
    func=<FILL_IN>,
    catalog=<FILL_IN>,
    schema=<FILL_IN>,
    replace=<FILL_IN>
)

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
# MAGIC function_info = client.create_python_function(
# MAGIC     func=est_listing_value,
# MAGIC     catalog=catalog_name,
# MAGIC     schema=schema_name,
# MAGIC     replace=True
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Verify the Improved Tool via MCP
# MAGIC
# MAGIC In the demo, you saw how MCP allows agents to discover and call UC functions. Let's use MCP to verify that our re-registered function has the improved description and works correctly.

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Discover the Tool
# MAGIC
# MAGIC **TODO:** Initialize the MCP client and list available tools. Verify that `est_listing_value` appears with the improved description.

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

# TODO
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

ws = WorkspaceClient()
workspace_host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"

mcp_url = f"{workspace_host}/api/2.0/mcp/functions/{catalog_name}/{schema_name}"
mcp_client = DatabricksMCPClient(server_url=mcp_url, workspace_client=ws)

tools = mcp_client.<FILL_IN>

for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"  Description: {tool.description[:150]}")
    print()

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
# MAGIC import nest_asyncio
# MAGIC nest_asyncio.apply()
# MAGIC
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_mcp import DatabricksMCPClient
# MAGIC
# MAGIC ws = WorkspaceClient()
# MAGIC workspace_host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
# MAGIC
# MAGIC mcp_url = f"{workspace_host}/api/2.0/mcp/functions/{catalog_name}/{schema_name}"
# MAGIC mcp_client = DatabricksMCPClient(server_url=mcp_url, workspace_client=ws)
# MAGIC
# MAGIC tools = mcp_client.list_tools()
# MAGIC
# MAGIC for tool in tools:
# MAGIC     print(f"Tool: {tool.name}")
# MAGIC     print(f"  Description: {tool.description[:150]}")
# MAGIC     print()
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Call the Tool via MCP
# MAGIC
# MAGIC **TODO:** Execute `est_listing_value` through MCP to verify it works end-to-end.

# COMMAND ----------

result = <FILL_IN>

print(f"MCP Result: {result}")

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
# MAGIC result = mcp_client.call_tool(f"{catalog_name}__{schema_name}__est_listing_value", 
# MAGIC {
# MAGIC     "nightly_price_usd": 250.0, 
# MAGIC     "num_bedrooms":2, 
# MAGIC     "room_type":"entire home/apt"
# MAGIC     }
# MAGIC  )
# MAGIC
# MAGIC print(f"MCP Result: {result}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Validate the Improved Tool with a Model
# MAGIC
# MAGIC Ask the model the same question as in B2. `ask_model` reads the tool's definition from the MCP server each time, so it now sees your improved names, type hints, and docstring.

# COMMAND ----------

ask_model(question, tool_names=["est_listing_value"])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Expected output</strong>
# MAGIC <pre style="margin: 10px 0; padding: 10px 12px; background: #ffffff; border: 1px solid #C6DCEA; border-radius: 6px; font-size: 12px; white-space: pre-wrap;">The model wants to call:
# MAGIC   - est_listing_value({'nightly_price_usd': 250, 'num_bedrooms': 2, 'room_type': 'Entire home/apt'})</pre>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Each value is now in the right parameter. If your docstring lists the allowed room types, as the answer does, <code>room_type</code> is one of them. If this call ran, the tool would return <strong>$88,147.50</strong>, the correct estimate. Numbers may appear as <code>250.0</code> and <code>2.0</code>.
# MAGIC </p>
# MAGIC </div>
# MAGIC
# MAGIC Compare the arguments with B2. The model now passes `nightly_price_usd`, `num_bedrooms`, and `room_type`, with values taken from the question, and uses a `room_type` from the list your docstring allows. The improved documentation gives the model the context it needs to use the tool correctly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab you learned that **documentation is the interface** between an agent and its tools. The same function logic can be useless or powerful depending on how well it's documented. You also verified your tools work through MCP -- the same protocol that agents use in production to discover and call tools.
# MAGIC
# MAGIC Here's what changed between B2 and E for the same question:
# MAGIC
# MAGIC | | Bad tool (B2) | Fixed tool (E) |
# MAGIC |---|---|---|
# MAGIC | **Call the model makes** | `est_listing_value(p1=250, p2=2, p3="San Francisco")` | `est_listing_value(nightly_price_usd=250, num_bedrooms=2, room_type="Entire home/apt")` |
# MAGIC | **What the tool would return** | $9,798.42 (wrong) | $88,147.50 (correct) |
# MAGIC | **Why** | The out-of-date docstring sent each value to the wrong parameter | The names, types, and docstring match what the code does |
# MAGIC
# MAGIC Key takeaways:
# MAGIC - Poor parameter names and docstrings make tools unusable by agents
# MAGIC - Out-of-date documentation is worse than none: the agent follows it confidently and gets a believable wrong answer
# MAGIC - Good documentation includes: clear names, detailed descriptions with examples, and business context
# MAGIC - MCP provides a standard way for agents to discover and call UC functions
# MAGIC - Always test tools via MCP, and with a model, before integrating them with agent code

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>