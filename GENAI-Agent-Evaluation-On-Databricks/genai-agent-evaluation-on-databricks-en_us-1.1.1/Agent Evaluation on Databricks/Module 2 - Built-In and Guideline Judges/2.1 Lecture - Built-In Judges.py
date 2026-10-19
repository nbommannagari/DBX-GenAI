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
# MAGIC # Lecture - Built-In Judges
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC MLflow provides research-validated built-in judges for common evaluation criteria like correctness, relevance, safety, and retrieval quality. This lecture introduces the spectrum of available judge types and takes a deep dive into built-in judges: the fastest way to start evaluating your agents.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Describe the spectrum of evaluation judge types available in MLflow
# MAGIC - Identify appropriate built-in judges for common evaluation criteria
# MAGIC - Understand which judges require ground truth and which do not
# MAGIC - Apply built-in judges using `mlflow.genai.evaluate()`

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. Judge Types Overview

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### A1. Evaluation Judge Spectrum
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:0;margin:18px 0;font-family:system-ui,sans-serif;flex-wrap:nowrap;overflow-x:auto;padding-bottom:8px;">
# MAGIC   <div style="flex:1;min-width:145px;background:#e8e8e8;color:#333;padding:14px;border-radius:8px 0 0 8px;border:1px solid #ccc;text-align:center;">
# MAGIC     <div style="font-size:13px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:0.5px;">Minimal Customization</div>
# MAGIC     <div style="font-size:15px;font-weight:700;color:#2272B4;margin:8px 0 4px;">Built-in Judges</div>
# MAGIC     <div style="font-size:13px;color:#555;">Correctness, Safety, RetrievalGroundedness</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:145px;background:#E3F2FD;color:#333;padding:14px;border:1px solid #90CAF9;text-align:center;">
# MAGIC     <div style="font-size:13px;font-weight:600;color:#1565C0;text-transform:uppercase;letter-spacing:0.5px;">Moderate</div>
# MAGIC     <div style="font-size:15px;font-weight:700;color:#00A972;margin:8px 0 4px;">Guideline Judges</div>
# MAGIC     <div style="font-size:13px;color:#555;">Natural-language rules for style, factuality</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:145px;background:#E8F5E9;color:#333;padding:14px;border:1px solid #A5D6A7;text-align:center;">
# MAGIC     <div style="font-size:13px;font-weight:600;color:#2E7D32;text-transform:uppercase;letter-spacing:0.5px;">Full</div>
# MAGIC     <div style="font-size:15px;font-weight:700;color:#FF5F46;margin:8px 0 4px;">Custom LLM Judges</div>
# MAGIC     <div style="font-size:13px;color:#555;">Scores, categories, booleans via make_judge()</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:145px;background:#FFF3E0;color:#333;padding:14px;border:1px solid #FFCC80;text-align:center;">
# MAGIC     <div style="font-size:13px;font-weight:600;color:#E65100;text-transform:uppercase;letter-spacing:0.5px;">Full (Deterministic)</div>
# MAGIC     <div style="font-size:15px;font-weight:700;color:#1B5162;margin:8px 0 4px;">Code-Based Scorers</div>
# MAGIC     <div style="font-size:13px;color:#555;">Exact match, format validation, metrics</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:145px;background:#F3E5F5;color:#333;padding:14px;border-radius:0 8px 8px 0;border:1px solid #CE93D8;text-align:center;">
# MAGIC     <div style="font-size:13px;font-weight:600;color:#7B1FA2;text-transform:uppercase;letter-spacing:0.5px;">Full (External)</div>
# MAGIC     <div style="font-size:15px;font-weight:700;color:#7B61FF;margin:8px 0 4px;">Third-Party Scorers</div>
# MAGIC     <div style="font-size:13px;color:#555;">Open-source evaluation frameworks</div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Built-In Judges for Common Criteria

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Research-Validated Judges
# MAGIC
# MAGIC MLflow provides research-validated judges for common evaluation tasks. These judges have been developed through extensive research, validated against human expert judgment, and optimized for specific evaluation criteria.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### B2. Common Built-in Judges
# MAGIC
# MAGIC <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:12px;margin:18px 0;font-family:system-ui,sans-serif;">
# MAGIC   <div style="flex:1;min-width:220px;max-width:340px;border:2px solid #2272B4;border-radius:10px;padding:14px;background:#fff;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#2272B4;font-family:monospace;">Correctness</div>
# MAGIC     <div style="font-size:13px;color:#333;margin-top:6px;">Factually correct vs. ground truth</div>
# MAGIC     <div style="margin-top:8px;font-size:13px;background:#E3F2FD;border-radius:4px;padding:4px 8px;display:inline-block;color:#2272B4;font-weight:600;">Requires: expectations</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:220px;max-width:340px;border:2px solid #00A972;border-radius:10px;padding:14px;background:#fff;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#00A972;font-family:monospace;">Safety</div>
# MAGIC     <div style="font-size:13px;color:#333;margin-top:6px;">Free from harmful, offensive, or unsafe content</div>
# MAGIC     <div style="margin-top:8px;font-size:13px;background:#E8F5E9;border-radius:4px;padding:4px 8px;display:inline-block;color:#00A972;font-weight:600;">No ground truth needed</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:220px;max-width:340px;border:2px solid #FF5F46;border-radius:10px;padding:14px;background:#fff;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#FF5F46;font-family:monospace;">RelevanceToQuery</div>
# MAGIC     <div style="font-size:13px;color:#333;margin-top:6px;">Response directly addresses the user's query</div>
# MAGIC     <div style="margin-top:8px;font-size:13px;background:#FFF3E0;border-radius:4px;padding:4px 8px;display:inline-block;color:#FF5F46;font-weight:600;">No ground truth needed</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:220px;max-width:340px;border:2px solid #1B5162;border-radius:10px;padding:14px;background:#fff;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#1B5162;font-family:monospace;">RetrievalGroundedness</div>
# MAGIC     <div style="font-size:13px;color:#333;margin-top:6px;">Response is grounded in retrieved context, not hallucinated</div>
# MAGIC     <div style="margin-top:8px;font-size:13px;background:#E8F0F3;border-radius:4px;padding:4px 8px;display:inline-block;color:#1B5162;font-weight:600;">Requires: traces with RETRIEVER spans</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:220px;max-width:340px;border:2px solid #7B61FF;border-radius:10px;padding:14px;background:#fff;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#7B61FF;font-family:monospace;">RetrievalSufficiency</div>
# MAGIC     <div style="font-size:13px;color:#333;margin-top:6px;">Retrieved context has enough info for ground-truth answer</div>
# MAGIC     <div style="margin-top:8px;font-size:13px;background:#F3E8FF;border-radius:4px;padding:4px 8px;display:inline-block;color:#7B61FF;font-weight:600;">Requires: expectations + RETRIEVER spans</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:220px;max-width:340px;border:2px solid #618794;border-radius:10px;padding:14px;background:#fff;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#618794;font-family:monospace;">RetrievalRelevance</div>
# MAGIC     <div style="font-size:13px;color:#333;margin-top:6px;">Retrieved documents are relevant to the query</div>
# MAGIC     <div style="margin-top:8px;font-size:13px;background:#EEEDE9;border-radius:4px;padding:4px 8px;display:inline-block;color:#618794;font-weight:600;">Requires: traces with RETRIEVER spans</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1; font-size: 1.0em;">Full list of built-in judges</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333; font-size: 14px;">MLflow provides additional built-in judges including <code>Guidelines</code>, <code>ExpectationsGuidelines</code>, and multi-turn scorers (<code>ConversationCompleteness</code>, <code>UserFrustration</code>, etc.). See the complete reference: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers#built-in-llm-judges" target="_blank" style="color:#1976d2;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/scorers" target="_blank" style="color:#1976d2;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/scorers#built-in-llm-judges" target="_blank" style="color:#1976d2;">GCP</a>)</p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### B3. Example Usage Pattern
# MAGIC
# MAGIC A typical evaluation has four steps. Click each step to see the code:
# MAGIC
# MAGIC <div style="max-width:1000px;margin:0 auto;font-family:sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .eval-step-row { display:flex; gap:8px; margin-bottom:0; flex-wrap:wrap; }
# MAGIC .eval-step-box {
# MAGIC   flex:1; min-width:160px; background:#F9F7F4; border-top:6px solid transparent;
# MAGIC   border-left:2px solid transparent; border-right:2px solid transparent; border-bottom:2px solid transparent;
# MAGIC   border-radius:8px; padding:12px 10px; text-align:center; cursor:pointer; user-select:none;
# MAGIC   transition:transform 0.12s;
# MAGIC }
# MAGIC .eval-step-box:hover { transform:translateY(-2px); }
# MAGIC .eval-step-box.active { background:#fff; border-left-color:var(--sc); border-right-color:var(--sc); border-bottom-color:var(--sc); }
# MAGIC .eval-step-title { display:block; font-size:15px; font-weight:700; color:#0b2026; line-height:1.3; pointer-events:none; }
# MAGIC .eval-step-sub { display:block; font-size:13px; color:#618794; margin-top:4px; pointer-events:none; }
# MAGIC .eval-detail-wrap { overflow:hidden; max-height:0; opacity:0; transition:max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease; margin-top:0; }
# MAGIC .eval-detail-wrap.open { max-height:800px; opacity:1; margin-top:12px; }
# MAGIC .eval-detail-card { background:#F9F7F4; border-radius:10px; padding:20px; border-top:6px solid #ccc; }
# MAGIC .eval-code-block { border-left:4px solid; border-radius:0 6px 6px 0; padding:14px; margin-top:10px; font-size:13px; line-height:1.5; font-family:monospace; white-space:pre; overflow-x:auto; }
# MAGIC </style>
# MAGIC
# MAGIC <div style="background:#F8F9FC;border:3px solid #1B5162;border-radius:10px;padding:20px;">
# MAGIC
# MAGIC <div class="eval-step-row">
# MAGIC   <span class="eval-step-box active" data-id="0" onclick="selectEvalStep(0)" style="--sc:#2272B4;border-top-color:#2272B4;">
# MAGIC     <span class="eval-step-title">1. Define Agent</span>
# MAGIC     <span class="eval-step-sub">Traced function</span>
# MAGIC   </span>
# MAGIC   <span class="eval-step-box" data-id="1" onclick="selectEvalStep(1)" style="--sc:#00A972;border-top-color:#00A972;">
# MAGIC     <span class="eval-step-title">2. Choose Scorer</span>
# MAGIC     <span class="eval-step-sub">Built-in judge</span>
# MAGIC   </span>
# MAGIC   <span class="eval-step-box" data-id="2" onclick="selectEvalStep(2)" style="--sc:#FF5F46;border-top-color:#FF5F46;">
# MAGIC     <span class="eval-step-title">3. Build Dataset</span>
# MAGIC     <span class="eval-step-sub">Inputs + expectations</span>
# MAGIC   </span>
# MAGIC   <span class="eval-step-box" data-id="3" onclick="selectEvalStep(3)" style="--sc:#1B5162;border-top-color:#1B5162;">
# MAGIC     <span class="eval-step-title">4. Evaluate</span>
# MAGIC     <span class="eval-step-sub">mlflow.genai.evaluate()</span>
# MAGIC   </span>
# MAGIC </div>
# MAGIC
# MAGIC <div class="eval-detail-wrap" id="eval-detail-wrap">
# MAGIC   <div class="eval-detail-card" id="eval-detail-card"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var EVAL_STEPS = [
# MAGIC   {
# MAGIC     title: "1. Define Your Agent",
# MAGIC     color: "#2272B4",
# MAGIC     bg: "rgba(34,114,180,0.10)",
# MAGIC     desc: "Your agent function should accept the keys from the <code>inputs</code> dictionary as keyword arguments and emit one trace per call. If your function doesn't already produce a trace, MLflow automatically applies <code>@mlflow.trace</code> for you. The return value can be any type (string, dict, list, etc.).",
# MAGIC     code: '@mlflow.trace\ndef my_agent(question: str) -> str:\n    """Your agent function."""\n    # Your agent logic here\n    return "..."'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "2. Choose a Scorer",
# MAGIC     color: "#00A972",
# MAGIC     bg: "rgba(0,169,114,0.10)",
# MAGIC     desc: "Import and instantiate a built-in judge. You can optionally override the default judge model with the <code>model</code> parameter.",
# MAGIC     code: 'from mlflow.genai.scorers import Correctness\n\ncorrectness_eval = Correctness(\n    model=\"databricks:/foundation-model-endpoint\"\n)'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "3. Build Evaluation Dataset",
# MAGIC     color: "#FF5F46",
# MAGIC     bg: "rgba(255,95,70,0.10)",
# MAGIC     desc: "Each record contains <code>inputs</code> (matching your agent's kwargs) and optionally <code>expectations</code> with ground truth facts or guidelines.",
# MAGIC     code: 'eval_data = [\n    {\"inputs\": {\"question\": \"What is MLflow?\"},\n     \"expectations\": {\"expected_facts\": [\"MLflow is an open-source platform\"]}},\n    {\"inputs\": {\"question\": \"What is Spark?\"},\n     \"expectations\": {\"expected_facts\": [\"Spark is a data engine\"]}},\n]'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "4. Run Evaluation",
# MAGIC     color: "#1B5162",
# MAGIC     bg: "rgba(27,81,98,0.10)",
# MAGIC     desc: "Pass your dataset, agent function, and scorers to <code>mlflow.genai.evaluate()</code>. Results are logged to MLflow automatically.",
# MAGIC     code: 'results = mlflow.genai.evaluate(\n    data=eval_data,\n    predict_fn=my_agent,\n    scorers=[correctness_eval],\n)'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var evalCurrent = null;
# MAGIC
# MAGIC function selectEvalStep(id) {
# MAGIC   var wrap = document.getElementById('eval-detail-wrap');
# MAGIC   var card = document.getElementById('eval-detail-card');
# MAGIC   var s = EVAL_STEPS[id];
# MAGIC
# MAGIC   document.querySelectorAll('.eval-step-box').forEach(function(b) {
# MAGIC     b.classList.toggle('active', parseInt(b.dataset.id, 10) === id);
# MAGIC   });
# MAGIC
# MAGIC   if (evalCurrent === id) {
# MAGIC     wrap.classList.remove('open');
# MAGIC     document.querySelectorAll('.eval-step-box').forEach(function(b) { b.classList.remove('active'); });
# MAGIC     evalCurrent = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   evalCurrent = id;
# MAGIC   card.style.borderTopColor = s.color;
# MAGIC   card.innerHTML =
# MAGIC     '<div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#0b2026;">' + s.title + '</div>'
# MAGIC     + '<div style="font-size:14px;line-height:1.6;color:#333;margin-bottom:10px;">' + s.desc + '</div>'
# MAGIC     + '<div class="eval-code-block" style="background:' + s.bg + ';border-color:' + s.color + ';">' 
# MAGIC     + '<code>' + s.code.replace(/\n/g, '<br>') + '</code></div>';
# MAGIC
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC
# MAGIC selectEvalStep(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to explore the full list of built-in judges and their use cases? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-2-1" style="flex: 1;">What built-in LLM judges are available in MLflow and when should I use each one?</span>
# MAGIC     <button onclick="
# MAGIC       var text = document.getElementById('genie-2-1').innerText;
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
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Conclusion
# MAGIC
# MAGIC Built-in judges give you the fastest path to evaluating your agents: research-validated, no custom logic needed. They cover the most common quality dimensions: correctness, relevance, safety, and retrieval quality.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC