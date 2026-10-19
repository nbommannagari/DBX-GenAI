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
# MAGIC # Lecture - Guideline Judges
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC While built-in judges cover common criteria, many evaluation needs are specific to your business: tone, compliance, formatting, or domain-specific rules. Guideline judges let you express these requirements in natural language, making evaluation accessible to domain experts without code changes.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Distinguish between global and per-row guideline judges
# MAGIC - Implement the `Guidelines` class for uniform evaluation criteria
# MAGIC - Apply the `ExpectationsGuidelines` class for scenario-specific evaluations
# MAGIC - Write effective natural language guidelines

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. Two Types of Guideline Judges

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. Global vs. Per-Row Guidelines
# MAGIC
# MAGIC <div style="display:flex;justify-content:center;gap:16px;margin:18px 0;font-family:system-ui,sans-serif;flex-wrap:wrap;">
# MAGIC   <div style="flex:1;min-width:300px;border:2px solid #2272B4;border-radius:10px;padding:16px;text-align:center;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#2272B4;margin-bottom:10px;">Global Guidelines</div>
# MAGIC     <div style="font-size:14px;color:#333;margin-bottom:8px;">Same rules applied to <strong>every row</strong></div>
# MAGIC     <div style="font-size:13px;color:#618794;">Tone, style, formatting, compliance</div>
# MAGIC     <div style="margin-top:10px;font-size:13px;background:#E3F2FD;border-radius:4px;padding:6px 10px;display:inline-block;color:#1565C0;font-family:monospace;font-weight:600;">Guidelines()</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:300px;border:2px solid #00A972;border-radius:10px;padding:16px;text-align:center;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#00A972;margin-bottom:10px;">Per-Row Guidelines</div>
# MAGIC     <div style="font-size:14px;color:#333;margin-bottom:8px;">Different rules <strong>per example</strong></div>
# MAGIC     <div style="font-size:13px;color:#618794;">Scenario-specific validation criteria</div>
# MAGIC     <div style="margin-top:10px;font-size:13px;background:#E8F5E9;border-radius:4px;padding:6px 10px;display:inline-block;color:#2E7D32;font-family:monospace;font-weight:600;">ExpectationsGuidelines()</div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### A2. Implementation Patterns
# MAGIC
# MAGIC Click each type to see how it works:
# MAGIC
# MAGIC <div style="max-width:1000px;margin:0 auto;font-family:sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .gl-step-row { display:flex; gap:12px; margin-bottom:0; flex-wrap:wrap; }
# MAGIC .gl-step-box {
# MAGIC   flex:1; min-width:280px; background:#F9F7F4; border-top:6px solid transparent;
# MAGIC   border-left:2px solid transparent; border-right:2px solid transparent; border-bottom:2px solid transparent;
# MAGIC   border-radius:8px; padding:14px 12px; text-align:center; cursor:pointer; user-select:none;
# MAGIC   transition:transform 0.12s;
# MAGIC }
# MAGIC .gl-step-box:hover { transform:translateY(-2px); }
# MAGIC .gl-step-box.active { background:#fff; border-left-color:var(--gc); border-right-color:var(--gc); border-bottom-color:var(--gc); }
# MAGIC .gl-step-title { display:block; font-size:16px; font-weight:700; color:#0b2026; line-height:1.3; pointer-events:none; }
# MAGIC .gl-step-sub { display:block; font-size:13px; color:#618794; margin-top:4px; pointer-events:none; }
# MAGIC .gl-detail-wrap { overflow:hidden; max-height:0; opacity:0; transition:max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease; margin-top:0; }
# MAGIC .gl-detail-wrap.open { max-height:1200px; opacity:1; margin-top:12px; }
# MAGIC .gl-detail-card { background:#F9F7F4; border-radius:10px; padding:20px; border-top:6px solid #ccc; }
# MAGIC .gl-code-block { border-left:4px solid; border-radius:0 6px 6px 0; padding:14px; margin-top:10px; font-size:13px; line-height:1.5; font-family:monospace; white-space:pre; overflow-x:auto; }
# MAGIC </style>
# MAGIC
# MAGIC <div style="background:#F8F9FC;border:3px solid #1B5162;border-radius:10px;padding:20px;">
# MAGIC
# MAGIC <div class="gl-step-row">
# MAGIC   <span class="gl-step-box active" data-id="0" onclick="selectGlStep(0)" style="--gc:#2272B4;border-top-color:#2272B4;">
# MAGIC     <span class="gl-step-title">Global Guidelines</span>
# MAGIC     <span class="gl-step-sub">Same rules for every row</span>
# MAGIC   </span>
# MAGIC   <span class="gl-step-box" data-id="1" onclick="selectGlStep(1)" style="--gc:#00A972;border-top-color:#00A972;">
# MAGIC     <span class="gl-step-title">Per-Row Guidelines</span>
# MAGIC     <span class="gl-step-sub">Different rules per example</span>
# MAGIC   </span>
# MAGIC </div>
# MAGIC
# MAGIC <div class="gl-detail-wrap" id="gl-detail-wrap">
# MAGIC   <div class="gl-detail-card" id="gl-detail-card"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var GL_TYPES = [
# MAGIC   {
# MAGIC     title: "Global Guidelines",
# MAGIC     color: "#2272B4",
# MAGIC     bg: "rgba(34,114,180,0.10)",
# MAGIC     desc: "Apply <strong>uniform criteria across all evaluations</strong> \u2014 ideal for consistent standards like tone, style, or formatting. Use the <code>Guidelines</code> class with a list of natural-language rules.",
# MAGIC     code: 'from mlflow.genai.scorers import Guidelines\nimport mlflow\n\ntone_guidelines = Guidelines(\n    name=\"professional_tone\",\n    guidelines=[\n        \"The response must use professional language\",\n        \"The response must not use slang\",\n        \"The response must address the user respectfully\"\n    ],\n    model=\"databricks:/foundation-model-endpoint\"\n)\n\n# Use in evaluation\nresults = mlflow.genai.evaluate(\n    data=eval_data,\n    scorers=[tone_guidelines]\n)'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "Per-Row Guidelines",
# MAGIC     color: "#00A972",
# MAGIC     bg: "rgba(0,169,114,0.10)",
# MAGIC     desc: "Apply <strong>different criteria to each example</strong> \u2014 ideal when different scenarios require different validation standards. Each row in your dataset includes its own guidelines in the <code>expectations</code> field.",
# MAGIC     code: 'from mlflow.genai.scorers import ExpectationsGuidelines\nimport mlflow\n\ndata = [\n    {\"inputs\": {\"question\": \"What\'s the refund policy?\"},\n     \"outputs\": \"Returns within 30 days with a receipt.\",\n     \"expectations\": {\n         \"guidelines\": [\"The response must mention the 30-day timeframe\",\n                         \"The response must include the receipt requirement\"]}},\n    {\"inputs\": {\"question\": \"How do I contact support?\"},\n     \"outputs\": \"Reach us at support@example.com.\",\n     \"expectations\": {\n         \"guidelines\": [\"The response must provide a contact method\",\n                         \"The response must be concise\"]}},\n]\n\nresults = mlflow.genai.evaluate(\n    data=data,\n    scorers=[ExpectationsGuidelines()]\n)'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var glCurrent = null;
# MAGIC
# MAGIC function selectGlStep(id) {
# MAGIC   var wrap = document.getElementById('gl-detail-wrap');
# MAGIC   var card = document.getElementById('gl-detail-card');
# MAGIC   var g = GL_TYPES[id];
# MAGIC
# MAGIC   document.querySelectorAll('.gl-step-box').forEach(function(b) {
# MAGIC     b.classList.toggle('active', parseInt(b.dataset.id, 10) === id);
# MAGIC   });
# MAGIC
# MAGIC   if (glCurrent === id) {
# MAGIC     wrap.classList.remove('open');
# MAGIC     document.querySelectorAll('.gl-step-box').forEach(function(b) { b.classList.remove('active'); });
# MAGIC     glCurrent = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   glCurrent = id;
# MAGIC   card.style.borderTopColor = g.color;
# MAGIC   card.innerHTML =
# MAGIC     '<div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#0b2026;">' + g.title + '</div>'
# MAGIC     + '<div style="font-size:14px;line-height:1.6;color:#333;margin-bottom:10px;">' + g.desc + '</div>'
# MAGIC     + '<div class="gl-code-block" style="background:' + g.bg + ';border-color:' + g.color + ';">' 
# MAGIC     + '<code>' + g.code.replace(/\n/g, '<br>') + '</code></div>';
# MAGIC
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC
# MAGIC selectGlStep(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Advantages and Best Practices

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Why Guideline Judges
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:10px;margin:18px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="flex:1;min-width:155px;background:#00A972;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Expert Accessible</div>
# MAGIC     <div style="font-size:14px;margin-top:4px;opacity:0.9;">No coding required</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:155px;background:#2272B4;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Rapid Iteration</div>
# MAGIC     <div style="font-size:14px;margin-top:4px;opacity:0.9;">Update without code changes</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:155px;background:#FF5F46;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Interpretable</div>
# MAGIC     <div style="font-size:14px;margin-top:4px;opacity:0.9;">Self-documenting rules</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:155px;background:#1B5162;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Flexible</div>
# MAGIC     <div style="font-size:14px;margin-top:4px;opacity:0.9;">Complex context-dependent rules</div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### B2. Writing Effective Guidelines
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1; font-size: 1.0em;">Key rule: Refer to inputs as "the request" and outputs as "the response"</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333; font-size: 14px;">The LLM judge automatically extracts request/response data from your trace. When writing guidelines, always frame them in terms of "the request" (inputs) and "the response" (outputs).</p>
# MAGIC </div>
# MAGIC
# MAGIC **Recommended phrasing patterns:**
# MAGIC - "The response must …" (required behavior)
# MAGIC - "The response must not …" (prohibited behavior)
# MAGIC - "The response may optionally …" (suggested but not required)
# MAGIC
# MAGIC **Additional tips:**
# MAGIC - Be specific and concrete ("The response must cite the source document" vs. "The response should be credible")
# MAGIC - Write guidelines that can be objectively verified
# MAGIC - Focus on observable attributes in the response
# MAGIC - Leave as little ambiguity as possible
# MAGIC - Test guidelines on multiple examples to ensure they work as intended
# MAGIC
# MAGIC Docs: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/guidelines/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/judges/guidelines" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/judges/guidelines/" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to explore guideline judge patterns further? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-2-3" style="flex: 1;">What is the difference between Guidelines and ExpectationsGuidelines in MLflow?</span>
# MAGIC     <button onclick="
# MAGIC       var text = document.getElementById('genie-2-3').innerText;
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
# MAGIC Guideline judges bridge the gap between built-in judges and fully custom evaluation logic. They let domain experts define quality standards in natural language: global guidelines for uniform criteria, per-row guidelines for scenario-specific validation.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC