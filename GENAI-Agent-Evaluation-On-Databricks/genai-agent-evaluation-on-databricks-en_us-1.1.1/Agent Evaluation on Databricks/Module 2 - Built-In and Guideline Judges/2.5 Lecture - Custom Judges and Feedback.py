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
# MAGIC # Lecture - Custom Judges and Feedback
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC When built-in and guideline judges don't cover your specific evaluation needs, MLflow supports fully custom evaluation logic: both code-based scorers for deterministic checks and custom LLM judges for sophisticated assessments. This lecture also covers `Feedback` objects, which provide the structured rationales that make evaluation results interpretable and actionable.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Create code-based scorers using the `@scorer` decorator
# MAGIC - Understand when and how to implement custom LLM judges with `make_judge()`
# MAGIC - Work with `Feedback` objects and understand the role of rationales
# MAGIC - Choose the right judge type for your evaluation requirements

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. Code-Based Scorers

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ### A1. Deterministic Evaluation with `@scorer`
# MAGIC
# MAGIC The `@scorer` decorator converts a Python function into a reusable evaluation component. Your function can accept any combination of these keyword arguments:
# MAGIC
# MAGIC | Parameter | Type | Description |
# MAGIC | --- | --- | --- |
# MAGIC | `inputs` | `dict[str, Any]` | The app's raw input (argument names → values) |
# MAGIC | `outputs` | `Any` | The app's raw output |
# MAGIC | `expectations` | `dict[str, Any]` | Ground truth labels (e.g., expected facts, guidelines) |
# MAGIC | `trace` | `mlflow.entities.Trace` | Complete trace with all spans and metadata |
# MAGIC
# MAGIC <div style="border-left: 4px solid #2272B4; background: #e3f2fd; padding: 12px 16px; border-radius: 4px; margin: 12px 0;">
# MAGIC <strong style="color: #1565c0; font-size: 13px;">Imperative approach:</strong>
# MAGIC <span style="font-size: 13px; color: #333;"> You write the evaluation logic directly in Python — full control over scoring decisions, data access, and error handling. Use this when evaluation criteria can be expressed as deterministic code (format checks, length validation, keyword matching, schema verification).</span>
# MAGIC </div>
# MAGIC
# MAGIC Click each approach to see how:
# MAGIC
# MAGIC <div style="max-width:1000px;margin:0 auto;font-family:sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .sc-step-row { display:flex; gap:12px; margin-bottom:0; flex-wrap:wrap; }
# MAGIC .sc-step-box {
# MAGIC   flex:1; min-width:280px; background:#F9F7F4; border-top:6px solid transparent;
# MAGIC   border-left:2px solid transparent; border-right:2px solid transparent; border-bottom:2px solid transparent;
# MAGIC   border-radius:8px; padding:14px 12px; text-align:center; cursor:pointer; user-select:none;
# MAGIC   transition:transform 0.12s;
# MAGIC }
# MAGIC .sc-step-box:hover { transform:translateY(-2px); }
# MAGIC .sc-step-box.active { background:#fff; border-left-color:var(--xc); border-right-color:var(--xc); border-bottom-color:var(--xc); }
# MAGIC .sc-step-title { display:block; font-size:16px; font-weight:700; color:#0b2026; line-height:1.3; pointer-events:none; }
# MAGIC .sc-step-sub { display:block; font-size:13px; color:#618794; margin-top:4px; pointer-events:none; }
# MAGIC .sc-detail-wrap { overflow:hidden; max-height:0; opacity:0; transition:max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease; margin-top:0; }
# MAGIC .sc-detail-wrap.open { max-height:1200px; opacity:1; margin-top:12px; }
# MAGIC .sc-detail-card { background:#F9F7F4; border-radius:10px; padding:20px; border-top:6px solid #ccc; }
# MAGIC .sc-code-block { border-left:4px solid; border-radius:0 6px 6px 0; padding:14px; margin-top:10px; font-size:13px; line-height:1.5; font-family:monospace; white-space:pre; overflow-x:auto; }
# MAGIC </style>
# MAGIC
# MAGIC <div style="background:#F8F9FC;border:3px solid #1B5162;border-radius:10px;padding:20px;">
# MAGIC
# MAGIC <div class="sc-step-row">
# MAGIC   <span class="sc-step-box active" data-id="0" onclick="selectScStep(0)" style="--xc:#2272B4;border-top-color:#2272B4;">
# MAGIC     <span class="sc-step-title">With Feedback Object</span>
# MAGIC     <span class="sc-step-sub">Score + rationale + metadata</span>
# MAGIC   </span>
# MAGIC   <span class="sc-step-box" data-id="1" onclick="selectScStep(1)" style="--xc:#00A972;border-top-color:#00A972;">
# MAGIC     <span class="sc-step-title">Primitive Return</span>
# MAGIC     <span class="sc-step-sub">Simple pass/fail, no rationale</span>
# MAGIC   </span>
# MAGIC </div>
# MAGIC
# MAGIC <div class="sc-detail-wrap" id="sc-detail-wrap">
# MAGIC   <div class="sc-detail-card" id="sc-detail-card"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var SC_TYPES = [
# MAGIC   {
# MAGIC     title: "With Feedback Object",
# MAGIC     color: "#2272B4",
# MAGIC     bg: "rgba(34,114,180,0.10)",
# MAGIC     desc: "Return a <code>Feedback</code> object with <strong>value</strong>, <strong>rationale</strong>, and optional <strong>metadata</strong>. This gives you full interpretability \u2014 you can see <em>why</em> an example passed or failed.",
# MAGIC     code: 'from mlflow.genai.scorers import scorer\nfrom mlflow.entities import Feedback\n\n@scorer\ndef response_length(outputs):\n    """Verify response length is appropriate."""\n    word_count = len(str(outputs.get("response", "")).split())\n\n    if 20 <= word_count <= 100:\n        return Feedback(\n            value="yes",\n            rationale=f"Response length ({word_count} words) is appropriate"\n        )\n    else:\n        return Feedback(\n            value="no",\n            rationale=f"Response is too {\\"short\\" if word_count < 20 else \\"long\\"}"\n        )'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "Primitive Return",
# MAGIC     color: "#00A972",
# MAGIC     bg: "rgba(0,169,114,0.10)",
# MAGIC     desc: "For straightforward pass/fail checks, scorers can return primitive values directly. More concise, but no rationale for the scoring decision.",
# MAGIC     code: '@scorer\ndef response_length(outputs):\n    wc = len(str(outputs.get("response", "")).split())\n    return "yes" if 20 <= wc <= 100 else "no"'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var scCurrent = null;
# MAGIC
# MAGIC function selectScStep(id) {
# MAGIC   var wrap = document.getElementById('sc-detail-wrap');
# MAGIC   var card = document.getElementById('sc-detail-card');
# MAGIC   var s = SC_TYPES[id];
# MAGIC
# MAGIC   document.querySelectorAll('.sc-step-box').forEach(function(b) {
# MAGIC     b.classList.toggle('active', parseInt(b.dataset.id, 10) === id);
# MAGIC   });
# MAGIC
# MAGIC   if (scCurrent === id) {
# MAGIC     wrap.classList.remove('open');
# MAGIC     document.querySelectorAll('.sc-step-box').forEach(function(b) { b.classList.remove('active'); });
# MAGIC     scCurrent = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   scCurrent = id;
# MAGIC   card.style.borderTopColor = s.color;
# MAGIC   card.innerHTML =
# MAGIC     '<div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#0b2026;">' + s.title + '</div>'
# MAGIC     + '<div style="font-size:14px;line-height:1.6;color:#333;margin-bottom:10px;">' + s.desc + '</div>'
# MAGIC     + '<div class="sc-code-block" style="background:' + s.bg + ';border-color:' + s.color + ';">' 
# MAGIC     + '<code>' + s.code.replace(/\n/g, '<br>') + '</code></div>';
# MAGIC
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC
# MAGIC selectScStep(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Custom LLM Judges

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### B1. When to Use Custom LLM Judges
# MAGIC
# MAGIC Custom LLM judges are for evaluation criteria not covered by built-in or guideline judges:
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:10px;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="flex:1;min-width:170px;background:#FF5F46;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Domain-Specific</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Criteria unique to your application</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:170px;background:#1B5162;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Nuanced Assessment</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Beyond what built-in judges capture</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:170px;background:#2272B4;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Proprietary Standards</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Regulations &amp; compliance</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:170px;background:#00A972;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Multi-Signal Logic</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Combine multiple data sources</div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### B2. Building Custom LLM Judges with `make_judge()`
# MAGIC
# MAGIC The `make_judge()` API lets you define evaluation criteria in natural language. MLflow handles the LLM invocation, response parsing, and `Feedback` generation automatically. Click each pattern:
# MAGIC
# MAGIC <div style="border-left: 4px solid #00A972; background: #e8f5e9; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #2e7d32; font-size: 1.0em;">Declarative approach:</strong>
# MAGIC <span style="font-size: 14px; color: #333;"> Unlike <code>@scorer</code> where you write evaluation logic in Python, <code>make_judge()</code> lets you describe <em>what</em> to evaluate in natural language. MLflow manages the LLM call, response parsing, and structured output enforcement. Use this when evaluation requires subjective reasoning that's hard to express as deterministic code.</span>
# MAGIC </div>
# MAGIC
# MAGIC <div style="max-width:1000px;margin:0 auto;font-family:sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .mj-step-row { display:flex; gap:12px; margin-bottom:0; flex-wrap:wrap; }
# MAGIC .mj-step-box {
# MAGIC   flex:1; min-width:280px; background:#F9F7F4; border-top:6px solid transparent;
# MAGIC   border-left:2px solid transparent; border-right:2px solid transparent; border-bottom:2px solid transparent;
# MAGIC   border-radius:8px; padding:14px 12px; text-align:center; cursor:pointer; user-select:none;
# MAGIC   transition:transform 0.12s;
# MAGIC }
# MAGIC .mj-step-box:hover { transform:translateY(-2px); }
# MAGIC .mj-step-box.active { background:#fff; border-left-color:var(--mc); border-right-color:var(--mc); border-bottom-color:var(--mc); }
# MAGIC .mj-step-title { display:block; font-size:16px; font-weight:700; color:#0b2026; line-height:1.3; pointer-events:none; }
# MAGIC .mj-step-sub { display:block; font-size:13px; color:#618794; margin-top:4px; pointer-events:none; }
# MAGIC .mj-detail-wrap { overflow:hidden; max-height:0; opacity:0; transition:max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease; margin-top:0; }
# MAGIC .mj-detail-wrap.open { max-height:1200px; opacity:1; margin-top:12px; }
# MAGIC .mj-detail-card { background:#F9F7F4; border-radius:10px; padding:20px; border-top:6px solid #ccc; }
# MAGIC .mj-code-block { border-left:4px solid; border-radius:0 6px 6px 0; padding:14px; margin-top:10px; font-size:13px; line-height:1.5; font-family:monospace; white-space:pre; overflow-x:auto; }
# MAGIC </style>
# MAGIC
# MAGIC <div style="background:#F8F9FC;border:3px solid #1B5162;border-radius:10px;padding:20px;">
# MAGIC
# MAGIC <div class="mj-step-row">
# MAGIC   <span class="mj-step-box active" data-id="0" onclick="selectMjStep(0)" style="--mc:#FF5F46;border-top-color:#FF5F46;">
# MAGIC     <span class="mj-step-title">Pass/Fail Judge</span>
# MAGIC     <span class="mj-step-sub">Constrained "yes"/"no" feedback</span>
# MAGIC   </span>
# MAGIC   <span class="mj-step-box" data-id="1" onclick="selectMjStep(1)" style="--mc:#7B61FF;border-top-color:#7B61FF;">
# MAGIC     <span class="mj-step-title">Multi-Category Judge</span>
# MAGIC     <span class="mj-step-sub">Custom Literal feedback values</span>
# MAGIC   </span>
# MAGIC </div>
# MAGIC
# MAGIC <div class="mj-detail-wrap" id="mj-detail-wrap">
# MAGIC   <div class="mj-detail-card" id="mj-detail-card"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var MJ_TYPES = [
# MAGIC   {
# MAGIC     title: "Pass/Fail Judge",
# MAGIC     color: "#FF5F46",
# MAGIC     bg: "rgba(255,95,70,0.10)",
# MAGIC     desc: "The simplest custom judge. Use <code>feedback_value_type=Literal[\"yes\", \"no\"]</code> to enforce binary pass/fail output. Provide a <code>name</code> and natural-language <code>instructions</code> with template variables (<code>{{inputs}}</code>, <code>{{outputs}}</code>, <code>{{expectations}}</code>).",
# MAGIC     code: 'from typing import Literal\nfrom mlflow.genai import make_judge\n\nformality_judge = make_judge(\n    name=\"formality\",\n    instructions=\"\"\"\n        Evaluate whether the response is written in a\n        professional, formal tone.\n        Request: {{inputs}}\n        Response: {{outputs}}\n    \"\"\",\n    model=\"databricks:/databricks-claude-sonnet-4-5\",\n    feedback_value_type=Literal[\"yes\", \"no\"],\n)'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "Multi-Category Judge",
# MAGIC     color: "#7B61FF",
# MAGIC     bg: "rgba(123,97,255,0.10)",
# MAGIC     desc: "Use <code>feedback_value_type</code> with <code>Literal</code> to define custom categories. The judge uses structured outputs to enforce the type.",
# MAGIC     code: 'from typing import Literal\nfrom mlflow.genai import make_judge\n\nquality_judge = make_judge(\n    name=\"answer_quality\",\n    instructions=\"\"\"\n        Evaluate the quality of the response.\n        Request: {{inputs}}\n        Response: {{outputs}}\n        Expected: {{expectations}}\n        Rate as excellent, acceptable, or poor.\n    \"\"\",\n    feedback_value_type=Literal[\"excellent\", \"acceptable\", \"poor\"],\n)'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var mjCurrent = null;
# MAGIC
# MAGIC function selectMjStep(id) {
# MAGIC   var wrap = document.getElementById('mj-detail-wrap');
# MAGIC   var card = document.getElementById('mj-detail-card');
# MAGIC   var m = MJ_TYPES[id];
# MAGIC
# MAGIC   document.querySelectorAll('.mj-step-box').forEach(function(b) {
# MAGIC     b.classList.toggle('active', parseInt(b.dataset.id, 10) === id);
# MAGIC   });
# MAGIC
# MAGIC   if (mjCurrent === id) {
# MAGIC     wrap.classList.remove('open');
# MAGIC     document.querySelectorAll('.mj-step-box').forEach(function(b) { b.classList.remove('active'); });
# MAGIC     mjCurrent = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   mjCurrent = id;
# MAGIC   card.style.borderTopColor = m.color;
# MAGIC   card.innerHTML =
# MAGIC     '<div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#0b2026;">' + m.title + '</div>'
# MAGIC     + '<div style="font-size:14px;line-height:1.6;color:#333;margin-bottom:10px;">' + m.desc + '</div>'
# MAGIC     + '<div class="mj-code-block" style="background:' + m.bg + ';border-color:' + m.color + ';">' 
# MAGIC     + '<code>' + m.code.replace(/\n/g, '<br>') + '</code></div>';
# MAGIC
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC
# MAGIC selectMjStep(0);
# MAGIC </script>
# MAGIC
# MAGIC <strong>Note:</strong> Both <code>@scorer</code> and <code>make_judge()</code> are supported in offline evaluation (<code>mlflow.genai.evaluate()</code>) and production monitoring (<code>ScorerScheduleConfig</code>).
# MAGIC
# MAGIC Docs: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-judge/index/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/custom-judge/" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/custom-judge/index/" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Feedback Objects and Rationales

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ### C1. Structured Feedback
# MAGIC
# MAGIC All MLflow judges return structured `Feedback` objects: not just scalar scores. This makes evaluation results interpretable and actionable.
# MAGIC
# MAGIC <div style="max-width:900px;margin:0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 130" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Feedback Object Structure</title>
# MAGIC   <desc>A Feedback object contains value, rationale, source, and metadata.</desc>
# MAGIC   <rect x="5" y="5" width="770" height="120" rx="12" fill="#F9F7F4" stroke="#1B3139" stroke-width="2"/>
# MAGIC   <text x="390" y="30" text-anchor="middle" font-size="17" font-weight="700" fill="#0b2026">Feedback</text>
# MAGIC   <rect x="15" y="45" width="170" height="65" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="100" y="70" text-anchor="middle" font-size="15" font-weight="600" fill="#2272B4" font-family="monospace">value</text>
# MAGIC   <text x="100" y="90" text-anchor="middle" font-size="12" fill="#618794">"yes" / "no" / score</text>
# MAGIC   <rect x="200" y="45" width="190" height="65" rx="8" fill="#00A972" fill-opacity="0.12" stroke="#00A972" stroke-width="1.5"/>
# MAGIC   <text x="295" y="70" text-anchor="middle" font-size="15" font-weight="600" fill="#00A972" font-family="monospace">rationale</text>
# MAGIC   <text x="295" y="90" text-anchor="middle" font-size="12" fill="#618794">Human-readable explanation</text>
# MAGIC   <rect x="405" y="45" width="180" height="65" rx="8" fill="#1B5162" fill-opacity="0.12" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="495" y="70" text-anchor="middle" font-size="15" font-weight="600" fill="#1B5162" font-family="monospace">source</text>
# MAGIC   <text x="495" y="90" text-anchor="middle" font-size="12" fill="#618794">AssessmentSource (type + id)</text>
# MAGIC   <rect x="600" y="45" width="160" height="65" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="680" y="70" text-anchor="middle" font-size="15" font-weight="600" fill="#FF5F46" font-family="monospace">metadata</text>
# MAGIC   <text x="680" y="90" text-anchor="middle" font-size="12" fill="#618794">Additional context (dict)</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <div style="font-size:13px;color:#618794;margin-top:8px;text-align:center;">For <code>@scorer</code>, <code>source</code> is auto-populated as <code>CODE</code>. For <code>make_judge()</code>, auto-populated as <code>LLM_JUDGE</code>. Set manually only when calling your own LLM inside a <code>@scorer</code>.</div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### C2. Why Rationales Matter
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:10px;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="flex:1;min-width:155px;background:#2272B4;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Debugging</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Why it failed, not just that it failed</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:155px;background:#00A972;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Judge Validation</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Verify reasoning is correct</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:155px;background:#FF5F46;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Pattern ID</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Common themes reveal issues</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:155px;background:#1B5162;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Communication</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Explain results to stakeholders</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #1B5162;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#1B5162;">Using rationales effectively</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC <ol style="margin:8px 0;">
# MAGIC <li>Read rationales for all failures to identify patterns</li>
# MAGIC <li>Spot-check rationales for passes to ensure judge is reasoning correctly</li>
# MAGIC <li>Extract common rationale phrases to categorize failure types</li>
# MAGIC <li>Share representative rationales when discussing evaluation with teams</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to explore custom judge patterns and feedback types? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-2-5" style="flex: 1;">How do I create a custom LLM judge using make_judge() in MLflow with trace-based evaluation?</span>
# MAGIC     <button onclick="
# MAGIC       var text = document.getElementById('genie-2-5').innerText;
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
# MAGIC ## D. Conclusion
# MAGIC
# MAGIC `@scorer` gives you imperative control for deterministic checks (format, length, schema), while `make_judge()` provides a declarative path for subjective LLM-based assessments. Both return structured `Feedback` objects with values, rationales, and auto-populated sources — and both work in offline evaluation and production monitoring.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC