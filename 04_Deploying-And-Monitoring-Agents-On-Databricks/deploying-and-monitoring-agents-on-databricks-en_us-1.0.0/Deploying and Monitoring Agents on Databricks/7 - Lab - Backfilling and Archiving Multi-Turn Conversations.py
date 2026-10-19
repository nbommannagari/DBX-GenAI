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
# MAGIC # Lab - Backfilling and Archiving Multi-Turn Conversations
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This lab shifts focus from online evaluation to production monitoring, backfill, and archival workflows.
# MAGIC
# MAGIC In this lab, you manage a production travel agent. The setup script creates an MLflow experiment with multi-turn conversation traces and deploys a `ConversationCompleteness` judge. Your job is to inspect the existing setup, run the monitoring job, add a new scorer via backfill, set up long-term trace archival, and clean up when done.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC 1. Inspect an MLflow experiment with deployed judges and run the trace metrics computation job
# MAGIC 2. Backfill a new scorer on historical traces using `BackfillScorerConfig`
# MAGIC 3. Enable and disable trace archival to Unity Catalog Delta tables
# MAGIC 4. Stop and delete scorers using management APIs
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Lab Structure</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Each section follows a <strong>build → verify</strong> pattern. You write the code in a TODO cell, then run a verification cell to confirm your work is correct. Expand the <strong>Answer</strong> dropdown if you get stuck.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Jobs Created in This Lab</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">As you work through this lab, three types of Databricks jobs will be created. You can find them in <strong>Jobs & Pipelines</strong>:</p>
# MAGIC <ul style="margin: 8px 0 0 16px; color: #333;">
# MAGIC <li><strong>[experiment_id] Trace Metrics Computation Job</strong>: created by the setup when it deploys the <code>ConversationCompleteness</code> judge via <code>.start()</code>. Runs scorers on incoming traces.</li>
# MAGIC <li><strong>[experiment_id] Metric Backfill Computation Job</strong>: created when you call <code>backfill_scorers()</code>. Retroactively evaluates historical traces.</li>
# MAGIC <li><strong>[experiment_id] Trace Archive Job</strong>: created when you call <code>enable_databricks_trace_archival()</code>. Streams traces to a Delta table.</li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC </div>

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
# MAGIC ## A. Setup

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-7

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Metric Backfill
# MAGIC
# MAGIC <div style="background: #1B5162; color: white; padding: 20px 24px; border-radius: 10px; margin: 12px 0;">
# MAGIC <div style="font-size: 11pt; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; margin-bottom: 6px;">Scenario</div>
# MAGIC <div style="font-size: 14pt; line-height: 1.5;">The setup script created an MLflow experiment with multi-turn conversation traces and deployed a <strong>ConversationCompleteness</strong> judge. Your first task is to verify the setup, run the monitoring job, and inspect the results. Then you will add a new <strong>Safety</strong> scorer and backfill it on the historical traces.</div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Inspect the Experiment and Run the Deployed Job
# MAGIC
# MAGIC The setup script has already:
# MAGIC - Created an MLflow experiment with 3 multi-turn conversation sessions (9 traces total)
# MAGIC - Registered and started a `ConversationCompleteness` judge
# MAGIC - Created the background trace metrics computation job
# MAGIC
# MAGIC Your first step is to verify this in the UI, run the job, and see the initial assessments. Click each step below to expand.
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Default Conversation Timing</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Multi-turn judges run after a conversation is considered complete. By default, no new traces with that session ID for <strong>5 minutes</strong>. Make sure the setup script was ran at least 5 minutes ago.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .lab5-acc-item {
# MAGIC   border: 2px solid #e0e0e0;
# MAGIC   border-radius: 8px;
# MAGIC   margin-bottom: 8px;
# MAGIC   overflow: hidden;
# MAGIC   transition: border-color 0.2s;
# MAGIC }
# MAGIC .lab5-acc-item.open { border-color: #2574B5; }
# MAGIC .lab5-acc-header {
# MAGIC   display: flex;
# MAGIC   align-items: center;
# MAGIC   gap: 14px;
# MAGIC   padding: 14px 18px;
# MAGIC   cursor: pointer;
# MAGIC   user-select: none;
# MAGIC   background: #F9F7F4;
# MAGIC   transition: background 0.15s;
# MAGIC }
# MAGIC .lab5-acc-item.open .lab5-acc-header { background: #E8F1FA; }
# MAGIC .lab5-acc-num {
# MAGIC   width: 32px;
# MAGIC   height: 32px;
# MAGIC   border-radius: 50%;
# MAGIC   background: #e0e0e0;
# MAGIC   color: #666;
# MAGIC   font-size: 13pt;
# MAGIC   font-weight: 700;
# MAGIC   display: flex;
# MAGIC   align-items: center;
# MAGIC   justify-content: center;
# MAGIC   flex-shrink: 0;
# MAGIC   transition: background 0.2s, color 0.2s;
# MAGIC }
# MAGIC .lab5-acc-item.open .lab5-acc-num { background: #2574B5; color: #fff; }
# MAGIC .lab5-acc-summary {
# MAGIC   flex: 1;
# MAGIC   font-size: 13.5pt;
# MAGIC   font-weight: 600;
# MAGIC   color: #0b2026;
# MAGIC }
# MAGIC .lab5-acc-chevron {
# MAGIC   font-size: 14pt;
# MAGIC   color: #888;
# MAGIC   transition: transform 0.25s;
# MAGIC }
# MAGIC .lab5-acc-item.open .lab5-acc-chevron { transform: rotate(180deg); color: #2574B5; }
# MAGIC .lab5-acc-body {
# MAGIC   max-height: 0;
# MAGIC   opacity: 0;
# MAGIC   overflow: hidden;
# MAGIC   transition: max-height 0.3s ease, opacity 0.25s ease, padding 0.25s ease;
# MAGIC   padding: 0 18px 0 64px;
# MAGIC   font-size: 14pt;
# MAGIC   line-height: 1.7;
# MAGIC   color: #333;
# MAGIC }
# MAGIC .lab5-acc-item.open .lab5-acc-body {
# MAGIC   max-height: 600px;
# MAGIC   opacity: 1;
# MAGIC   padding: 4px 18px 18px 64px;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <div id="lab5-acc-b1"></div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var LAB5_B1 = [
# MAGIC   {
# MAGIC     summary: 'Open the MLflow Experiment and review the sessions',
# MAGIC     detail: 'Navigate to <strong>Experiments</strong> in the left menu. Search for <strong>backfill-and-archival-training</strong> and click on it. Click <strong>Sessions</strong> to see the 3 conversation sessions. Expand each session to review the turns:<br><br><strong>Session 1 (user_101)</strong>: Hotel cancellation. The agent ignores the request entirely and suggests tourism tips. The user repeats with increasing frustration.<br><strong>Session 2 (user_202)</strong>: Flight booking. The agent gives vague, incomplete answers ("check online travel sites"). The user has to keep asking for specifics.<br><strong>Session 3 (user_303)</strong>: Japan trip planning. The agent gives thorough, complete answers. This is the baseline.<br><br>These are the quality issues that multi-turn judges are designed to catch.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Verify the deployed scorer and its sample rate',
# MAGIC     detail: 'Click <strong>Scorers</strong> under <strong>Evaluation</strong>. You should see <strong>conversation_completeness</strong> listed as an active scorer. This was deployed by the setup script. Note the <strong>sample rate is set to 50%</strong>, meaning only about half of the sessions will be evaluated on each job run. This is a cost-saving measure for expensive LLM judges.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Find and run the Trace Metrics Computation Job',
# MAGIC     detail: 'Navigate to <strong>Jobs &amp; Pipelines</strong> and search for <em>Trace Metrics Computation Job</em>. Click <strong>Pause</strong> to stop the automatic schedule, then click <strong>Run now</strong>.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Inspect the job output',
# MAGIC     detail: 'After the job completes, click the last successful run (green bar). Review the log. you should see traces fetched, session-level scoring for <code>conversation_completeness</code>, and successful evaluations. Because the sample rate is 50%, not all sessions will have assessments.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'View the initial assessments',
# MAGIC     detail: 'Navigate back to the experiment. Click <strong>Sessions</strong> and expand a session that was evaluated. You should see <strong>ConversationCompleteness</strong> feedback on the first trace of the session. Compare the results: Session 1 (Japan trip) should pass, while Session 2 (vague flight answers) should be flagged as incomplete. exactly the quality issue the users reported.'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC (function() {
# MAGIC   var container = document.getElementById('lab5-acc-b1');
# MAGIC   LAB5_B1.forEach(function(item, i) {
# MAGIC     var div = document.createElement('div');
# MAGIC     div.className = 'lab5-acc-item' + (i === 0 ? ' open' : '');
# MAGIC     div.innerHTML =
# MAGIC       '<div class="lab5-acc-header" onclick="lab5ToggleB1(' + i + ')">'
# MAGIC       + '<div class="lab5-acc-num">' + (i + 1) + '</div>'
# MAGIC       + '<div class="lab5-acc-summary">' + item.summary + '</div>'
# MAGIC       + '<div class="lab5-acc-chevron">&#9660;</div>'
# MAGIC       + '</div>'
# MAGIC       + '<div class="lab5-acc-body">' + item.detail + '</div>';
# MAGIC     container.appendChild(div);
# MAGIC   });
# MAGIC })();
# MAGIC
# MAGIC function lab5ToggleB1(idx) {
# MAGIC   var items = document.querySelectorAll('#lab5-acc-b1 .lab5-acc-item');
# MAGIC   items.forEach(function(el, i) {
# MAGIC     el.classList.toggle('open', i === idx && !el.classList.contains('open'));
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Backfill a New Safety Scorer
# MAGIC
# MAGIC Now that you have seen the completeness results, you want to add a **Safety** scorer to check all historical traces for harmful content. Since Safety is a single-turn scorer, you can use `backfill_scorers` with a `BackfillScorerConfig` to retroactively evaluate the existing traces at a custom sample rate.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Why backfill instead of just .start()?</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><code>.start()</code> only evaluates <strong>new incoming traces</strong>. The 9 traces from the setup already exist. To evaluate them, you need <code>backfill_scorers()</code> which retroactively applies the scorer to historical traces.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Running `backfill_scorers()` will create a **[experiment_id] Metric Backfill Computation Job** in Jobs & Pipelines. After running the TODO cell below, navigate to the job in the Jobs UI to inspect the output. you should see the traces being fetched and the Safety scorer evaluating them.
# MAGIC
# MAGIC **TODO:** Register and start a `Safety` scorer, then backfill it on the historical traces using `BackfillScorerConfig` with `sample_rate=0.8`.

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# 1. Import Safety, ScorerSamplingConfig from mlflow.genai.scorers
# 2. Import backfill_scorers, BackfillScorerConfig from databricks.agents.scorers
# 3. Register and start a Safety scorer (name="safety_check", sample_rate=0.5)
# 4. Call backfill_scorers with BackfillScorerConfig(scorer=safety, sample_rate=0.8)
# 5. Print the returned job_id
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> B2. Backfill a New Safety Scorer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import Safety, ScorerSamplingConfig
# MAGIC from databricks.agents.scorers import backfill_scorers, BackfillScorerConfig
# MAGIC
# MAGIC # Register and start the Safety scorer
# MAGIC safety = Safety().register(name="safety_check")
# MAGIC safety = safety.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
# MAGIC
# MAGIC # Backfill historical traces at 80% sample rate
# MAGIC job_id = backfill_scorers(
# MAGIC     scorers=[BackfillScorerConfig(scorer=safety, sample_rate=0.8)]
# MAGIC )
# MAGIC print(f"Backfill job ID: {job_id}")
# MAGIC ```

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC <details style="margin:12px 0;border:0.5px solid #dde0e8;border-radius:8px;overflow:hidden;font-family:system-ui,sans-serif">
# MAGIC <summary style="background:#f7f8fc;padding:12px 16px;cursor:pointer;font-weight:500;font-size:17px;color:#555;user-select:none;display:flex;align-items:center;gap:10px">
# MAGIC <span style="background:#f0f2f7;border-radius:4px;padding:2px 8px;font-size:13px;color:#999">DEEP DIVE</span> What does backfill_scorers() actually do?
# MAGIC </summary>
# MAGIC <div style="padding:16px 18px;font-size:17px;line-height:1.7;color:#555">
# MAGIC
# MAGIC <p><code>backfill_scorers</code> retroactively applies scorers to historical traces that already exist in the experiment. Per the docs: <em>"You can retroactively apply new or updated metrics to historical traces."</em></p>
# MAGIC
# MAGIC <p style="font-weight:500;color:#0b2026;margin:14px 0 6px">Step by step:</p>
# MAGIC <ol style="padding-left:18px">
# MAGIC <li><strong>Resolves the active experiment</strong>. Uses the currently active MLflow experiment, or an explicit <code>experiment_id=</code> parameter</li>
# MAGIC <li><strong>Looks up scorers</strong>. Accepts name strings (uses registered sample rates) or <code>BackfillScorerConfig</code> objects (custom rates per scorer)</li>
# MAGIC <li><strong>Creates a backfill job</strong>. Returns a <code>job_id</code> (string) for status tracking</li>
# MAGIC </ol>
# MAGIC
# MAGIC <table style="width:100%;border-collapse:collapse;margin:14px 0;font-size:17px">
# MAGIC <thead><tr><th style="background:#f0f2f7;color:#555;padding:8px 12px;text-align:left;font-weight:500;border-bottom:0.5px solid #dde0e8">Scheduled job (.start())</th><th style="background:#f0f2f7;color:#555;padding:8px 12px;text-align:left;font-weight:500;border-bottom:0.5px solid #dde0e8">Backfill (backfill_scorers())</th></tr></thead>
# MAGIC <tbody>
# MAGIC <tr><td style="padding:8px 12px;border-bottom:0.5px solid #dde0e8">Evaluates new incoming traces on a recurring schedule</td><td style="padding:8px 12px;border-bottom:0.5px solid #dde0e8">Evaluates historical traces retroactively, once</td></tr>
# MAGIC <tr><td style="padding:8px 12px;border-bottom:0.5px solid #dde0e8;background:#f8f8f8">Ongoing</td><td style="padding:8px 12px;border-bottom:0.5px solid #dde0e8;background:#f8f8f8">One-shot</td></tr>
# MAGIC </tbody></table>
# MAGIC
# MAGIC <p style="font-weight:500;color:#0b2026;margin:14px 0 6px">Important constraints:</p>
# MAGIC <ul style="padding-left:18px">
# MAGIC <li>Scorers must already be registered and started in the experiment</li>
# MAGIC <li>Concurrent backfill attempts on the same experiment may return a <code>RESOURCE_CONFLICT</code> error</li>
# MAGIC </ul>
# MAGIC
# MAGIC <p style="font-size:14px;color:#999;margin-top:12px"><em>Source: Monitor GenAI in production (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring" style="color:#378ADD">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/production-monitoring" style="color:#378ADD">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/production-monitoring" style="color:#378ADD">GCP</a>)</em></p>
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B3. Verify Assessments on Traces
# MAGIC
# MAGIC Run the cell below to search for recent traces and confirm that the Safety backfill assessments have been attached.

# COMMAND ----------

# Verify: Search traces and check for assessments
import mlflow

experiment = mlflow.get_experiment_by_name(mlflow_experiment_path)
experiment_id = experiment.experiment_id if experiment else mlflow.get_experiment(0).experiment_id

traces_df = mlflow.search_traces(
    locations=[experiment_id],
    max_results=10
)

# Show trace IDs and assessments
display(traces_df[["trace_id", "assessments"]])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Trace Archival
# MAGIC
# MAGIC <div style="background: #1B5162; color: white; padding: 20px 24px; border-radius: 10px; margin: 12px 0;">
# MAGIC <div style="font-size: 11pt; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; margin-bottom: 6px;">Scenario</div>
# MAGIC <div style="font-size: 14pt; line-height: 1.5;">The data engineering team wants to build Lakeview dashboards over agent traces. You need to stream traces into a Delta table in Unity Catalog for long-term SQL-accessible storage.</div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1a. Enable Trace Archival
# MAGIC
# MAGIC Enabling archival creates a **[experiment_id] Trace Archive Job** in Jobs & Pipelines that continuously streams traces and their assessments to a Unity Catalog Delta table. The target table is created automatically if it doesn't exist.
# MAGIC
# MAGIC **TODO:** Enable archival using `mlflow.tracing.archival`. Target table: `{catalog_name}.{schema_name}.agent_traces`. Pass the `experiment_id` from the verification cell above.

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# Import enable_databricks_trace_archival from mlflow.tracing.archival
# Enable archival with:
#   delta_table_fullname = f"{catalog_name}.{schema_name}.agent_traces"
#   experiment_id = experiment_id (from D3 above)
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> C1. Enable Archival
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.tracing.archival import enable_databricks_trace_archival
# MAGIC
# MAGIC table_name = f"{catalog_name}.{schema_name}.agent_traces"
# MAGIC
# MAGIC enable_databricks_trace_archival(
# MAGIC     delta_table_fullname=table_name,
# MAGIC     experiment_id=experiment_id,
# MAGIC )
# MAGIC print(f"Archival enabled to: {table_name}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1b. Inspect the Archived Traces
# MAGIC
# MAGIC After enabling archival, the **Trace Archive Job** streams traces to the Delta table. Verify the data landed by inspecting the table in Unity Catalog. Click each step to expand.
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .lab5-acc2-item {
# MAGIC   border: 2px solid #e0e0e0;
# MAGIC   border-radius: 8px;
# MAGIC   margin-bottom: 8px;
# MAGIC   overflow: hidden;
# MAGIC   transition: border-color 0.2s;
# MAGIC }
# MAGIC .lab5-acc2-item.open { border-color: #2574B5; }
# MAGIC .lab5-acc2-header {
# MAGIC   display: flex;
# MAGIC   align-items: center;
# MAGIC   gap: 14px;
# MAGIC   padding: 14px 18px;
# MAGIC   cursor: pointer;
# MAGIC   user-select: none;
# MAGIC   background: #F9F7F4;
# MAGIC   transition: background 0.15s;
# MAGIC }
# MAGIC .lab5-acc2-item.open .lab5-acc2-header { background: #E8F1FA; }
# MAGIC .lab5-acc2-num {
# MAGIC   width: 32px;
# MAGIC   height: 32px;
# MAGIC   border-radius: 50%;
# MAGIC   background: #e0e0e0;
# MAGIC   color: #666;
# MAGIC   font-size: 13pt;
# MAGIC   font-weight: 700;
# MAGIC   display: flex;
# MAGIC   align-items: center;
# MAGIC   justify-content: center;
# MAGIC   flex-shrink: 0;
# MAGIC   transition: background 0.2s, color 0.2s;
# MAGIC }
# MAGIC .lab5-acc2-item.open .lab5-acc2-num { background: #2574B5; color: #fff; }
# MAGIC .lab5-acc2-summary {
# MAGIC   flex: 1;
# MAGIC   font-size: 13.5pt;
# MAGIC   font-weight: 600;
# MAGIC   color: #0b2026;
# MAGIC }
# MAGIC .lab5-acc2-chevron {
# MAGIC   font-size: 14pt;
# MAGIC   color: #888;
# MAGIC   transition: transform 0.25s;
# MAGIC }
# MAGIC .lab5-acc2-item.open .lab5-acc2-chevron { transform: rotate(180deg); color: #2574B5; }
# MAGIC .lab5-acc2-body {
# MAGIC   max-height: 0;
# MAGIC   opacity: 0;
# MAGIC   overflow: hidden;
# MAGIC   transition: max-height 0.3s ease, opacity 0.25s ease, padding 0.25s ease;
# MAGIC   padding: 0 18px 0 64px;
# MAGIC   font-size: 14pt;
# MAGIC   line-height: 1.7;
# MAGIC   color: #333;
# MAGIC }
# MAGIC .lab5-acc2-item.open .lab5-acc2-body {
# MAGIC   max-height: 600px;
# MAGIC   opacity: 1;
# MAGIC   padding: 4px 18px 18px 64px;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <div id="lab5-acc-c1b"></div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var LAB5_C1B = [
# MAGIC   {
# MAGIC     summary: 'Find and run the Trace Archive Job',
# MAGIC     detail: 'Navigate to <strong>Jobs &amp; Pipelines</strong> and search for <em>Trace Archive Job</em>. Your job will have the experiment ID prepended, e.g., <strong>[experiment_id] Trace Archive Job</strong>. Click <strong>Run now</strong> to trigger the archival. After the job completes, the traces and assessments will be written to the Delta table.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Navigate to the agent_traces table in Unity Catalog',
# MAGIC     detail: 'Open <strong>Catalog</strong> in the left menu. Navigate to your catalog and schema. You should see the <strong>agent_traces</strong> table that was created by the archival job.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Attach a SQL warehouse',
# MAGIC     detail: 'To preview the data, you need a SQL warehouse attached. Click the <strong>warehouse selector</strong> at the top of the table page and select an available SQL warehouse. If none are running, start one.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'View sample data',
# MAGIC     detail: 'Click <strong>Sample Data</strong> to preview the first rows. You should see columns for trace IDs, request/response data, timestamps, and assessments. This is the same trace data from your experiment, now queryable with SQL for dashboards and long-term analysis.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Inspect the table schema',
# MAGIC     detail: 'Click <strong>Schema</strong> to see the column definitions. Note the structure. the table schema is managed automatically by the archival job. You can join this table with other business data in your lakehouse for advanced analytics.'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC (function() {
# MAGIC   var container = document.getElementById('lab5-acc-c1b');
# MAGIC   LAB5_C1B.forEach(function(item, i) {
# MAGIC     var div = document.createElement('div');
# MAGIC     div.className = 'lab5-acc2-item' + (i === 0 ? ' open' : '');
# MAGIC     div.innerHTML =
# MAGIC       '<div class="lab5-acc2-header" onclick="lab5ToggleC1b(' + i + ')">'
# MAGIC       + '<div class="lab5-acc2-num">' + (i + 1) + '</div>'
# MAGIC       + '<div class="lab5-acc2-summary">' + item.summary + '</div>'
# MAGIC       + '<div class="lab5-acc2-chevron">&#9660;</div>'
# MAGIC       + '</div>'
# MAGIC       + '<div class="lab5-acc2-body">' + item.detail + '</div>';
# MAGIC     container.appendChild(div);
# MAGIC   });
# MAGIC })();
# MAGIC
# MAGIC function lab5ToggleC1b(idx) {
# MAGIC   var items = document.querySelectorAll('#lab5-acc-c1b .lab5-acc2-item');
# MAGIC   items.forEach(function(el, i) {
# MAGIC     el.classList.toggle('open', i === idx && !el.classList.contains('open'));
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C2. Disable Archival
# MAGIC
# MAGIC After enabling archival, the **Trace Archive Job** streams traces continuously to the Delta table. You can stop archival two ways:
# MAGIC
# MAGIC **Option 1. API:** Call `disable_databricks_trace_archival(experiment_id=...)` in code (the TODO below).
# MAGIC
# MAGIC **Option 2. UI:** Navigate to your experiment page, click **Delta sync: Enabled**, then click **Disable syncing**.
# MAGIC
# MAGIC Both methods stop the archive job. Existing archived data in the Delta table is preserved. Only new traces stop flowing.
# MAGIC
# MAGIC **TODO:** Disable archival for this experiment using the API.

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# Import disable_databricks_trace_archival from mlflow.tracing.archival
# Disable archival, passing experiment_id
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> C2. Disable Archival
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.tracing.archival import disable_databricks_trace_archival
# MAGIC
# MAGIC disable_databricks_trace_archival(experiment_id=experiment_id)
# MAGIC print("Archival disabled. Existing data preserved.")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Cleanup: Stop and Delete All Scorers
# MAGIC Stop all scorers, then delete them. Use `list_scorers()` to enumerate, `.stop()` to pause, and `delete_scorer()` to remove permanently. Historical assessments remain on their traces even after deletion.

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# 1. List all scorers
# 2. Stop each one
# 3. Delete each one using delete_scorer(name=...)
# 4. Verify the list is empty
#
# Hint: from mlflow.genai.scorers import list_scorers, delete_scorer
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> D1. Cleanup
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import list_scorers, delete_scorer
# MAGIC
# MAGIC scorers = list_scorers()
# MAGIC print(f"Stopping and deleting {len(scorers)} scorers:\n")
# MAGIC
# MAGIC for s in scorers:
# MAGIC     print(f"  {s.name}...", end=" ")
# MAGIC     s.stop()
# MAGIC     delete_scorer(name=s.name)
# MAGIC     print("deleted")
# MAGIC
# MAGIC # Verify
# MAGIC remaining = list_scorers()
# MAGIC print(f"\nRemaining scorers: {len(remaining)}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC 1. **Metric backfill** (`databricks.agents.scorers.backfill_scorers`) retroactively evaluates historical traces with per-scorer sample rate and time range controls
# MAGIC 2. **Two backfill modes**: pass scorer name strings (uses registered rates) or `BackfillScorerConfig` objects (custom rates)
# MAGIC 3. **Trace archival** (`mlflow.tracing.archival`) streams traces to Unity Catalog Delta tables for long-term SQL analysis and dashboarding
# MAGIC 4. **Management APIs** (`list_scorers`, `get_scorer`, `delete_scorer`) let you audit, update, and clean up scorers in production
# MAGIC 5. **Cleanup matters**: always stop and delete scorers when done to avoid unnecessary evaluation costs

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC