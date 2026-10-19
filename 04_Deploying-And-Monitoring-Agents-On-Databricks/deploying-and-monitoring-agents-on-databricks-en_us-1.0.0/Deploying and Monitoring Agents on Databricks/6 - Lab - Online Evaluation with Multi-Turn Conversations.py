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
# MAGIC # Lab - Online Evaluation with Multi-Turn Conversations
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC In this lab, you are the ML engineer responsible for a production travel agent. The setup cell generates multi-turn conversation traces tagged with session IDs. Your job is to deploy multi-turn evaluation judges, verify they are running, and tune their sampling rates for production.
# MAGIC
# MAGIC **This lab assumes familiarity with online evaluation architecture, scorer types, and sampling strategies. You will now apply those concepts hands-on.**
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC 1. Configure multi-turn judges and verify they are active using `list_scorers()`
# MAGIC 2. Inspect and operate the trace metrics computation job
# MAGIC 3. Create custom scorers with the `@scorer` decorator and appropriate aggregations
# MAGIC 4. Manage the scorer lifecycle: stop, update, and delete
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Lab Structure</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Each section follows a <strong>build → verify</strong> pattern. TODO cells contain code you write; instructional cells guide you through the UI with expandable steps. Expand the <strong>Answer</strong> dropdown if you get stuck.</p>
# MAGIC </div>
# MAGIC </div>

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
# MAGIC ## A. Setup

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-5

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Multi-Turn Judges
# MAGIC
# MAGIC <div style="background: #1B5162; color: white; padding: 20px 24px; border-radius: 10px; margin: 12px 0;">
# MAGIC <div style="font-size: 11pt; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; margin-bottom: 6px;">Scenario</div>
# MAGIC <div style="font-size: 14pt; line-height: 1.5;">Users have reported that the agent sometimes gives incomplete answers or takes too many turns to resolve simple questions. You need multi-turn evaluation judges to detect these patterns across entire conversations.</div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Create and Deploy Conversation Judges
# MAGIC
# MAGIC The user complaints describe two distinct problems: **incomplete answers** (the agent doesn't fully resolve the request) and **too many turns** (users have to repeat themselves). Single-turn safety checks can't catch these because they evaluate each response in isolation. You need multi-turn judges that analyze the full conversation thread to detect these patterns. Each judge below targets a specific dimension of the problem:
# MAGIC
# MAGIC <div style="display: flex; gap: 16px; margin: 16px 0;">
# MAGIC   <div style="flex: 1; background: #F8F9FC; border-radius: 10px; padding: 16px 18px; border-left: 4px solid #2574B5;">
# MAGIC     <div style="font-weight: 700; color: #2574B5; font-size: 12pt; margin-bottom: 6px;">ConversationCompleteness</div>
# MAGIC     <div style="font-size: 12pt; color: #444;">Did the agent fully resolve the request?</div>
# MAGIC   </div>
# MAGIC   <div style="flex: 1; background: #F8F9FC; border-radius: 10px; padding: 16px 18px; border-left: 4px solid #FE3722;">
# MAGIC     <div style="font-weight: 700; color: #FE3722; font-size: 12pt; margin-bottom: 6px;">UserFrustration</div>
# MAGIC     <div style="font-size: 12pt; color: #444;">Is the user showing signs of frustration?</div>
# MAGIC   </div>
# MAGIC   <div style="flex: 1; background: #F8F9FC; border-radius: 10px; padding: 16px 18px; border-left: 4px solid #02A36F;">
# MAGIC     <div style="font-weight: 700; color: #02A36F; font-size: 12pt; margin-bottom: 6px;">ConversationalSafety</div>
# MAGIC     <div style="font-size: 12pt; color: #444;">Are there safety concerns across the session?</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">What happens when you call .start()</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Calling <code>.register()</code> and <code>.start()</code> activates <strong>online quality monitoring</strong> for your experiment. Databricks creates a background <strong>trace metrics computation job</strong> that continuously:</p>
# MAGIC <ul style="margin: 8px 0 0 16px; color: #333;">
# MAGIC <li>Pulls new traces from the experiment</li>
# MAGIC <li>Groups them into conversations via <code>mlflow.trace.session</code> metadata</li>
# MAGIC <li>Runs your scorers against each conversation</li>
# MAGIC <li>Logs the resulting assessments back as feedback on the traces</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">This job is created automatically the first time you call <code>.start()</code>. It keeps running as long as you have active scorers (<code>sample_rate > 0</code>).</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC **TODO:**
# MAGIC 1. Import all three judges from `mlflow.genai.scorers`, register them with unique names, and start them.
# MAGIC     - Safety is critical: **sample_rate=1.0**
# MAGIC     - Completeness and frustration are expensive LLM judges: **sample_rate=0.5**

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# Step 1: Import the three multi-turn judges + ScorerSamplingConfig
#         from mlflow.genai.scorers
# Step 2: Use mlflow.set_experiment() with mlflow_experiment_path
# Step 3: Instantiate, register (name= must be unique), and start each
#
# Safety:       sample_rate=1.0
# Completeness: sample_rate=0.5
# Frustration:  sample_rate=0.5
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> B1. Create and Deploy Conversation Judges
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC mlflow.set_experiment(mlflow_experiment_path)
# MAGIC
# MAGIC from mlflow.genai.scorers import (
# MAGIC     ConversationCompleteness,
# MAGIC     UserFrustration,
# MAGIC     ConversationalSafety,
# MAGIC     ScorerSamplingConfig,
# MAGIC )
# MAGIC
# MAGIC completeness = ConversationCompleteness().register(name="conversation_completeness")
# MAGIC completeness = completeness.start(
# MAGIC     sampling_config=ScorerSamplingConfig(sample_rate=0.5),
# MAGIC )
# MAGIC
# MAGIC frustration = UserFrustration().register(name="user_frustration")
# MAGIC frustration = frustration.start(
# MAGIC     sampling_config=ScorerSamplingConfig(sample_rate=0.5),
# MAGIC )
# MAGIC
# MAGIC safety = ConversationalSafety().register(name="conversational_safety")
# MAGIC safety = safety.start(
# MAGIC     sampling_config=ScorerSamplingConfig(sample_rate=1.0),
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Add a Custom Judge via the UI
# MAGIC
# MAGIC Before running the job, add one more judge. this time through the UI instead of code. This demonstrates that you can mix programmatic and UI-created judges on the same experiment. The job will evaluate all active judges in a single run.
# MAGIC
# MAGIC Click each step below to expand the instructions.
# MAGIC
# MAGIC <br></br>
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto; font-family: sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .lab-acc-item {
# MAGIC   border: 2px solid #e0e0e0;
# MAGIC   border-radius: 8px;
# MAGIC   margin-bottom: 8px;
# MAGIC   overflow: hidden;
# MAGIC   transition: border-color 0.2s;
# MAGIC }
# MAGIC .lab-acc-item.open { border-color: #2574B5; }
# MAGIC .lab-acc-header {
# MAGIC   display: flex;
# MAGIC   align-items: center;
# MAGIC   gap: 14px;
# MAGIC   padding: 14px 18px;
# MAGIC   cursor: pointer;
# MAGIC   user-select: none;
# MAGIC   background: #F9F7F4;
# MAGIC   transition: background 0.15s;
# MAGIC }
# MAGIC .lab-acc-item.open .lab-acc-header { background: #E8F1FA; }
# MAGIC .lab-acc-num {
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
# MAGIC .lab-acc-item.open .lab-acc-num { background: #2574B5; color: #fff; }
# MAGIC .lab-acc-summary {
# MAGIC   flex: 1;
# MAGIC   font-size: 13.5pt;
# MAGIC   font-weight: 600;
# MAGIC   color: #0b2026;
# MAGIC }
# MAGIC .lab-acc-chevron {
# MAGIC   font-size: 14pt;
# MAGIC   color: #888;
# MAGIC   transition: transform 0.25s;
# MAGIC }
# MAGIC .lab-acc-item.open .lab-acc-chevron { transform: rotate(180deg); color: #2574B5; }
# MAGIC .lab-acc-body {
# MAGIC   max-height: 0;
# MAGIC   opacity: 0;
# MAGIC   overflow: hidden;
# MAGIC   transition: max-height 0.3s ease, opacity 0.25s ease, padding 0.25s ease;
# MAGIC   padding: 0 18px 0 64px;
# MAGIC   font-size: 14pt;
# MAGIC   line-height: 1.7;
# MAGIC   color: #333;
# MAGIC }
# MAGIC .lab-acc-item.open .lab-acc-body {
# MAGIC   max-height: 600px;
# MAGIC   opacity: 1;
# MAGIC   padding: 4px 18px 18px 64px;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <div id="lab-acc-b2"></div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var LAB_B2 = [
# MAGIC   {
# MAGIC     summary: 'Navigate to the experiment',
# MAGIC     detail: 'Navigate to <strong>Experiments</strong> in the left menu. Search for and click on <strong>evaluation-and-monitoring-training</strong>.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Open the Scorers panel',
# MAGIC     detail: 'In your experiment, click <strong>Scorers</strong> under <strong>Evaluation</strong>, then click <strong>New LLM scorer</strong>.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Configure the scorer',
# MAGIC     detail: 'Click <strong>Sessions</strong> under <strong>General</strong>. Name it <strong>professional</strong>. Under <strong>Evaluation criteria</strong>, select <strong>Custom scorer</strong> and enter: <em>Evaluate if the interaction is professional {{conversation}}</em>'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Create',
# MAGIC     detail: 'Click <strong>Create Scorers</strong> at the bottom right. This will populate your experiment with a new scorer called <strong>professional</strong>.'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC (function() {
# MAGIC   var container = document.getElementById('lab-acc-b2');
# MAGIC   LAB_B2.forEach(function(item, i) {
# MAGIC     var div = document.createElement('div');
# MAGIC     div.className = 'lab-acc-item' + (i === 0 ? ' open' : '');
# MAGIC     div.innerHTML =
# MAGIC       '<div class="lab-acc-header" onclick="labToggleB2(' + i + ')">'
# MAGIC       + '<div class="lab-acc-num">' + (i + 1) + '</div>'
# MAGIC       + '<div class="lab-acc-summary">' + item.summary + '</div>'
# MAGIC       + '<div class="lab-acc-chevron">&#9660;</div>'
# MAGIC       + '</div>'
# MAGIC       + '<div class="lab-acc-body">' + item.detail + '</div>';
# MAGIC     container.appendChild(div);
# MAGIC   });
# MAGIC })();
# MAGIC
# MAGIC function labToggleB2(idx) {
# MAGIC   var items = document.querySelectorAll('#lab-acc-b2 .lab-acc-item');
# MAGIC   items.forEach(function(el, i) {
# MAGIC     el.classList.toggle('open', i === idx && !el.classList.contains('open'));
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B3. Add a Custom Scorer in Code
# MAGIC
# MAGIC The built-in judges evaluate quality dimensions like safety and completeness, but your team also wants to track a basic metric: **response length**. Overly long responses slow down users and increase token costs; overly short ones may indicate the agent is not being helpful. A custom scorer lets you measure this programmatically and get mean/min/max aggregations across all evaluated traces.
# MAGIC
# MAGIC This demonstrates three ways to deploy scorers: out-of-the-box (B1), via the UI (B2), and custom code (B3). All will run together when the job executes.
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Serialization Rules</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Custom scorers run remotely. All <code>import</code> statements must be <strong>inside the function body</strong>. Return consistent types. Handle missing data gracefully.</p>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# Create a @scorer function that:
#   - Measures len(str(outputs.get("response", "")))
#   - Has aggregations=["mean", "min", "max"]
# Then register as "response_length" and start with sample_rate=0.5
#
# Hint: from mlflow.genai.scorers import scorer, ScorerSamplingConfig
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> B3. Custom Scorer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import scorer, ScorerSamplingConfig
# MAGIC
# MAGIC @scorer(aggregations=["mean", "min", "max"])
# MAGIC def response_length(outputs):
# MAGIC     """Measure response length in characters"""
# MAGIC     return len(str(outputs.get("response", "")))
# MAGIC
# MAGIC length_scorer = response_length.register(name="response_length")
# MAGIC length_scorer = length_scorer.start(
# MAGIC     sampling_config=ScorerSamplingConfig(sample_rate=0.5)
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B4. Verify Sessions and Run the Job
# MAGIC
# MAGIC All five scorers are now deployed (3 out-of-the-box + 1 UI-created + 1 custom code). Verify the traces exist and run the job once to evaluate everything together. Click each step to expand.
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
# MAGIC .lab-acc-item {
# MAGIC   border: 2px solid #e0e0e0;
# MAGIC   border-radius: 8px;
# MAGIC   margin-bottom: 8px;
# MAGIC   overflow: hidden;
# MAGIC   transition: border-color 0.2s;
# MAGIC }
# MAGIC .lab-acc-item.open { border-color: #2574B5; }
# MAGIC .lab-acc-header {
# MAGIC   display: flex;
# MAGIC   align-items: center;
# MAGIC   gap: 14px;
# MAGIC   padding: 14px 18px;
# MAGIC   cursor: pointer;
# MAGIC   user-select: none;
# MAGIC   background: #F9F7F4;
# MAGIC   transition: background 0.15s;
# MAGIC }
# MAGIC .lab-acc-item.open .lab-acc-header { background: #E8F1FA; }
# MAGIC .lab-acc-num {
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
# MAGIC .lab-acc-item.open .lab-acc-num { background: #2574B5; color: #fff; }
# MAGIC .lab-acc-summary {
# MAGIC   flex: 1;
# MAGIC   font-size: 13.5pt;
# MAGIC   font-weight: 600;
# MAGIC   color: #0b2026;
# MAGIC }
# MAGIC .lab-acc-chevron {
# MAGIC   font-size: 14pt;
# MAGIC   color: #888;
# MAGIC   transition: transform 0.25s;
# MAGIC }
# MAGIC .lab-acc-item.open .lab-acc-chevron { transform: rotate(180deg); color: #2574B5; }
# MAGIC .lab-acc-body {
# MAGIC   max-height: 0;
# MAGIC   opacity: 0;
# MAGIC   overflow: hidden;
# MAGIC   transition: max-height 0.3s ease, opacity 0.25s ease, padding 0.25s ease;
# MAGIC   padding: 0 18px 0 64px;
# MAGIC   font-size: 14pt;
# MAGIC   line-height: 1.7;
# MAGIC   color: #333;
# MAGIC }
# MAGIC .lab-acc-item.open .lab-acc-body {
# MAGIC   max-height: 600px;
# MAGIC   opacity: 1;
# MAGIC   padding: 4px 18px 18px 64px;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <div id="lab-acc-b4"></div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var LAB_B4 = [
# MAGIC   {
# MAGIC     summary: 'Verify sessions in the Experiment UI',
# MAGIC     detail: 'Navigate to <strong>Experiments</strong> in the left menu. Search for and click on <strong>evaluation-and-monitoring-training</strong>. Click on <strong>Sessions</strong> and note that your sessions have been populated. Click the dropdown next to a session to view its 3 turns.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Find the Trace Metrics Computation Job',
# MAGIC     detail: 'Navigate to <strong>Jobs &amp; Pipelines</strong> and search for <em>Trace Metrics Computation Job</em>. Your job has the experiment ID prepended, e.g., <strong>[0123456789] Trace Metrics Computation Job</strong>. It runs on a fixed ~15-minute schedule. Your <code>sample_rate</code> controls how many traces are evaluated, not how often the job runs.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Pause the schedule and run manually',
# MAGIC     detail: 'For this lab, click <strong>Pause</strong> to stop the automatic schedule. Then click <strong>Run now</strong>. This single run will evaluate traces using all five active judges: the three multi-turn programmatic judges, the <strong>professional</strong> judge you created via the UI, and the <strong>response_length</strong> custom scorer.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'Inspect the job output with Genie Code',
# MAGIC     detail: 'After the job completes, navigate to <strong>Runs</strong> and click the last successful run (green bar). Review the log output. Try <strong>Genie Code</strong> (top right next to your workspace name) and ask: <em>"What does this output tell me?"</em> Genie Code should respond with something like: <em>"In short, this output confirms that the agent monitoring metrics computation completed without errors for this run."</em> Read through the full response to understand what was evaluated.'
# MAGIC   },
# MAGIC   {
# MAGIC     summary: 'View scorer results on traces',
# MAGIC     detail: 'Navigate back to the experiment. Click on <strong>Sessions</strong> to see session-level feedback. note that <strong>UserFrustration</strong> and <strong>ConversationCompleteness</strong> results appear only at the session level (on the first trace of each session), not on individual traces. Click <strong>Traces</strong> to see per-trace assessments from <strong>Safety</strong>, <strong>professional</strong>, and <strong>response_length</strong>.<br><br><strong>Take a moment to read the judge feedback carefully.</strong> Do you agree with the assessments? For example, does the completeness judge correctly flag Session 2 (vague flight answers) as incomplete? Does the frustration judge catch Session 3 (ignored cancellation request)? Evaluating whether you agree with the judges is a critical step in tuning your evaluation strategy.'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC (function() {
# MAGIC   var container = document.getElementById('lab-acc-b4');
# MAGIC   LAB_B4.forEach(function(item, i) {
# MAGIC     var div = document.createElement('div');
# MAGIC     div.className = 'lab-acc-item' + (i === 0 ? ' open' : '');
# MAGIC     div.innerHTML =
# MAGIC       '<div class="lab-acc-header" onclick="labToggleB4(' + i + ')">'
# MAGIC       + '<div class="lab-acc-num">' + (i + 1) + '</div>'
# MAGIC       + '<div class="lab-acc-summary">' + item.summary + '</div>'
# MAGIC       + '<div class="lab-acc-chevron">&#9660;</div>'
# MAGIC       + '</div>'
# MAGIC       + '<div class="lab-acc-body">' + item.detail + '</div>';
# MAGIC     container.appendChild(div);
# MAGIC   });
# MAGIC })();
# MAGIC
# MAGIC function labToggleB4(idx) {
# MAGIC   var items = document.querySelectorAll('#lab-acc-b4 .lab-acc-item');
# MAGIC   items.forEach(function(el, i) {
# MAGIC     el.classList.toggle('open', i === idx && !el.classList.contains('open'));
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ### B5. Verify Your Scorers Are Running
# MAGIC
# MAGIC Confirm all five scorers were deployed correctly and are active. In production, this is the same audit step you would run to verify what is evaluating your traces and at what rates.
# MAGIC
# MAGIC **TODO**: Use `list_scorers()` to enumerate all registered scorers. You should see 5 active scorers.

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# Import list_scorers from mlflow.genai.scorers
# List all scorers and print their name and sample rate
#
# Hint: scorer objects have .name and .sample_rate
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> B5. Verify Scorers
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import list_scorers
# MAGIC
# MAGIC scorers = list_scorers()
# MAGIC print(f"Found {len(scorers)} registered scorers:\n")
# MAGIC for s in scorers:
# MAGIC     status = "ACTIVE" if s.sample_rate > 0 else "STOPPED"
# MAGIC     print(f"  {s.name}: rate={s.sample_rate} [{status}]")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### B6. Stop Deployed Scorers
# MAGIC You have confirmed the scorers work and seen their assessments. Before moving to Section C, stop the current scorers. In production, you would stop a scorer when you want to pause evaluation temporarily without losing the scorer's registration. The background job stops running once all scorers have `sample_rate = 0`.
# MAGIC
# MAGIC **TODO**: Stop all five scorers: `completeness`, `frustration`, `safety`, `length_scorer` (from B3), and the `professional` judge (use `get_scorer(name="professional")` since it was created via the UI). Note that `.stop()` returns a new Python object . Always reassign (e.g., `safety = safety.stop()`) so your local variable reflects the updated state.

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# Stop all 5 scorers:
# - completeness, frustration, safety (local variables from B1)
# - length_scorer (local variable from B3)
# - professional (use get_scorer(name="professional") to fetch it first)
# Note: .stop() returns a new Python object. reassign so your variable reflects the change
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> B6. Stop Deployed Scorers
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import get_scorer
# MAGIC
# MAGIC # Programmatic scorers (B1)
# MAGIC completeness = completeness.stop()
# MAGIC print(f"Sample rate for completeness: {completeness.sample_rate}")
# MAGIC frustration = frustration.stop()
# MAGIC print(f"Sample rate for frustration: {frustration.sample_rate}")
# MAGIC safety = safety.stop()
# MAGIC print(f"Sample rate for safety: {safety.sample_rate}")
# MAGIC
# MAGIC # Custom scorer (B3)
# MAGIC length_scorer = length_scorer.stop()
# MAGIC print(f"Sample rate for response_length: {length_scorer.sample_rate}")
# MAGIC
# MAGIC # UI-created scorer (B2). fetch with get_scorer since no local variable
# MAGIC professional = get_scorer(name="professional")
# MAGIC professional = professional.stop()
# MAGIC print(f"Sample rate for professional: {professional.sample_rate}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Sampling Strategy
# MAGIC
# MAGIC <div style="background: #1B5162; color: white; padding: 20px 24px; border-radius: 10px; margin: 12px 0;">
# MAGIC <div style="font-size: 11pt; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; margin-bottom: 6px;">Scenario</div>
# MAGIC <div style="font-size: 14pt; line-height: 1.5;">Your scorers have been running and the quality signals are stabilizing. Now you need to tune the sampling rates to balance cost and coverage for production.</div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. Update a Sampling Rate
# MAGIC
# MAGIC After running the judges for a while, you notice the completeness scorer's results have stabilized. the pass rate hasn't changed in the last several evaluation cycles. Continuing to evaluate 50% of traces is wasting LLM calls. You can reduce the rate to 5% to save costs while still catching regressions. This is the typical production lifecycle: start high for fast feedback, dial back once the signal is stable.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Return value pattern</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><code>.update()</code> changes the scorer's configuration on the server but returns a <strong>new Python object</strong> reflecting the change. Always reassign: <code>scorer = scorer.update(...)</code></p>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# 1. Use get_scorer() to fetch "conversation_completeness"
# 2. Update its sample_rate to 0.05
# 3. Print the original and updated rates to show the return value pattern
# 4. List all scorers to see the full picture
#
# Hint: from mlflow.genai.scorers import get_scorer, ScorerSamplingConfig
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> C1. Update and Verify
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import get_scorer, list_scorers, ScorerSamplingConfig
# MAGIC
# MAGIC # Fetch and update
# MAGIC original = get_scorer(name="conversation_completeness")
# MAGIC print(f"Original rate: {original.sample_rate}")
# MAGIC
# MAGIC updated = original.update(
# MAGIC     sampling_config=ScorerSamplingConfig(sample_rate=0.05)
# MAGIC )
# MAGIC print(f"Updated rate:  {updated.sample_rate}")
# MAGIC
# MAGIC # Full audit
# MAGIC print("\nAll scorers:")
# MAGIC for s in list_scorers():
# MAGIC     status = "ACTIVE" if s.sample_rate > 0 else "STOPPED"
# MAGIC     print(f"  {s.name}: rate={s.sample_rate} [{status}]")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Clean Up Scorers
# MAGIC In a shared workspace, abandoned scorers continue consuming resources. The background job keeps running, making LLM calls on every evaluation cycle. Deleting scorers you no longer need is important for cost management. Unlike `.stop()` (which pauses but keeps the scorer registered), `delete_scorer()` permanently removes it. Historical assessments already written to traces are preserved.
# MAGIC
# MAGIC **TODO**: List all scorers, delete each one using `delete_scorer(name=...)`, and verify the list is empty.

# COMMAND ----------

# TODO ──────────────────────────────────────────────────────────────────
# 1. List all scorers using list_scorers()
# 2. Delete each one using delete_scorer(name=...)
# 3. Verify the list is empty
#
# Hint: from mlflow.genai.scorers import list_scorers, delete_scorer
# ──────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> D. Clean Up Scorers
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import list_scorers, delete_scorer
# MAGIC
# MAGIC scorers = list_scorers()
# MAGIC print(f"Deleting {len(scorers)} scorers:\n")
# MAGIC
# MAGIC for s in scorers:
# MAGIC     print(f"  Deleting: {s.name}...", end=" ")
# MAGIC     delete_scorer(name=s.name)
# MAGIC     print("done")
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
# MAGIC 1. **Multi-turn judges** evaluate entire conversations using the `mlflow.trace.session` metadata to group traces into sessions
# MAGIC 2. **Custom scorers** use the `@scorer` decorator with all imports inline and support `aggregations` for numeric metrics
# MAGIC 3. **Sampling rates** should be tuned by scorer type: 100% for safety, 5% for expensive LLM judges once stable, 30-50% during active development
# MAGIC 4. **Return value pattern**: `.update()` and `.stop()` change the server state but return new Python objects -- always reassign so your local variable reflects the change
# MAGIC 5. **`list_scorers()`** lets you audit what is running and at what rates

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC