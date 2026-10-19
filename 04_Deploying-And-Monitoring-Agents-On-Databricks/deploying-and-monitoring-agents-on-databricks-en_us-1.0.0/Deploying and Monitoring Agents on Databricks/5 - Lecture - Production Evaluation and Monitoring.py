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
# MAGIC # Lecture - Production Evaluation and Monitoring
# MAGIC
# MAGIC ## Overview
# MAGIC This lecture covers the full spectrum of production evaluation and monitoring for generative AI applications on Databricks. It begins with the four scorer types available for automated quality assessment, progresses through multi-turn conversation judges for session-level evaluation, and then covers sampling strategies, the scorer lifecycle, management APIs, metric backfill for historical traces, and trace archival to Delta tables for long-term analysis.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lecture you will be able to:
# MAGIC - Differentiate the four scorer types and select the appropriate one for a given evaluation need
# MAGIC - Construct serializable custom scorers using the `@scorer` decorator
# MAGIC - Configure multi-turn judges for conversation-level quality assessment
# MAGIC - Design a sampling strategy that balances safety coverage, quality signal, and cost
# MAGIC - Manage scorers through their full lifecycle: register, start, update, stop, and delete
# MAGIC - Apply metric backfill to evaluate historical traces with new scorers
# MAGIC - Enable trace archival to stream production traces into Delta tables for SQL-based analysis
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Prerequisites</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">This lecture assumes familiarity with MLflow Tracing and agent deployment concepts from earlier modules in this course. You should understand how traces are logged to MLflow Experiments before proceeding.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Scorer Types
# MAGIC In MLflow, a **scorer** is the evaluation function you use to assess model or agent outputs, while a **judge** is a specific type of scorer that uses a Large Language Model (LLM) to perform the evaluation. Judges are LLM-powered and focus on subjective or nuanced criteria, whereas scorers can also be deterministic code-based functions for objective checks. Judges must be wrapped in scorers to be used in the evaluation harness or production monitoring. You can read more about the nuances between scorers and judges here ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/concepts/#scorers) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/concepts/) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/concepts/)).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. Built-in LLM Judges
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC MLflow provides **pre-built LLM judges** for common quality dimensions. These are ready-to-use scorers that require no custom prompt engineering. Each judge uses a Databricks-hosted LLM by default, or you can override with `model="databricks:/<endpoint>"`.
# MAGIC
# MAGIC The standard pattern for using any built-in judge is two steps: **register** (bind to an experiment) and **start** (begin evaluating traces):
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Registering and starting a Safety judge</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.genai.scorers <span class="kw">import</span> Safety, ScorerSamplingConfig
# MAGIC
# MAGIC <span class="cm"># 1. Register the scorer (binds to active experiment)</span>
# MAGIC safety_judge = <span class="fn">Safety</span>().register(name=<span class="st">"my_safety_judge"</span>)
# MAGIC
# MAGIC <span class="cm"># 2. Start with a sampling rate</span>
# MAGIC safety_judge = safety_judge.<span class="fn">start</span>(
# MAGIC     sampling_config=<span class="fn">ScorerSamplingConfig</span>(sample_rate=<span class="nm">0.7</span>)
# MAGIC )
# MAGIC
# MAGIC <span class="cm"># Custom model override</span>
# MAGIC custom = <span class="fn">Safety</span>(
# MAGIC     model=<span class="st">"databricks:/databricks-gpt-oss-20b"</span>
# MAGIC ).register(name=<span class="st">"custom_safety_judge"</span>)</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC The full set of built-in judges spans safety, response quality, RAG, tool calls, and guidelines:
# MAGIC
# MAGIC | Category | Judges | Ground truth? |
# MAGIC |---|---|---|
# MAGIC | **Safety** | `Safety` | No |
# MAGIC | **Response quality** | `RelevanceToQuery` | No |
# MAGIC | | `Correctness` | Yes |
# MAGIC | **RAG** | `RetrievalRelevance`, `RetrievalGroundedness` | No |
# MAGIC | | `RetrievalSufficiency` | Yes |
# MAGIC | **Tool calls** | `ToolCallEfficiency` | No |
# MAGIC | | `ToolCallCorrectness` | Optional |
# MAGIC | **Guidelines** | `Guidelines`, `ExpectationsGuidelines` | No |
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">All built-in judges accept an optional <code>model=</code> parameter to override the default LLM. Format: <code>"databricks:/&lt;endpoint-name&gt;"</code>. The <code>.register(name=...)</code> name must be unique within the experiment.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Pre-built judges and scorers ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/pre-built-judges-scorers) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/scorers) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/judges/pre-built-judges-scorers))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A2. Guidelines Judges
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC **Guidelines judges** evaluate traces against natural-language criteria you define. Each guideline is a pass/fail rule. The judge returns `Feedback` with value `"yes"` or `"no"` (strings, not booleans).
# MAGIC
# MAGIC - Takes a `name` and a `guidelines` list of plain-English rules
# MAGIC - Guidelines can reference **context variable keys** by name (e.g., `retrieved_documents`, `max_allowed_discount`)
# MAGIC - Optionally override the judge model with `model="databricks:/..."`
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Guidelines judge with custom rules</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.genai.scorers <span class="kw">import</span> Guidelines, ScorerSamplingConfig
# MAGIC
# MAGIC english_judge = <span class="fn">Guidelines</span>(
# MAGIC     name=<span class="st">"english"</span>,
# MAGIC     guidelines=[<span class="st">"The response must be in English"</span>]
# MAGIC ).register(name=<span class="st">"is_english"</span>)
# MAGIC
# MAGIC english_judge = english_judge.<span class="fn">start</span>(
# MAGIC     sampling_config=<span class="fn">ScorerSamplingConfig</span>(sample_rate=<span class="nm">0.7</span>)
# MAGIC )</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Guidelines judges are the fastest way to encode domain-specific quality rules without writing code. Write the rule in plain English and the LLM judge evaluates each trace against it.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Guidelines judges and scorer types ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/scorers) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/scorers))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A3. Custom Prompt Judges
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC Use `make_judge` from `mlflow.genai.judges` when you need **full control over the evaluation prompt**. This lets you write a custom template with specific instructions and define multi-level quality scales.
# MAGIC
# MAGIC - Only five template variables are allowed: `{{inputs}}`, `{{outputs}}`, `{{expectations}}`, `{{trace}}`, `{{conversation}}`. Custom variable names throw validation errors
# MAGIC - Specify quality levels with `feedback_value_type` using `Literal` types
# MAGIC - The `model` parameter is required when using `{{trace}}`-based judges; optional otherwise
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Custom prompt judge with make_judge()</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> typing <span class="kw">import</span> Literal
# MAGIC <span class="kw">from</span> mlflow.genai.judges <span class="kw">import</span> make_judge
# MAGIC
# MAGIC formality_judge = <span class="fn">make_judge</span>(
# MAGIC     name=<span class="st">"formality"</span>,
# MAGIC     instructions=<span class="st">"""Evaluate whether the response is
# MAGIC       formal, somewhat formal, or not formal.
# MAGIC       Request: {{inputs}}
# MAGIC       Response: {{outputs}}"""</span>,
# MAGIC     feedback_value_type=Literal[
# MAGIC         <span class="st">"formal"</span>, <span class="st">"semi_formal"</span>, <span class="st">"not_formal"</span>
# MAGIC     ],
# MAGIC ).register(name=<span class="st">"formality_judge"</span>)</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Custom prompt judges sit between guidelines judges (natural-language rules) and custom function scorers (arbitrary Python). Use them when you need a specific prompt structure or multi-level quality scales that go beyond pass/fail.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A4. Custom Function Scorers
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC The `@scorer` decorator turns any Python function into an MLflow-compatible evaluation function. This gives you **full control**: regex checks, schema validation, keyword matching, business rules: anything you can express in Python.
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: @scorer decorator with aggregations</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.genai.scorers <span class="kw">import</span> scorer
# MAGIC
# MAGIC <span class="dc">@scorer</span>
# MAGIC <span class="kw">def</span> <span class="fn">mentions_databricks</span>(outputs):
# MAGIC     <span class="kw">return</span> <span class="st">"databricks"</span> <span class="kw">in</span> <span class="fn">str</span>(outputs.get(<span class="st">"response"</span>, <span class="st">""</span>)).lower()
# MAGIC
# MAGIC <span class="dc">@scorer</span>(aggregations=[<span class="st">"mean"</span>, <span class="st">"min"</span>, <span class="st">"max"</span>])
# MAGIC <span class="kw">def</span> <span class="fn">response_length</span>(outputs):
# MAGIC     <span class="kw">return</span> <span class="fn">len</span>(<span class="fn">str</span>(outputs.get(<span class="st">"response"</span>, <span class="st">""</span>)))</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC **Serialization rules:** custom scorers run remotely and must be fully self-contained:
# MAGIC
# MAGIC | Rule | Details |
# MAGIC |---|---|
# MAGIC | `import` inline | Every import must be inside the function body |
# MAGIC | Notebook-defined | Must be defined in a Databricks notebook |
# MAGIC | No class-based scorers | `Scorer` subclasses cannot be serialized for remote execution |
# MAGIC | No external references | Cannot reference variables or modules defined outside the function |
# MAGIC | No import-requiring type hints | Signature hints cannot reference types that need imports |
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Use <code>aggregations=["mean", "min", "max"]</code> on numeric scorers to get experiment-level summary statistics. Aggregated metrics appear in the experiment view for trend analysis.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Custom scorers with @scorer ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-scorers) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/custom-scorers) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/custom-scorers))

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Multi-Turn Judges

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Conversation Judges
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC Multi-turn judges evaluate **entire conversations**, not individual requests. They operate on sessions (groups of traces that share the same `mlflow.trace.session` metadata value). Assessments attach to the **first trace** in the session.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:24px 0;flex-wrap:nowrap;overflow-x:auto;font-family:system-ui,sans-serif;padding-bottom:8px;">
# MAGIC   <div style="border:2px dashed #999;border-radius:12px;padding:18px 20px;display:flex;align-items:center;gap:10px;position:relative;flex-shrink:0;">
# MAGIC     <div style="position:absolute;top:-10px;left:14px;background:#fff;padding:0 6px;font-size:14px;font-weight:600;color:#999;letter-spacing:0.5px;text-transform:uppercase;">Session</div>
# MAGIC     <div style="background:#2574B5;color:#fff;padding:10px 18px;border-radius:6px;font-size:16px;font-weight:600;">Trace 1</div>
# MAGIC     <div style="background:#2574B5;color:#fff;padding:10px 18px;border-radius:6px;font-size:16px;font-weight:600;">Trace 2</div>
# MAGIC     <div style="background:#2574B5;color:#fff;padding:10px 18px;border-radius:6px;font-size:16px;font-weight:600;">Trace 3</div>
# MAGIC   </div>
# MAGIC   <div style="color:#999;font-size:24px;padding:0 10px;flex-shrink:0;">&#x2192;</div>
# MAGIC   <div style="background:#02A36F;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:140px;flex-shrink:0;">Multi-Turn<br/>Judge</div>
# MAGIC   <div style="color:#999;font-size:24px;padding:0 10px;flex-shrink:0;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:140px;flex-shrink:0;">Assessment<br/><span style="font-weight:400;font-size:14px;">on Trace 1</span></div>
# MAGIC </div>
# MAGIC
# MAGIC The built-in multi-turn judges:
# MAGIC
# MAGIC | Judge | What it evaluates |
# MAGIC |---|---|
# MAGIC | `ConversationCompleteness` | Whether the agent fully resolved the user's request across all turns |
# MAGIC | `UserFrustration` | Repeated questions, escalation language, confusion signals |
# MAGIC | `ConversationalSafety` | Safety violations that emerge only across multiple turns |
# MAGIC | `KnowledgeRetention` | Whether the agent correctly retains information from earlier in the conversation |
# MAGIC | `ConversationalGuidelines` | Whether responses comply with provided guidelines throughout the conversation |
# MAGIC | `ConversationalRoleAdherence` | Whether the agent maintains its assigned role throughout the conversation |
# MAGIC | `ConversationalToolCallEfficiency` | Whether tool usage across the conversation was efficient and appropriate |
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Multi-turn judge deployment</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.genai.scorers <span class="kw">import</span> (
# MAGIC     ConversationCompleteness, UserFrustration,
# MAGIC     ConversationalSafety, ScorerSamplingConfig,
# MAGIC )
# MAGIC
# MAGIC completeness = <span class="fn">ConversationCompleteness</span>().register(
# MAGIC     name=<span class="st">"conversation_completeness"</span>)
# MAGIC completeness = completeness.<span class="fn">start</span>(
# MAGIC     sampling_config=<span class="fn">ScorerSamplingConfig</span>(sample_rate=<span class="nm">1.0</span>)
# MAGIC )</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">A single-turn safety check might pass every response, but a user repeating the same question four times signals a broken experience only a multi-turn judge can detect. All multi-turn judges are built-in and require no external integrations. None require ground truth.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Session Configuration
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC Sessions are identified by the `mlflow.trace.session` **metadata field**. All traces sharing the same session ID are grouped into a single conversation for multi-turn evaluation.
# MAGIC
# MAGIC - Set via `mlflow.update_current_trace(session_id=..., user=...)` or equivalently via the `metadata=` dict
# MAGIC - Metadata is **immutable** once logged (unlike tags which can be updated)
# MAGIC - A conversation is considered complete after a configurable buffer (default **5 minutes** of inactivity)
# MAGIC - Configure the buffer via `MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS`
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Session metadata configuration</summary>
# MAGIC <div class="cb"><pre><span class="cm"># Set session metadata for multi-turn grouping</span>
# MAGIC <span class="cm"># Option 1: Dedicated parameters (recommended)</span>
# MAGIC mlflow.<span class="fn">update_current_trace</span>(
# MAGIC     session_id=session_id,
# MAGIC     user=user_id,
# MAGIC )
# MAGIC
# MAGIC <span class="cm"># Option 2: Equivalent using the metadata dict</span>
# MAGIC mlflow.<span class="fn">update_current_trace</span>(
# MAGIC     metadata={
# MAGIC         <span class="st">"mlflow.trace.session"</span>: session_id,
# MAGIC         <span class="st">"mlflow.trace.user"</span>: user_id,
# MAGIC     }
# MAGIC )</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Session tracking only requires <strong>metadata</strong> — do not set <code>mlflow.trace.session</code> in tags. The production monitoring service groups traces into conversations using the metadata field. The dedicated <code>session_id=</code> parameter stores the value in metadata under the <code>mlflow.trace.session</code> key automatically.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Session metadata and multi-turn configuration ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/production-monitoring) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/production-monitoring))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/deploying-and-monitoring-agents-on-databricks-en_us-1.0.0/images/20261010T172734Z/Deploying and Monitoring Agents on Databricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to learn more about <strong>multi-turn conversation evaluation</strong>? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/deploying-and-monitoring-agents-on-databricks-en_us-1.0.0/images/20261010T172734Z/Deploying and Monitoring Agents on Databricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-query" style="flex: 1;">How do multi-turn conversation judges work in MLflow? How do I configure sessions for conversation-level evaluation?</span>
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

# MAGIC %md
# MAGIC ## C. Online Evaluation

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. Sampling Strategy
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC Not every scorer needs to evaluate every trace. A well-designed sampling strategy balances **coverage**, **quality signal**, and **cost**. The general principle: the higher the risk of missing an issue, the higher the sampling rate should be.
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:16px;margin:24px 0;flex-wrap:nowrap;overflow-x:auto;font-family:system-ui,sans-serif;padding-bottom:8px;">
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:24px 32px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:190px;display:flex;flex-direction:column;justify-content:center;box-shadow:0 4px 12px rgba(255,95,70,0.35);border:2px solid #cc4030;flex-shrink:0;">Safety<br/><span style="font-weight:400;font-size:28px;">100%</span><br/><span style="font-weight:400;font-size:14px;">Every trace evaluated</span></div>
# MAGIC   <div style="background:#02A36F;color:#fff;padding:18px 24px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:160px;display:flex;flex-direction:column;justify-content:center;flex-shrink:0;">Quality<br/><span style="font-weight:400;font-size:24px;">5-10%</span><br/><span style="font-weight:400;font-size:14px;">Statistically meaningful</span></div>
# MAGIC   <div style="background:#2574B5;color:#fff;padding:18px 24px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:160px;display:flex;flex-direction:column;justify-content:center;flex-shrink:0;">Custom<br/><span style="font-weight:400;font-size:24px;">Variable</span><br/><span style="font-weight:400;font-size:14px;">Adjust per use case</span></div>
# MAGIC </div>
# MAGIC
# MAGIC - **Critical (100%)**: Safety and security scorers should run on every trace. The cost of missing a safety violation far outweighs evaluation cost.
# MAGIC - **Expensive (5-10%)**: LLM-based judges that make their own model calls are expensive per evaluation. 5% still provides statistically meaningful quality signals at scale.
# MAGIC - **Custom (variable)**: Lightweight function scorers can run at higher rates. Combine single-turn and multi-turn judges at different rates.
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Auditing active scorers with list_scorers()</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.genai.scorers <span class="kw">import</span> list_scorers
# MAGIC
# MAGIC <span class="cm"># Audit all active scorers and their rates</span>
# MAGIC <span class="kw">for</span> s <span class="kw">in</span> <span class="fn">list_scorers</span>():
# MAGIC     <span class="kw">if</span> s.sample_rate > <span class="nm">0</span>:
# MAGIC         <span class="fn">print</span>(<span class="st">f"{s.name} active at {s.sample_rate}"</span>)</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Maximum <strong>20 scorers</strong> per experiment. Use <code>list_scorers()</code> to audit what is running and at what rates. <code>ScorerSamplingConfig(sample_rate=X)</code> can be changed at any time via <code>.update()</code>.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C2. Scorer Lifecycle
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC Every scorer moves through a defined set of states. Understanding these transitions is essential for managing scorers in production.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;gap:0;margin:24px 0;flex-wrap:nowrap;overflow-x:auto;font-family:system-ui,sans-serif;padding-bottom:8px;">
# MAGIC   <div style="background:#e8e8e8;color:#666;padding:10px 14px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:110px;flex-shrink:0;">Created<br/><span style="font-weight:400;font-size:14px;">In code only</span></div>
# MAGIC   <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 4px;flex-shrink:0;">
# MAGIC     <div style="color:#999;font-size:24px;">&#x2192;</div>
# MAGIC     <div style="color:#999;font-size:14px;">.register()</div>
# MAGIC   </div>
# MAGIC   <div style="background:#E3F2FD;color:#1565C0;padding:10px 14px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:110px;flex-shrink:0;">Registered<br/><span style="font-weight:400;font-size:14px;">Known to experiment</span></div>
# MAGIC   <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 4px;flex-shrink:0;">
# MAGIC     <div style="color:#999;font-size:24px;">&#x2192;</div>
# MAGIC     <div style="color:#999;font-size:14px;">.start()</div>
# MAGIC   </div>
# MAGIC   <div style="background:#E8F5E9;color:#2E7D32;padding:10px 14px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:110px;flex-shrink:0;">Running<br/><span style="font-weight:400;font-size:14px;">Evaluating traces</span></div>
# MAGIC   <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 4px;flex-shrink:0;">
# MAGIC     <div style="color:#999;font-size:14px;">.stop()</div>
# MAGIC     <div style="color:#999;font-size:24px;">&#x21C4;</div>
# MAGIC     <div style="color:#999;font-size:14px;">.start()</div>
# MAGIC   </div>
# MAGIC   <div style="background:#FFF3E0;color:#E65100;padding:10px 14px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:110px;flex-shrink:0;">Stopped<br/><span style="font-weight:400;font-size:14px;">Paused, restartable</span></div>
# MAGIC   <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 4px;flex-shrink:0;">
# MAGIC     <div style="color:#999;font-size:24px;">&#x2192;</div>
# MAGIC     <div style="color:#999;font-size:14px;">delete_scorer()</div>
# MAGIC   </div>
# MAGIC   <div style="background:#FFEBEE;color:#C62828;padding:10px 14px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:110px;flex-shrink:0;">Deleted<br/><span style="font-weight:400;font-size:14px;">Permanent removal</span></div>
# MAGIC </div>
# MAGIC
# MAGIC - **register**: Binds the scorer to an MLflow experiment. The name must be unique within the experiment. Appears in the Scorers tab but does not evaluate traces yet.
# MAGIC - **start**: Activates the scorer with a `ScorerSamplingConfig`. It begins evaluating new traces at the specified sample rate.
# MAGIC - **stop**: Pauses evaluation. The scorer remains registered and can be restarted. Historical assessments are preserved.
# MAGIC - **update**: Changes the sampling configuration on a running scorer. Returns a new instance.
# MAGIC - **delete**: Permanently removes the scorer. Must be stopped first. Historical assessments remain on their traces.
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Full scorer lifecycle (register, start, update, stop, delete)</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.genai.scorers <span class="kw">import</span> Safety, ScorerSamplingConfig, delete_scorer
# MAGIC
# MAGIC <span class="cm"># Full lifecycle</span>
# MAGIC safety = <span class="fn">Safety</span>().register(name=<span class="st">"safety_v1"</span>)
# MAGIC safety = safety.<span class="fn">start</span>(sampling_config=<span class="fn">ScorerSamplingConfig</span>(sample_rate=<span class="nm">1.0</span>))
# MAGIC
# MAGIC <span class="cm"># Update sampling rate</span>
# MAGIC safety = safety.<span class="fn">update</span>(sampling_config=<span class="fn">ScorerSamplingConfig</span>(sample_rate=<span class="nm">0.5</span>))
# MAGIC
# MAGIC <span class="cm"># Stop (preserves history)</span>
# MAGIC safety = safety.<span class="fn">stop</span>()
# MAGIC
# MAGIC <span class="cm"># Restart from stopped state</span>
# MAGIC safety = safety.<span class="fn">start</span>(sampling_config=<span class="fn">ScorerSamplingConfig</span>(sample_rate=<span class="nm">1.0</span>))
# MAGIC
# MAGIC <span class="cm"># Permanent removal</span>
# MAGIC safety = safety.<span class="fn">stop</span>()
# MAGIC <span class="fn">delete_scorer</span>(name=<span class="st">"safety_v1"</span>)</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>Immutability pattern:</strong> Every lifecycle method (<code>.register()</code>, <code>.start()</code>, <code>.stop()</code>, <code>.update()</code>) returns a <strong>new instance</strong>. Always reassign: <code>scorer = scorer.start(...)</code>. If you call <code>scorer.stop()</code> without reassigning, the variable still points to the old (running) state.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Scorer Management

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D1. get_scorer()
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC Use `get_scorer()` to retrieve any registered scorer by name. This is essential when you need to manage a scorer that was created in a different notebook session or through the Databricks UI.
# MAGIC
# MAGIC - Returns the scorer object with its current state, sample rate, and configuration
# MAGIC - Works with UI-created scorers and API-created scorers alike
# MAGIC - The returned object supports all lifecycle methods (`.start()`, `.stop()`, `.update()`)
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Retrieving and updating a scorer by name</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.genai.scorers <span class="kw">import</span> get_scorer, ScorerSamplingConfig
# MAGIC
# MAGIC <span class="cm"># Fetch a scorer created in the UI or another session</span>
# MAGIC my_scorer = <span class="fn">get_scorer</span>(name=<span class="st">"professional"</span>)
# MAGIC <span class="fn">print</span>(<span class="st">f"Name: {my_scorer.name}"</span>)
# MAGIC <span class="fn">print</span>(<span class="st">f"Rate: {my_scorer.sample_rate}"</span>)
# MAGIC
# MAGIC <span class="cm"># Update the sampling rate</span>
# MAGIC updated = my_scorer.<span class="fn">update</span>(
# MAGIC     sampling_config=<span class="fn">ScorerSamplingConfig</span>(sample_rate=<span class="nm">0.8</span>)
# MAGIC )
# MAGIC
# MAGIC <span class="cm"># Original remains unchanged (immutable)</span>
# MAGIC <span class="fn">print</span>(<span class="st">f"Original: {my_scorer.sample_rate}"</span>)
# MAGIC <span class="fn">print</span>(<span class="st">f"Updated:  {updated.sample_rate}"</span>)</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><code>get_scorer()</code> resolves against the active MLflow experiment. Make sure <code>mlflow.set_experiment()</code> points to the correct experiment before calling it.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D2. Stopping and Deleting
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC **Stopping** pauses a running scorer. It remains registered and can be restarted at any time. The background Trace Metrics Computation Job continues running until all scorers in the experiment are stopped.
# MAGIC
# MAGIC **Deleting** permanently removes a scorer from the experiment. The scorer must be stopped before deletion. Historical assessments produced by the scorer are preserved on their traces.
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Stopping and restarting individual scorers</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.genai.scorers <span class="kw">import</span> get_scorer, list_scorers, delete_scorer, ScorerSamplingConfig
# MAGIC
# MAGIC <span class="cm"># Stop a scorer (always reassign)</span>
# MAGIC safety = safety.<span class="fn">stop</span>()
# MAGIC
# MAGIC <span class="cm"># Stop a scorer fetched by name</span>
# MAGIC ui_scorer = <span class="fn">get_scorer</span>(name=<span class="st">"professional"</span>)
# MAGIC ui_scorer = ui_scorer.<span class="fn">stop</span>()
# MAGIC
# MAGIC <span class="cm"># Resume later</span>
# MAGIC safety = safety.<span class="fn">start</span>(
# MAGIC     sampling_config=<span class="fn">ScorerSamplingConfig</span>(sample_rate=<span class="nm">1.0</span>)
# MAGIC )</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Bulk stop and delete cleanup</summary>
# MAGIC <div class="cb"><pre><span class="cm"># Bulk cleanup: stop and delete all scorers</span>
# MAGIC <span class="kw">for</span> s <span class="kw">in</span> <span class="fn">list_scorers</span>():
# MAGIC     s = s.<span class="fn">stop</span>()
# MAGIC     <span class="fn">delete_scorer</span>(name=s.name)
# MAGIC
# MAGIC <span class="cm"># Verify</span>
# MAGIC <span class="fn">print</span>(<span class="fn">list_scorers</span>())  <span class="cm"># []</span></pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>.stop() vs. delete_scorer():</strong> Stopping pauses evaluation but keeps the scorer registered (it can be restarted). Deleting permanently removes it. Historical assessments are preserved in both cases.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Backfill and Archival

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E1. Metric Backfill
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC **Metric backfill** applies scorers to traces logged before the scorer existed. This lets you retroactively evaluate historical traces with new quality checks without reprocessing the original requests.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:24px 0;flex-wrap:nowrap;overflow-x:auto;font-family:system-ui,sans-serif;padding-bottom:8px;">
# MAGIC   <div style="background:#1B5162;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:140px;flex-shrink:0;">Historical<br/>Traces</div>
# MAGIC   <div style="color:#999;font-size:24px;padding:0 10px;flex-shrink:0;">&#x2192;</div>
# MAGIC   <div style="background:#02A36F;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:140px;flex-shrink:0;">Backfill<br/>Job</div>
# MAGIC   <div style="color:#999;font-size:24px;padding:0 10px;flex-shrink:0;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:140px;flex-shrink:0;">Assessments<br/><span style="font-weight:400;font-size:14px;">Attached to traces</span></div>
# MAGIC </div>
# MAGIC <div style="display:flex;justify-content:center;margin:-12px 0 24px 0;font-family:system-ui,sans-serif;">
# MAGIC   <div style="color:#999;font-size:13px;font-style:italic;">Assessments are written back onto the original historical traces</div>
# MAGIC </div>
# MAGIC
# MAGIC - `backfill_scorers` is from `databricks.agents.scorers`
# MAGIC - Creates a **[experiment_id] Metric Backfill Computation Job** in Jobs and Pipelines
# MAGIC - Pass a list of scorer name strings to use their current sample rates, or use `BackfillScorerConfig` for custom rates
# MAGIC - Returns a `job_id` for tracking progress
# MAGIC - Only one backfill job can run per experiment at a time (`RESOURCE_CONFLICT` error if you try to start a second)
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Backfill with BackfillScorerConfig</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> databricks.agents.scorers <span class="kw">import</span> backfill_scorers, BackfillScorerConfig
# MAGIC <span class="kw">from</span> datetime <span class="kw">import</span> datetime, timedelta
# MAGIC
# MAGIC <span class="cm"># Simple backfill using registered sample rates</span>
# MAGIC job_id = <span class="fn">backfill_scorers</span>(
# MAGIC     scorers=[<span class="st">"safety_check"</span>, <span class="st">"response_length"</span>]
# MAGIC )
# MAGIC
# MAGIC <span class="cm"># Custom sample rates with time range</span>
# MAGIC job_id = <span class="fn">backfill_scorers</span>(
# MAGIC     experiment_id=YOUR_EXPERIMENT_ID,
# MAGIC     scorers=[
# MAGIC         <span class="fn">BackfillScorerConfig</span>(scorer=safety_judge, sample_rate=<span class="nm">0.8</span>),
# MAGIC         <span class="fn">BackfillScorerConfig</span>(scorer=response_length, sample_rate=<span class="nm">0.9</span>),
# MAGIC     ],
# MAGIC     start_time=datetime.now() - timedelta(days=<span class="nm">7</span>),
# MAGIC     end_time=datetime.now(),
# MAGIC )</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>Best practices:</strong> Test with a narrow time range first. Use lower sample rates for expensive judges during large backfills. Verify scorer names with <code>list_scorers()</code> before starting. Track progress via the <code>job_id</code> in the Jobs UI.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E2. Trace Archival
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC **Trace archival** streams production traces from an MLflow Experiment into a Unity Catalog Delta table. Once archived, traces can be queried with SQL, used in Lakeview dashboards, or joined with business data for long-term analysis.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:24px 0;flex-wrap:nowrap;overflow-x:auto;font-family:system-ui,sans-serif;padding-bottom:8px;">
# MAGIC   <div style="background:#2574B5;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:140px;flex-shrink:0;">MLflow<br/>Experiment</div>
# MAGIC   <div style="color:#999;font-size:24px;padding:0 10px;flex-shrink:0;">&#x2192;</div>
# MAGIC   <div style="background:#02A36F;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:140px;flex-shrink:0;">Archival<br/>Job</div>
# MAGIC   <div style="color:#999;font-size:24px;padding:0 10px;flex-shrink:0;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:140px;flex-shrink:0;">Delta<br/>Table</div>
# MAGIC   <div style="color:#999;font-size:24px;padding:0 10px;flex-shrink:0;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:16px;text-align:center;min-width:140px;flex-shrink:0;">SQL /<br/>Dashboards</div>
# MAGIC </div>
# MAGIC
# MAGIC - Uses `mlflow.tracing.archival` module
# MAGIC - Creates a **[experiment_id] Trace Archive Job** that continuously streams traces
# MAGIC - The Delta table is created automatically if it does not exist
# MAGIC - Traces are written in **append-only** mode. Existing data is never overwritten
# MAGIC - Can also be enabled from the MLflow Experiment UI (click "Delta sync: Not enabled")
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Enabling and disabling trace archival to Delta</summary>
# MAGIC <div class="cb"><pre><span class="kw">from</span> mlflow.tracing.archival <span class="kw">import</span> (
# MAGIC     enable_databricks_trace_archival,
# MAGIC     disable_databricks_trace_archival,
# MAGIC )
# MAGIC
# MAGIC <span class="cm"># Enable archival to a Delta table</span>
# MAGIC <span class="fn">enable_databricks_trace_archival</span>(
# MAGIC     delta_table_fullname=<span class="st">"my_catalog.my_schema.archived_traces"</span>,
# MAGIC     experiment_id=<span class="st">"YOUR_EXPERIMENT_ID"</span>,
# MAGIC )
# MAGIC
# MAGIC <span class="cm"># Disable (existing data preserved, new traces stop flowing)</span>
# MAGIC <span class="fn">disable_databricks_trace_archival</span>(
# MAGIC     experiment_id=<span class="st">"YOUR_EXPERIMENT_ID"</span>
# MAGIC )</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Disabling archival stops the Trace Archive Job but <strong>preserves all existing data</strong> in the Delta table. Archival can be re-enabled at any time. The Delta table schema is managed automatically: no manual schema definition is needed.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Trace archival to Delta tables ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/production-monitoring) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/production-monitoring))

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC