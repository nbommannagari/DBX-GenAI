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
# MAGIC # Lecture - MLflow's Evaluation Framework
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC MLflow provides a comprehensive evaluation framework specifically designed for generative AI applications, offering automated judges, tracing capabilities, and systematic assessment tools. This lecture explores MLflow's architecture, core components, and how they work together to enable rigorous agent evaluation.
# MAGIC
# MAGIC We'll examine the three fundamental components of MLflow evaluation, understand the `mlflow.genai.evaluate()` function, and explore how tracing provides the foundation for sophisticated evaluation capabilities.
# MAGIC
# MAGIC ##Learning Objectives
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Describe MLflow's evaluation framework architecture and key components
# MAGIC - Understand the role of evaluation datasets, scorers, and predict functions
# MAGIC - Explain how the `mlflow.genai.evaluate()` function orchestrates evaluation
# MAGIC - Understand the role of tracing in agent evaluation and debugging
# MAGIC - Recognize how Unity Gateway integration enables production monitoring

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. MLflow for Agent Evaluation

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. MLflow for Agent Evaluation
# MAGIC
# MAGIC During development, an **evaluation harness** helps you quickly execute your application for every record in your evaluation set and then run each output through your LLM judges and metric computations. In this module, we focus on the evaluation-specific capabilities that are purpose-built for assessing AI agents using Databricks AI Agent Evaluation.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 130" role="img" style="font-family: sans-serif;">
# MAGIC   <title>MLflow Evaluation Capabilities</title>
# MAGIC   <desc>The five MLflow capabilities most relevant to agent evaluation: Tracing, Scorers and Judges, mlflow.genai.evaluate(), Experiment Tracking for Eval, and Production Monitoring.</desc>
# MAGIC   <!-- Background -->
# MAGIC   <rect x="5" y="5" width="770" height="120" rx="14" fill="#F9F7F4" stroke="#1B3139" stroke-width="1.5" stroke-opacity="0.4"/>
# MAGIC   <text x="390" y="30" text-anchor="middle" font-size="18" font-weight="700" fill="#FF5F46">MLflow Evaluation Stack</text>
# MAGIC   <!-- Tiles -->
# MAGIC   <rect x="15" y="46" width="145" height="66" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="87" y="72" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">Tracing</text>
# MAGIC   <text x="87" y="92" text-anchor="middle" font-size="13" fill="#618794">Agent observability</text>
# MAGIC   <rect x="170" y="46" width="145" height="66" rx="8" fill="#00A972" fill-opacity="0.12" stroke="#00A972" stroke-width="1.5"/>
# MAGIC   <text x="242" y="72" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">Scorers &amp; Judges</text>
# MAGIC   <text x="242" y="92" text-anchor="middle" font-size="13" fill="#618794">Automated quality</text>
# MAGIC   <rect x="325" y="46" width="145" height="66" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="397" y="68" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">mlflow.genai</text>
# MAGIC   <text x="397" y="86" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">.evaluate()</text>
# MAGIC   <text x="397" y="104" text-anchor="middle" font-size="13" fill="#618794">Orchestration</text>
# MAGIC   <rect x="480" y="46" width="145" height="66" rx="8" fill="#1B5162" fill-opacity="0.12" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="552" y="72" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">Eval Experiments</text>
# MAGIC   <text x="552" y="92" text-anchor="middle" font-size="13" fill="#618794">Compare &amp; track</text>
# MAGIC   <rect x="635" y="46" width="130" height="66" rx="8" fill="#7B61FF" fill-opacity="0.12" stroke="#7B61FF" stroke-width="1.5"/>
# MAGIC   <text x="700" y="72" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">Monitoring</text>
# MAGIC   <text x="700" y="92" text-anchor="middle" font-size="13" fill="#618794">Production scoring</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1; font-size: 1.0em;">OpenTelemetry Compatible</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333; font-size: 14px;">MLflow traces are <a href="https://docs.databricks.com/aws/en/mlflow3/genai/tracing/integrations/open-telemetry/" target="_blank" style="color: #1976d2;">fully compatible with OpenTelemetry specs</a>, enabling export to external observability platforms (Datadog, Grafana, Splunk, etc.) via MLflow-only, OTel-only, or dual export modes.</p>
# MAGIC <p style="margin: 6px 0 0 0; font-size: 13px;">Docs: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/tracing/integrations/open-telemetry/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/integrations/open-telemetry/" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/integrations/open-telemetry/" target="_blank">GCP</a>)</p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Core Architecture

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ### B1. Three Fundamental Components
# MAGIC
# MAGIC MLflow's evaluation architecture centers on three components that flow into `mlflow.genai.evaluate()`:
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 200" role="img" style="font-family: sans-serif;">
# MAGIC   <title>MLflow Evaluation: Highlighting the Three Input Components</title>
# MAGIC   <desc>Evaluation Dataset, Scorers, and Predict Function are highlighted; evaluate() and EvaluationResult are greyed out.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="b1-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC     <marker id="b1-arrow-grey" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#ccc" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- HIGHLIGHTED: Evaluation Dataset -->
# MAGIC   <rect x="10" y="10" width="210" height="60" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="2"/>
# MAGIC   <text x="115" y="36" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Evaluation Dataset</text>
# MAGIC   <text x="115" y="56" text-anchor="middle" font-size="14" fill="#618794">inputs, expectations, outputs</text>
# MAGIC   <!-- HIGHLIGHTED: Scorers -->
# MAGIC   <rect x="10" y="85" width="210" height="60" rx="8" fill="#00A972" fill-opacity="0.12" stroke="#00A972" stroke-width="2"/>
# MAGIC   <text x="115" y="111" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Scorers (Judges)</text>
# MAGIC   <text x="115" y="131" text-anchor="middle" font-size="14" fill="#618794">built-in, custom, code-based</text>
# MAGIC   <!-- HIGHLIGHTED: Predict Function -->
# MAGIC   <rect x="10" y="160" width="210" height="35" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="2"/>
# MAGIC   <text x="115" y="183" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Predict Function</text>
# MAGIC   <!-- Arrows into evaluate -->
# MAGIC   <path d="M220 40 L290 85" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#b1-arrow)"/>
# MAGIC   <path d="M220 115 L290 100" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#b1-arrow)"/>
# MAGIC   <path d="M220 177 L290 115" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#b1-arrow)"/>
# MAGIC   <!-- GREYED: Central evaluate box -->
# MAGIC   <rect x="295" y="55" width="260" height="90" rx="12" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="425" y="88" text-anchor="middle" font-size="17" font-weight="700" fill="#bbb">mlflow.genai.evaluate()</text>
# MAGIC   <text x="425" y="110" text-anchor="middle" font-size="14" fill="#ccc">Orchestrates evaluation workflow</text>
# MAGIC   <text x="425" y="128" text-anchor="middle" font-size="14" fill="#ccc">Logs results to MLflow</text>
# MAGIC   <!-- GREYED: Arrow to results -->
# MAGIC   <path d="M555 100 L610 100" fill="none" stroke="#ccc" stroke-width="1.5" marker-end="url(#b1-arrow-grey)"/>
# MAGIC   <!-- GREYED: Results -->
# MAGIC   <rect x="615" y="55" width="155" height="90" rx="8" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="692" y="82" text-anchor="middle" font-size="16" font-weight="600" fill="#bbb">EvaluationResult</text>
# MAGIC   <text x="692" y="102" text-anchor="middle" font-size="14" fill="#ccc">run_id</text>
# MAGIC   <text x="692" y="118" text-anchor="middle" font-size="14" fill="#ccc">metrics (aggregated)</text>
# MAGIC   <text x="692" y="134" text-anchor="middle" font-size="14" fill="#ccc">tables (per-example)</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #2272B4;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#2272B4;">Evaluation Dataset</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC Your evaluation dataset defines what you're testing. Each record in an evaluation dataset represents a single test case with the following structure:
# MAGIC <ul style="margin:8px 0;">
# MAGIC <li><strong>inputs</strong> (required): The test input data that will be passed to your model or application</li>
# MAGIC <li><strong>outputs</strong> (optional): The actual outputs generated by your model, typically used for post-hoc evaluation</li>
# MAGIC <li><strong>expectations</strong> (optional): The expected outputs or quality criteria that define correct behavior (e.g., expected facts, expected responses, or per-row guidelines)</li>
# MAGIC <li><strong>source</strong> (optional): Provenance information about how this record was created</li>
# MAGIC <li><strong>tags</strong> (optional): Metadata specific to this individual record for organization and filtering</li>
# MAGIC </ul>
# MAGIC Datasets are typically stored as JSON files or Pandas DataFrames for easy manipulation and versioning.
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #00A972;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#00A972;">Scorers (Judges)</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC Scorers evaluate your agent's outputs against defined criteria. Choose the right type depending on how much customization and control you need:
# MAGIC <ul style="margin:8px 0;">
# MAGIC <li><strong>Built-in judges</strong> - Research-validated LLM-based assessments for common quality dimensions (Correctness, Safety, RelevanceToQuery, RetrievalGroundedness, etc.). This category also includes <strong>Guidelines</strong> and <strong>ExpectationsGuidelines</strong> judges that check whether responses pass or fail custom natural-language rules.</li>
# MAGIC <li><strong>Custom LLM judges</strong> - Fully customized LLM judges with detailed evaluation criteria and feedback optimization. Capable of returning numerical scores, categories, or boolean values.</li>
# MAGIC <li><strong>Code-based scorers</strong> - Programmatic and deterministic scorers for exact matching, format validation, and performance metrics.</li>
# MAGIC <li><strong>Third-party scorers</strong> - Specialized metrics from open-source evaluation frameworks.</li>
# MAGIC </ul>
# MAGIC Start with built-in judges for quick evaluation. As your needs evolve, build custom LLM judges for domain-specific criteria and create custom code-based scorers for deterministic business logic.
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #FF5F46;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#FF5F46;">Predict Function</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC The predict function generates outputs for your evaluation dataset. This can be:
# MAGIC <ul style="margin:8px 0;">
# MAGIC <li>Your agent's prediction method (for evaluating on-the-fly)</li>
# MAGIC <li>A lambda that transforms inputs to the format your agent expects</li>
# MAGIC <li>Omitted entirely if you're evaluating pre-generated outputs ("answer sheet" mode)</li>
# MAGIC </ul>
# MAGIC These three components come together in <code>mlflow.genai.evaluate()</code>, which orchestrates the evaluation process, collects metrics, and logs comprehensive results for analysis.
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. The `mlflow.genai.evaluate()` Function

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### C1. Central Orchestration Point
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 200" role="img" style="font-family: sans-serif;">
# MAGIC   <title>MLflow Evaluation: Highlighting evaluate()</title>
# MAGIC   <desc>mlflow.genai.evaluate() is highlighted; input components and EvaluationResult are greyed out.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="c1-arrow-grey" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#ccc" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC     <marker id="c1-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- GREYED: Evaluation Dataset -->
# MAGIC   <rect x="10" y="10" width="210" height="60" rx="8" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="115" y="36" text-anchor="middle" font-size="16" font-weight="600" fill="#bbb">Evaluation Dataset</text>
# MAGIC   <text x="115" y="56" text-anchor="middle" font-size="14" fill="#ccc">inputs, expectations, outputs</text>
# MAGIC   <!-- GREYED: Scorers -->
# MAGIC   <rect x="10" y="85" width="210" height="60" rx="8" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="115" y="111" text-anchor="middle" font-size="16" font-weight="600" fill="#bbb">Scorers (Judges)</text>
# MAGIC   <text x="115" y="131" text-anchor="middle" font-size="14" fill="#ccc">built-in, custom, code-based</text>
# MAGIC   <!-- GREYED: Predict Function -->
# MAGIC   <rect x="10" y="160" width="210" height="35" rx="8" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="115" y="183" text-anchor="middle" font-size="16" font-weight="600" fill="#bbb">Predict Function</text>
# MAGIC   <!-- Greyed arrows into evaluate -->
# MAGIC   <path d="M220 40 L290 85" fill="none" stroke="#ccc" stroke-width="1.5" marker-end="url(#c1-arrow-grey)"/>
# MAGIC   <path d="M220 115 L290 100" fill="none" stroke="#ccc" stroke-width="1.5" marker-end="url(#c1-arrow-grey)"/>
# MAGIC   <path d="M220 177 L290 115" fill="none" stroke="#ccc" stroke-width="1.5" marker-end="url(#c1-arrow-grey)"/>
# MAGIC   <!-- HIGHLIGHTED: Central evaluate box -->
# MAGIC   <rect x="295" y="55" width="260" height="90" rx="12" fill="#F9F7F4" stroke="#FF5F46" stroke-width="3"/>
# MAGIC   <text x="425" y="88" text-anchor="middle" font-size="17" font-weight="700" fill="#FF5F46">mlflow.genai.evaluate()</text>
# MAGIC   <text x="425" y="110" text-anchor="middle" font-size="14" fill="#618794">Orchestrates evaluation workflow</text>
# MAGIC   <text x="425" y="128" text-anchor="middle" font-size="14" fill="#618794">Logs results to MLflow</text>
# MAGIC   <!-- Greyed arrow to results -->
# MAGIC   <path d="M555 100 L610 100" fill="none" stroke="#ccc" stroke-width="1.5" marker-end="url(#c1-arrow-grey)"/>
# MAGIC   <!-- GREYED: Results -->
# MAGIC   <rect x="615" y="55" width="155" height="90" rx="8" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="692" y="82" text-anchor="middle" font-size="16" font-weight="600" fill="#bbb">EvaluationResult</text>
# MAGIC   <text x="692" y="102" text-anchor="middle" font-size="14" fill="#ccc">run_id</text>
# MAGIC   <text x="692" y="118" text-anchor="middle" font-size="14" fill="#ccc">metrics (aggregated)</text>
# MAGIC   <text x="692" y="134" text-anchor="middle" font-size="14" fill="#ccc">tables (per-example)</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC The `mlflow.genai.evaluate()` function provides an **evaluation harness**: a structured way to feed in test data, run your app, and automatically score the results.
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #FF5F46;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#FF5F46;">How it works</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC Instead of manually running your app and checking outputs one by one, MLflow Evaluation connects offline testing with production monitoring: the same evaluation logic you use in development can also run in production.
# MAGIC
# MAGIC <pre style="background:#f8f8f8;border:1px solid #e0e0e0;border-radius:6px;padding:12px;margin:10px 0;font-size:13px;overflow-x:auto;">
# MAGIC import mlflow
# MAGIC from mlflow.genai.scorers import Correctness
# MAGIC
# MAGIC results = mlflow.genai.evaluate(
# MAGIC     data=eval_dataset,              # EvaluationDataset, DataFrame, list[dict], etc.
# MAGIC     scorers=[Correctness()],        # Built-in and/or custom scorers
# MAGIC     predict_fn=my_app,              # Optional: direct evaluation
# MAGIC     # model_id="m-abc123...",        # Optional: link to LoggedModel
# MAGIC )
# MAGIC </pre>
# MAGIC
# MAGIC Docs: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-harness/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/eval-harness" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/eval-harness/" target="_blank">GCP</a>)
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### C2. Key Parameters
# MAGIC
# MAGIC <div style="max-width:900px;margin:0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 210" role="img" style="font-family: sans-serif;">
# MAGIC   <title>mlflow.genai.evaluate() Key Parameters</title>
# MAGIC   <desc>Four key parameters: data, scorers, predict_fn, and model_id flow into evaluate().</desc>
# MAGIC   <defs>
# MAGIC     <marker id="c2-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- data -->
# MAGIC   <rect x="10" y="10" width="330" height="42" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="20" y="36" font-size="15" font-weight="700" fill="#2272B4" font-family="monospace">data</text>
# MAGIC   <text x="68" y="36" font-size="14" fill="#0b2026">EvaluationDataset, DataFrame, list</text>
# MAGIC   <!-- scorers -->
# MAGIC   <rect x="10" y="60" width="330" height="42" rx="8" fill="#00A972" fill-opacity="0.12" stroke="#00A972" stroke-width="1.5"/>
# MAGIC   <text x="20" y="86" font-size="15" font-weight="700" fill="#00A972" font-family="monospace">scorers</text>
# MAGIC   <text x="90" y="86" font-size="14" fill="#0b2026">Built-in, custom, code-based</text>
# MAGIC   <!-- predict_fn -->
# MAGIC   <rect x="10" y="110" width="330" height="42" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="20" y="136" font-size="15" font-weight="700" fill="#FF5F46" font-family="monospace">predict_fn</text>
# MAGIC   <text x="115" y="136" font-size="14" fill="#0b2026">Agent/app function (optional)</text>
# MAGIC   <!-- model_id -->
# MAGIC   <rect x="10" y="160" width="330" height="42" rx="8" fill="#1B5162" fill-opacity="0.12" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="20" y="186" font-size="15" font-weight="700" fill="#1B5162" font-family="monospace">model_id</text>
# MAGIC   <text x="102" y="186" font-size="14" fill="#0b2026">LoggedModel linkage (optional)</text>
# MAGIC   <!-- Arrows -->
# MAGIC   <path d="M340 31 L420 90" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#c2-arrow)"/>
# MAGIC   <path d="M340 81 L420 98" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#c2-arrow)"/>
# MAGIC   <path d="M340 131 L420 108" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#c2-arrow)"/>
# MAGIC   <path d="M340 181 L420 118" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#c2-arrow)"/>
# MAGIC   <!-- evaluate box -->
# MAGIC   <rect x="425" y="65" width="240" height="80" rx="12" fill="#F9F7F4" stroke="#FF5F46" stroke-width="3"/>
# MAGIC   <text x="545" y="98" text-anchor="middle" font-size="17" font-weight="700" fill="#FF5F46">mlflow.genai.evaluate()</text>
# MAGIC   <text x="545" y="118" text-anchor="middle" font-size="14" fill="#618794">Orchestrates &amp; logs</text>
# MAGIC   <!-- Arrow out -->
# MAGIC   <path d="M665 105 L710 105" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#c2-arrow)"/>
# MAGIC   <!-- Result -->
# MAGIC   <rect x="715" y="80" width="60" height="50" rx="8" fill="#fff" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="745" y="102" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Result</text>
# MAGIC   <text x="745" y="118" text-anchor="middle" font-size="12" fill="#618794">metrics</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1; font-size: 1.0em;">predict_fn requirements</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333; font-size: 14px;">When provided, <code>predict_fn</code> must accept keys from the <code>inputs</code> dictionary as keyword arguments, return a JSON-serializable dictionary, be instrumented with MLflow Tracing, and emit exactly one trace per call.</p>
# MAGIC </div>
# MAGIC
# MAGIC Docs: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-harness/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/eval-harness" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/eval-harness/" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### C3. Evaluation Workflow
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="background:#2272B4;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:110px;">1. Data<br/>Loading<br/><span style="font-weight:400;font-size:13px;">Validate structure</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#2272B4;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:110px;">2. Output<br/>Generation<br/><span style="font-weight:400;font-size:13px;">Call predict_fn</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:110px;">3. Trace<br/>Creation<br/><span style="font-weight:400;font-size:13px;">Auto-instrumented</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:110px;">4. Scorer<br/>Execution<br/><span style="font-weight:400;font-size:13px;">Apply judges</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:110px;">5. Aggregate<br/>&amp; Log<br/><span style="font-weight:400;font-size:13px;">Metrics to MLflow</span></div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### C4. Return Value and Results Access
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 200" role="img" style="font-family: sans-serif;">
# MAGIC   <title>MLflow Evaluation: Highlighting EvaluationResult</title>
# MAGIC   <desc>EvaluationResult is highlighted; input components and evaluate() are greyed out.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="c4-arrow-grey" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#ccc" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC     <marker id="c4-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- GREYED: Evaluation Dataset -->
# MAGIC   <rect x="10" y="10" width="210" height="60" rx="8" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="115" y="36" text-anchor="middle" font-size="16" font-weight="600" fill="#bbb">Evaluation Dataset</text>
# MAGIC   <text x="115" y="56" text-anchor="middle" font-size="14" fill="#ccc">inputs, expectations, outputs</text>
# MAGIC   <!-- GREYED: Scorers -->
# MAGIC   <rect x="10" y="85" width="210" height="60" rx="8" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="115" y="111" text-anchor="middle" font-size="16" font-weight="600" fill="#bbb">Scorers (Judges)</text>
# MAGIC   <text x="115" y="131" text-anchor="middle" font-size="14" fill="#ccc">built-in, custom, code-based</text>
# MAGIC   <!-- GREYED: Predict Function -->
# MAGIC   <rect x="10" y="160" width="210" height="35" rx="8" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="115" y="183" text-anchor="middle" font-size="16" font-weight="600" fill="#bbb">Predict Function</text>
# MAGIC   <!-- Greyed arrows -->
# MAGIC   <path d="M220 40 L290 85" fill="none" stroke="#ccc" stroke-width="1.5" marker-end="url(#c4-arrow-grey)"/>
# MAGIC   <path d="M220 115 L290 100" fill="none" stroke="#ccc" stroke-width="1.5" marker-end="url(#c4-arrow-grey)"/>
# MAGIC   <path d="M220 177 L290 115" fill="none" stroke="#ccc" stroke-width="1.5" marker-end="url(#c4-arrow-grey)"/>
# MAGIC   <!-- GREYED: evaluate box -->
# MAGIC   <rect x="295" y="55" width="260" height="90" rx="12" fill="#F5F5F5" stroke="#ccc" stroke-width="1.5"/>
# MAGIC   <text x="425" y="88" text-anchor="middle" font-size="17" font-weight="700" fill="#bbb">mlflow.genai.evaluate()</text>
# MAGIC   <text x="425" y="110" text-anchor="middle" font-size="14" fill="#ccc">Orchestrates evaluation workflow</text>
# MAGIC   <text x="425" y="128" text-anchor="middle" font-size="14" fill="#ccc">Logs results to MLflow</text>
# MAGIC   <!-- Arrow to results (highlighted) -->
# MAGIC   <path d="M555 100 L610 100" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#c4-arrow)"/>
# MAGIC   <!-- HIGHLIGHTED: Results -->
# MAGIC   <rect x="615" y="55" width="155" height="90" rx="8" fill="#ffffff" stroke="#1B3139" stroke-width="3"/>
# MAGIC   <text x="692" y="82" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">EvaluationResult</text>
# MAGIC   <text x="692" y="102" text-anchor="middle" font-size="14" fill="#618794">run_id</text>
# MAGIC   <text x="692" y="118" text-anchor="middle" font-size="14" fill="#618794">metrics (aggregated)</text>
# MAGIC   <text x="692" y="134" text-anchor="middle" font-size="14" fill="#618794">tables (per-example)</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC The function returns an `EvaluationResult` object with `run_id`, `metrics` (aggregated dictionary), `artifacts`, and `tables` (per-example DataFrames). Per-example traces and feedback are accessed separately.
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #1B3139;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#0b2026;">Evaluation run structure</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC Each evaluation run contains:
# MAGIC <ul style="margin:8px 0;">
# MAGIC <li><strong>Traces</strong> - One trace for each input in your evaluation dataset</li>
# MAGIC <li><strong>Feedback</strong> - Quality assessments from scorers attached to each trace</li>
# MAGIC <li><strong>Metrics</strong> - Aggregate statistics (average score, pass rate, etc.)</li>
# MAGIC <li><strong>Metadata</strong> - Information about the evaluation configuration</li>
# MAGIC </ul>
# MAGIC The <code>EvaluationResult</code> object provides <strong>run_id</strong> and <strong>metrics</strong> (aggregated dictionary).
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #2272B4;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#2272B4;">Accessing per-example results</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC In MLflow 3, per-example results are accessed via traces:
# MAGIC <pre style="background:#f8f8f8;border:1px solid #e0e0e0;border-radius:6px;padding:12px;margin:10px 0;font-size:13px;overflow-x:auto;">
# MAGIC eval_traces = mlflow.search_traces(run_id=results.run_id, return_type="list")
# MAGIC
# MAGIC for trace in eval_traces:
# MAGIC     assessments = trace.info.assessments
# MAGIC     for assessment in assessments:
# MAGIC         print(f"Scorer: {assessment.name}, Value: {assessment.value}")
# MAGIC </pre>
# MAGIC Docs: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/evaluation-runs/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/evaluation-runs" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/evaluation-runs/" target="_blank">GCP</a>)
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. MLflow Tracing: The Foundation of Agent Observability

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### D1. Comprehensive Observability
# MAGIC
# MAGIC MLflow Tracing provides comprehensive observability into agent execution, capturing every step of your agent's reasoning process. Tracing is enabled when using `mlflow.genai.evaluate()` with properly instrumented predict functions.
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:10px;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="flex:1;min-width:130px;background:#2272B4;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Model Calls</div>
# MAGIC     <div style="font-size:14px;margin-top:4px;opacity:0.9;">Prompts, responses, parameters</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:130px;background:#00A972;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Tool Invocations</div>
# MAGIC     <div style="font-size:14px;margin-top:4px;opacity:0.9;">Inputs &amp; return values</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:130px;background:#FF5F46;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Retrieval Ops</div>
# MAGIC     <div style="font-size:14px;margin-top:4px;opacity:0.9;">Documents &amp; metadata</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:130px;background:#1B5162;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Timing</div>
# MAGIC     <div style="font-size:14px;margin-top:4px;opacity:0.9;">Duration per operation</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:130px;background:#7B61FF;color:#fff;padding:12px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:14px;">Hierarchy</div>
# MAGIC     <div style="font-size:14px;margin-top:4px;opacity:0.9;">Parent-child execution flow</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC Docs: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/tracing/span-concepts/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/span-concepts" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/span-concepts/" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### D2. Span Types in Traces
# MAGIC
# MAGIC MLflow organizes traces into spans with specific types. The most common in agent evaluation:
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:10px;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="border:2px solid #2272B4;border-radius:8px;padding:10px 14px;text-align:center;min-width:120px;background:#fff;">
# MAGIC     <div style="font-size:14px;font-weight:700;color:#2272B4;font-family:monospace;">CHAT_MODEL</div>
# MAGIC     <div style="font-size:13px;color:#618794;">LLM interactions</div>
# MAGIC   </div>
# MAGIC   <div style="border:2px solid #00A972;border-radius:8px;padding:10px 14px;text-align:center;min-width:120px;background:#fff;">
# MAGIC     <div style="font-size:14px;font-weight:700;color:#00A972;font-family:monospace;">AGENT</div>
# MAGIC     <div style="font-size:13px;color:#618794;">Autonomous agent ops</div>
# MAGIC   </div>
# MAGIC   <div style="border:2px solid #FF5F46;border-radius:8px;padding:10px 14px;text-align:center;min-width:120px;background:#fff;">
# MAGIC     <div style="font-size:14px;font-weight:700;color:#FF5F46;font-family:monospace;">TOOL</div>
# MAGIC     <div style="font-size:13px;color:#618794;">Function calls</div>
# MAGIC   </div>
# MAGIC   <div style="border:2px solid #1B5162;border-radius:8px;padding:10px 14px;text-align:center;min-width:120px;background:#fff;">
# MAGIC     <div style="font-size:14px;font-weight:700;color:#1B5162;font-family:monospace;">RETRIEVER</div>
# MAGIC     <div style="font-size:13px;color:#618794;">AI Search fetches</div>
# MAGIC   </div>
# MAGIC   <div style="border:2px solid #7B61FF;border-radius:8px;padding:10px 14px;text-align:center;min-width:120px;background:#fff;">
# MAGIC     <div style="font-size:14px;font-weight:700;color:#7B61FF;font-family:monospace;">CHAIN</div>
# MAGIC     <div style="font-size:13px;color:#618794;">Operation sequences</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1; font-size: 1.0em;">Full list of span types</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333; font-size: 14px;">MLflow supports additional span types including <code>LLM</code>, <code>EMBEDDING</code>, <code>GUARDRAIL</code>, <code>PARSER</code>, <code>RERANKER</code>, <code>EVALUATOR</code>, <code>MEMORY</code>, <code>TASK</code>, <code>WORKFLOW</code>, and <code>UNKNOWN</code>. See the complete reference: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/tracing/span-concepts/" target="_blank" style="color:#1976d2;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/span-concepts/" target="_blank" style="color:#1976d2;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/span-concepts/" target="_blank" style="color:#1976d2;">GCP</a>)</p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### D3. Why Tracing Matters for Evaluation
# MAGIC
# MAGIC **Why tracing matters for evaluation:**
# MAGIC
# MAGIC Certain evaluation judges, like `RetrievalSufficiency`, require traces to function. They analyze what was retrieved (not just the final response) to assess whether the retrieval system provided adequate information. Without traces, these sophisticated evaluations would be impossible.
# MAGIC
# MAGIC Tracing also enables debugging by allowing you to inspect exactly what happened during a failed evaluation, identifying whether issues stem from retrieval quality, tool selection, or LLM reasoning.
# MAGIC
# MAGIC Docs: (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/scorers" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/scorers/" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## E. Unity Gateway Integration and Production Monitoring
# MAGIC
# MAGIC When you deploy agents to Model Serving, Databricks enables **Unity Gateway-enabled inference tables**: creating a feedback loop from production back into evaluation.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:12px 16px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:140px;">Production Agent<br/><span style="font-weight:400;font-size:13px;">Logs interactions</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:12px 16px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:140px;">Inference Tables<br/><span style="font-weight:400;font-size:13px;">Rich trace data</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#2272B4;color:#fff;padding:12px 16px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:140px;">Extract Examples<br/><span style="font-weight:400;font-size:13px;">Failures &amp; edge cases</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:12px 16px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:140px;">Augment Dataset<br/><span style="font-weight:400;font-size:13px;">Real-world scenarios</span></div>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #1B5162;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#1B5162;">Inference table benefits</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC <ul style="margin:8px 0;">
# MAGIC <li><strong>Automatic logging</strong> - Every request captured without additional instrumentation</li>
# MAGIC <li><strong>Rich metadata</strong> - Request/response content, timestamps, latency, model versions, trace data</li>
# MAGIC <li><strong>Query interface</strong> - SQL-queryable tables in Unity Catalog</li>
# MAGIC <li><strong>Evaluation integration</strong> - Inference table data can be directly used as evaluation datasets</li>
# MAGIC </ul>
# MAGIC Docs: (<a href="https://docs.databricks.com/aws/en/ai-gateway/inference-tables/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/inference-tables" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/ai-gateway/inference-tables/" target="_blank">GCP</a>)
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC This approach ensures your evaluation dataset evolves with real user behavior rather than remaining static.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to learn more about MLflow's evaluation capabilities? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-1-3" style="flex: 1;">How does mlflow.genai.evaluate() work and what parameters does it accept?</span>
# MAGIC     <button onclick="
# MAGIC       var text = document.getElementById('genie-1-3').innerText;
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

# MAGIC %md
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC In this lecture, we explored MLflow's evaluation framework: a structured system for assessing GenAI applications built on three core components: evaluation datasets, scorers (judges), and predict functions. The `mlflow.genai.evaluate()` function acts as a central evaluation harness, orchestrating test execution, scorer application, and result logging in a single call.
# MAGIC
# MAGIC We also examined how MLflow Tracing provides the observability foundation that powers sophisticated evaluation judges, and how Unity Gateway-enabled inference tables close the loop between production monitoring and offline evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC
# MAGIC
# MAGIC