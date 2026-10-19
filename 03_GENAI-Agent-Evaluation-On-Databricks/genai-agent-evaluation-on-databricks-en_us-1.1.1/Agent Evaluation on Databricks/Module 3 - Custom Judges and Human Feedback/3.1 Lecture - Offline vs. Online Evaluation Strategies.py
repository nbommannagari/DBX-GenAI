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
# MAGIC # Lecture - Offline vs. Online Evaluation Strategies
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC Effective agent evaluation requires both offline validation using curated datasets and online monitoring of production performance. This lecture explores the complementary roles of offline and online evaluation, their respective strengths and limitations, and how to create feedback loops between them.
# MAGIC
# MAGIC We'll examine when to use each approach, understand their implementation patterns, and learn how to build evaluation systems that evolve with real-world usage while maintaining rigorous quality standards.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Identify when to use offline versus online evaluation strategies
# MAGIC - Understand the strengths and limitations of each approach
# MAGIC - Describe how offline and online evaluation complement each other
# MAGIC - Explain how to create feedback loops between production and evaluation
# MAGIC - Recognize best practices for comprehensive evaluation strategies

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. Evaluation Strategy Overview

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### A1. Two Complementary Approaches
# MAGIC
# MAGIC <div style="display:flex;justify-content:center;gap:16px;margin:18px 0;font-family:sans-serif;flex-wrap:wrap;">
# MAGIC   <div style="flex:1;min-width:200px;max-width:220px;border:2px solid #2272B4;border-radius:10px;padding:16px;text-align:center;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#2272B4;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:12px;">Offline Evaluation</div>
# MAGIC     <div style="background:#2272B4;color:#fff;border-radius:6px;padding:8px;font-size:14px;font-weight:600;margin-bottom:6px;">Curated Datasets</div>
# MAGIC     <div style="color:#999;font-size:16px;">&#x2193;</div>
# MAGIC     <div style="background:#F9F7F4;border:1px solid #2272B4;border-radius:6px;padding:8px;font-size:14px;margin-bottom:6px;">Agent Execution</div>
# MAGIC     <div style="color:#999;font-size:16px;">&#x2193;</div>
# MAGIC     <div style="background:#F9F7F4;border:1px solid #2272B4;border-radius:6px;padding:8px;font-size:14px;margin-bottom:6px;">Scorer Application</div>
# MAGIC     <div style="color:#999;font-size:16px;">&#x2193;</div>
# MAGIC     <div style="background:#2272B4;color:#fff;border-radius:6px;padding:8px;font-size:14px;font-weight:600;">Pre-deployment Gate</div>
# MAGIC     <div style="margin-top:10px;font-size:13px;color:#618794;">Controlled &amp; reproducible</div>
# MAGIC   </div>
# MAGIC   <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:80px;">
# MAGIC     <div style="font-size:14px;color:#618794;text-align:center;line-height:1.4;">Feedback<br/>Loop</div>
# MAGIC     <div style="color:#FF5F46;font-size:24px;">&#x21C4;</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:200px;max-width:220px;border:2px solid #00A972;border-radius:10px;padding:16px;text-align:center;">
# MAGIC     <div style="font-size:15px;font-weight:700;color:#00A972;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:12px;">Online Evaluation</div>
# MAGIC     <div style="background:#00A972;color:#fff;border-radius:6px;padding:8px;font-size:14px;font-weight:600;margin-bottom:6px;">Production Traffic</div>
# MAGIC     <div style="color:#999;font-size:16px;">&#x2193;</div>
# MAGIC     <div style="background:#F9F7F4;border:1px solid #00A972;border-radius:6px;padding:8px;font-size:14px;margin-bottom:6px;">MLflow Traces</div>
# MAGIC     <div style="color:#999;font-size:16px;">&#x2193;</div>
# MAGIC     <div style="background:#F9F7F4;border:1px solid #00A972;border-radius:6px;padding:8px;font-size:14px;margin-bottom:6px;">Automatic Scoring</div>
# MAGIC     <div style="color:#999;font-size:16px;">&#x2193;</div>
# MAGIC     <div style="background:#00A972;color:#fff;border-radius:6px;padding:8px;font-size:14px;font-weight:600;">Monitoring &amp; Alerts</div>
# MAGIC     <div style="margin-top:10px;font-size:13px;color:#618794;">Real-world &amp; continuous</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC Agent evaluation requires both controlled validation and real-world monitoring. Offline and online evaluation serve different but complementary purposes in ensuring agent quality throughout the development and deployment lifecycle.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Offline Evaluation: Pre-Deployment Validation

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### B1. Offline Evaluation Workflow
# MAGIC
# MAGIC Offline evaluation tests your agent using curated datasets before deployment. Controlled, reproducible, and gated.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="background:#2272B4;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:100px;">1. Curate<br/>Dataset<br/><span style="font-weight:400;font-size:13px;">Use cases &amp;<br/>edge cases</span></div>
# MAGIC   <div style="color:#999;font-size:16px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#2272B4;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:100px;">2. Define<br/>Expectations<br/><span style="font-weight:400;font-size:13px;">Optional ground<br/>truth/guidelines</span></div>
# MAGIC   <div style="color:#999;font-size:16px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:100px;">3. Run<br/>Agent<br/><span style="font-weight:400;font-size:13px;">Generate<br/>responses</span></div>
# MAGIC   <div style="color:#999;font-size:16px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:100px;">4. Apply<br/>Scorers<br/><span style="font-weight:400;font-size:13px;">Multiple<br/>judges</span></div>
# MAGIC   <div style="color:#999;font-size:16px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:100px;">5. Analyze<br/>&amp; Iterate<br/><span style="font-weight:400;font-size:13px;">Compare<br/>results</span></div>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #1B5162;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#1B5162;">Strengths and limitations</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC <div style="display:flex;gap:16px;flex-wrap:wrap;">
# MAGIC <div style="flex:1;min-width:200px;">
# MAGIC <strong style="color:#00A972;">Strengths</strong>
# MAGIC <ul style="margin:6px 0;"><li>Rigorous validation before users encounter the agent</li><li>A/B testing different configurations</li><li>Baseline metrics for production</li></ul>
# MAGIC </div>
# MAGIC <div style="flex:1;min-width:200px;">
# MAGIC <strong style="color:#FF5F46;">Limitations</strong>
# MAGIC <ul style="margin:6px 0;"><li>May not fully represent real user behavior</li><li>Static datasets become stale</li><li>Cannot capture scale/diversity issues</li></ul>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC Evaluation Datasets (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-datasets/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/eval-datasets" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/eval-datasets/" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Online Evaluation: Production Monitoring

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### C1. Online Evaluation Workflow
# MAGIC
# MAGIC Online evaluation analyzes agent performance using real user interactions: capturing actual usage patterns, unexpected inputs, and diverse user populations.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="background:#00A972;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:105px;">1. Deploy<br/>with Tracing<br/><span style="font-weight:400;font-size:13px;">Unity AI<br/>Gateway</span></div>
# MAGIC   <div style="color:#999;font-size:16px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:105px;">2. Capture<br/>Traces<br/><span style="font-weight:400;font-size:13px;">MLflow<br/>experiment</span></div>
# MAGIC   <div style="color:#999;font-size:16px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#2272B4;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:105px;">3. Automatic<br/>Scoring<br/><span style="font-weight:400;font-size:13px;">.register() +<br/>.start()</span></div>
# MAGIC   <div style="color:#999;font-size:16px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:105px;">4. Alerts &amp;<br/>Feedback<br/><span style="font-weight:400;font-size:13px;">Quality<br/>monitoring</span></div>
# MAGIC   <div style="color:#999;font-size:16px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:105px;">5. Augment<br/>Dataset<br/><span style="font-weight:400;font-size:13px;">Production<br/>examples</span></div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 12px 16px; border-radius: 4px; margin: 12px 0;">
# MAGIC <strong style="color: #0d47a1; font-size: 13px;">Supported scorer types:</strong>
# MAGIC <span style="font-size: 13px; color: #333;"> Built-in judges, <code>make_judge()</code>, <code>Guidelines</code>, and <code>@scorer</code> decorated functions all support production monitoring via <code>.register()</code> + <code>.start()</code>. Only direct <code>Scorer</code> subclasses and third-party scorers are excluded.</span>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #1B5162;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#1B5162;">Production monitoring capabilities</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC <div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0;">
# MAGIC <div style="flex:1;min-width:140px;background:#E8F5E9;border-radius:6px;padding:10px;text-align:center;"><strong>Auto Scoring</strong><br/><span style="font-size:13px;">LLM judges on production traces</span></div>
# MAGIC <div style="flex:1;min-width:140px;background:#E8F5E9;border-radius:6px;padding:10px;text-align:center;"><strong>Sampling</strong><br/><span style="font-size:13px;">Configurable sample rates</span></div>
# MAGIC <div style="flex:1;min-width:140px;background:#E8F5E9;border-radius:6px;padding:10px;text-align:center;"><strong>Alerting</strong><br/><span style="font-size:13px;">Quality threshold alerts</span></div>
# MAGIC <div style="flex:1;min-width:140px;background:#E8F5E9;border-radius:6px;padding:10px;text-align:center;"><strong>Trace Inspection</strong><br/><span style="font-size:13px;">Deep dive into failures</span></div>
# MAGIC </div>
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #1B5162;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#1B5162;">Strengths and limitations</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC <div style="display:flex;gap:16px;flex-wrap:wrap;">
# MAGIC <div style="flex:1;min-width:200px;">
# MAGIC <strong style="color:#00A972;">Strengths</strong>
# MAGIC <ul style="margin:6px 0;"><li>Reflects actual user experience</li><li>Detects unanticipated issues</li><li>Data for continuous improvement</li></ul>
# MAGIC </div>
# MAGIC <div style="flex:1;min-width:200px;">
# MAGIC <strong style="color:#FF5F46;">Limitations</strong>
# MAGIC <ul style="margin:6px 0;"><li>Users may hit failures first</li><li>Harder to attribute root causes</li><li>Requires production infrastructure</li></ul>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC Production Monitoring (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/" target="_blank">GCP</a>) &nbsp;|&nbsp; Inference Tables (<a href="https://docs.databricks.com/aws/en/ai-gateway/inference-tables/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/inference-tables" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/ai-gateway/inference-tables/" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Complementary Strategies

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D1. When to Use Each Approach
# MAGIC
# MAGIC Effective agent evaluation combines both. Click each to see when it applies:
# MAGIC
# MAGIC <div style="max-width:1000px;margin:0 auto;font-family:sans-serif;">
# MAGIC
# MAGIC <style>
# MAGIC .oo-step-row { display:flex; gap:12px; margin-bottom:0; flex-wrap:wrap; }
# MAGIC .oo-step-box {
# MAGIC   flex:1; min-width:280px; background:#F9F7F4; border-top:6px solid transparent;
# MAGIC   border-left:2px solid transparent; border-right:2px solid transparent; border-bottom:2px solid transparent;
# MAGIC   border-radius:8px; padding:14px 12px; text-align:center; cursor:pointer; user-select:none;
# MAGIC   transition:transform 0.12s;
# MAGIC }
# MAGIC .oo-step-box:hover { transform:translateY(-2px); }
# MAGIC .oo-step-box.active { background:#fff; border-left-color:var(--oc); border-right-color:var(--oc); border-bottom-color:var(--oc); }
# MAGIC .oo-step-title { display:block; font-size:16px; font-weight:700; color:#0b2026; line-height:1.3; pointer-events:none; }
# MAGIC .oo-step-sub { display:block; font-size:13px; color:#618794; margin-top:4px; pointer-events:none; }
# MAGIC .oo-detail-wrap { overflow:hidden; max-height:0; opacity:0; transition:max-height 0.35s ease, opacity 0.28s ease, margin-top 0.28s ease; margin-top:0; }
# MAGIC .oo-detail-wrap.open { max-height:800px; opacity:1; margin-top:12px; }
# MAGIC .oo-detail-card { background:#F9F7F4; border-radius:10px; padding:20px; border-top:6px solid #ccc; font-size:14px; line-height:1.6; }
# MAGIC </style>
# MAGIC
# MAGIC <div style="background:#F8F9FC;border:3px solid #1B5162;border-radius:10px;padding:20px;">
# MAGIC
# MAGIC <div class="oo-step-row">
# MAGIC   <span class="oo-step-box active" data-id="0" onclick="selectOoStep(0)" style="--oc:#2272B4;border-top-color:#2272B4;">
# MAGIC     <span class="oo-step-title">Offline Evaluation</span>
# MAGIC     <span class="oo-step-sub">Pre-deployment validation</span>
# MAGIC   </span>
# MAGIC   <span class="oo-step-box" data-id="1" onclick="selectOoStep(1)" style="--oc:#00A972;border-top-color:#00A972;">
# MAGIC     <span class="oo-step-title">Online Evaluation</span>
# MAGIC     <span class="oo-step-sub">Production monitoring</span>
# MAGIC   </span>
# MAGIC </div>
# MAGIC
# MAGIC <div class="oo-detail-wrap" id="oo-detail-wrap">
# MAGIC   <div class="oo-detail-card" id="oo-detail-card"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var OO_TYPES = [
# MAGIC   {
# MAGIC     title: "Offline Evaluation",
# MAGIC     color: "#2272B4",
# MAGIC     content: '<ul style="margin:8px 0;"><li><strong>Quality gates</strong> - Pre-deployment validation before users see the agent</li><li><strong>A/B testing</strong> - Compare alternative agent implementations side-by-side</li><li><strong>Hypothesis testing</strong> - Test specific behaviors (e.g., "does the new prompt reduce hallucinations?")</li><li><strong>Rapid iteration</strong> - Fast feedback during development cycles</li></ul>'
# MAGIC   },
# MAGIC   {
# MAGIC     title: "Online Evaluation",
# MAGIC     color: "#00A972",
# MAGIC     content: '<ul style="margin:8px 0;"><li><strong>Generalization check</strong> - Validate that offline results hold in production</li><li><strong>Drift detection</strong> - Catch performance degradation over time</li><li><strong>User insight</strong> - Understand real needs, pain points, and edge cases</li><li><strong>Dataset mining</strong> - Build evaluation datasets from actual usage patterns</li></ul>'
# MAGIC   }
# MAGIC ];
# MAGIC
# MAGIC var ooCurrent = null;
# MAGIC
# MAGIC function selectOoStep(id) {
# MAGIC   var wrap = document.getElementById('oo-detail-wrap');
# MAGIC   var card = document.getElementById('oo-detail-card');
# MAGIC   var o = OO_TYPES[id];
# MAGIC
# MAGIC   document.querySelectorAll('.oo-step-box').forEach(function(b) {
# MAGIC     b.classList.toggle('active', parseInt(b.dataset.id, 10) === id);
# MAGIC   });
# MAGIC
# MAGIC   if (ooCurrent === id) {
# MAGIC     wrap.classList.remove('open');
# MAGIC     document.querySelectorAll('.oo-step-box').forEach(function(b) { b.classList.remove('active'); });
# MAGIC     ooCurrent = null;
# MAGIC     return;
# MAGIC   }
# MAGIC
# MAGIC   ooCurrent = id;
# MAGIC   card.style.borderTopColor = o.color;
# MAGIC   card.innerHTML =
# MAGIC     '<div style="font-size:17px;font-weight:700;margin-bottom:10px;color:#0b2026;">' + o.title + '</div>'
# MAGIC     + o.content;
# MAGIC
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC
# MAGIC selectOoStep(0);
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### D2. Creating a Feedback Loop
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="background:#2272B4;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:130px;">1. Offline Eval<br/><span style="font-weight:400;font-size:13px;">Curated datasets</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:130px;">2. Deploy<br/><span style="font-weight:400;font-size:13px;">Monitoring enabled</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:130px;">3. Analyze Traces<br/><span style="font-weight:400;font-size:13px;">Failures &amp; edge cases</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:130px;">4. Augment Dataset<br/><span style="font-weight:400;font-size:13px;">Real examples</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#2272B4;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:130px;">5. Re-evaluate<br/><span style="font-weight:400;font-size:13px;">Validate &amp; redeploy</span></div>
# MAGIC </div>
# MAGIC
# MAGIC This cycle ensures your evaluation evolves with your understanding of real-world usage while maintaining rigorous pre-deployment validation.
# MAGIC
# MAGIC Human Feedback (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/human-feedback/" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/human-feedback/" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## E. Integration with Unity Catalog

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### E1. Governance for Evaluation
# MAGIC
# MAGIC Unity Catalog provides the governance layer for your evaluation workflow: traceability, access control, and lineage tracking across agents, datasets, and traces.
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:10px;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="flex:1;min-width:200px;background:#2272B4;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:15px;">Agent Registration</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Dependencies, access controls, aliases</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:200px;background:#00A972;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:15px;">Dataset Storage</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Unity Catalog, versioned, governed</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:200px;background:#FF5F46;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:15px;">Trace Storage</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">SQL-queryable, cross-team sharing</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:200px;background:#1B5162;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:15px;">Lineage</div>
# MAGIC     <div style="font-size:13px;margin-top:4px;opacity:0.9;">Datasets &#x2192; metrics &#x2192; versions</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin:10px 0;border:2px solid #1B5162;border-radius:10px;background:#fff;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:700;font-size:16px;color:#1B5162;">Details</summary>
# MAGIC <div style="padding:0 0 14px 0;font-size:14px;color:#333;">
# MAGIC <strong>Agent registration:</strong> All dependencies (vector indexes, UC functions, endpoints) are tracked. Aliases like <code>@champion</code> enable clean promotion workflows.<br/><br/>
# MAGIC <strong>Dataset storage:</strong> Version datasets alongside agent versions, apply access controls, enable SQL-based analysis and collaborative curation.<br/><br/>
# MAGIC <strong>Trace storage:</strong> Query traces using SQL, join with evaluation results, apply governance policies, and enable cross-team sharing.
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC MLflow Tracing (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/tracing/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/" target="_blank">GCP</a>) &nbsp;|&nbsp; Evaluation Datasets (<a href="https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-datasets/" target="_blank">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/eval-datasets" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/concepts/eval-datasets/" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### E2. Storing Traces in Unity Catalog Tables
# MAGIC
# MAGIC By default, MLflow stores traces in the MLflow control plane service. For production workloads, you can instead store OpenTelemetry traces directly in Unity Catalog Delta tables. This enables SQL-based analytics over traces, UC-level access control, and long-term retention.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="background:#2272B4;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:130px;">1. Your Agent<br/><span style="font-weight:400;font-size:13px;">Model Serving /<br/>Notebook</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:130px;">2. MLflow Tracing<br/><span style="font-weight:400;font-size:13px;">OpenTelemetry<br/>spans</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:130px;">3. Unity Catalog<br/><span style="font-weight:400;font-size:13px;">Delta tables<br/>(governed)</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:10px 14px;border-radius:8px;font-weight:600;font-size:13px;text-align:center;min-width:130px;">4. SQL Analytics<br/><span style="font-weight:400;font-size:13px;">Databricks SQL /<br/>Notebooks</span></div>
# MAGIC </div>
# MAGIC
# MAGIC Traces flow from your agent through MLflow into UC Delta tables, queryable via SQL.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Key benefits over default trace storage:</strong>
# MAGIC <ul style="margin: 8px 0 0 16px; color: #333;">
# MAGIC <li><strong>SQL queryable</strong>: Join traces with evaluation results using Databricks SQL</li>
# MAGIC <li><strong>UC access control</strong>: Govern trace access with catalog/schema permissions instead of experiment ACLs</li>
# MAGIC <li><strong>Long-term retention</strong>: Delta tables for high-volume production trace storage</li>
# MAGIC <li><strong>OTel compatible</strong>: Works with third-party OpenTelemetry clients via OTLP exporters</li>
# MAGIC </ul>
# MAGIC <p style="margin: 10px 0 0 0;">
# MAGIC For setup instructions, see <a href="https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog" style="color: #1976d2;">Store OpenTelemetry Traces in Unity Catalog</a> (<a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/trace-unity-catalog" style="color: #1976d2;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/trace-unity-catalog" style="color: #1976d2;">GCP</a>).
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to learn more about connecting offline and online evaluation? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-3-1" style="flex: 1;">How do I set up online evaluation with MLflow to monitor agent quality in production?</span>
# MAGIC     <button onclick="
# MAGIC       var text = document.getElementById('genie-3-1').innerText;
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

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC Offline and online evaluation are complementary strategies. Offline evaluation uses curated datasets and scorers to validate agent quality before deployment, while online evaluation monitors real production traffic using the same scorers via `.register()` + `.start()`. Together they form a feedback loop: production traces reveal failures and edge cases that augment your offline datasets, driving continuous improvement. Unity Catalog ties it all together with governed storage for agents, datasets, and traces.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC