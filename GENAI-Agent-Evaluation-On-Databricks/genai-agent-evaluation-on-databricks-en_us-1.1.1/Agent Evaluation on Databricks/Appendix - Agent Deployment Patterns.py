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
# MAGIC # Appendix - Agent Deployment Patterns
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This course uses `agents.deploy()` to deploy agents to Model Serving endpoints. While this approach works well for evaluation workflows, Databricks now recommends **Databricks Apps** as the deployment target for new agent projects.
# MAGIC
# MAGIC This appendix provides a brief overview of the deployment landscape so you can make informed decisions when deploying your own agents.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">This appendix is reference material only. The evaluation techniques covered in this course apply regardless of which deployment method you choose.</p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. Model Serving vs. Databricks Apps
# MAGIC
# MAGIC <div style="display:flex;gap:16px;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC
# MAGIC <div style="flex:1;min-width:280px;border:2px solid #2272B4;border-radius:10px;padding:18px;background:#fff;">
# MAGIC <div style="font-weight:700;font-size:16px;color:#2272B4;margin-bottom:10px;">Model Serving (agents.deploy)</div>
# MAGIC <ul style="margin:0;padding-left:18px;font-size:14px;color:#333;line-height:1.8;">
# MAGIC <li>Single function call to deploy</li>
# MAGIC <li>Automatic infrastructure provisioning</li>
# MAGIC <li>Built-in inference tables and tracing</li>
# MAGIC <li>Review App for stakeholder feedback</li>
# MAGIC <li>Agent code logged as MLflow artifact</li>
# MAGIC <li>Limited control over server configuration</li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC
# MAGIC <div style="flex:1;min-width:280px;border:2px solid #00A972;border-radius:10px;padding:18px;background:#fff;">
# MAGIC <div style="font-weight:700;font-size:16px;color:#00A972;margin-bottom:10px;">Databricks Apps (Recommended)</div>
# MAGIC <ul style="margin:0;padding-left:18px;font-size:14px;color:#333;line-height:1.8;">
# MAGIC <li>Git-based development with DABs</li>
# MAGIC <li>Full control over code and server config</li>
# MAGIC <li>Built-in Chat UI and <code>/invocations</code> endpoint</li>
# MAGIC <li>Local development with <code>uv run start-app</code></li>
# MAGIC <li>Async FastAPI server (MLflow AgentServer)</li>
# MAGIC <li>Explicit auth via <code>databricks.yml</code></li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC ### When to use each
# MAGIC
# MAGIC - **Model Serving** works well for quick prototyping, evaluation workflows (like this course), and existing deployments that don't need migration yet.
# MAGIC - **Databricks Apps** is the recommended path for new production deployments where you need Git-based CI/CD, local development, and full control over the serving infrastructure.
# MAGIC
# MAGIC Both approaches support MLflow tracing and evaluation. The `mlflow.genai.evaluate()` patterns you learned in this course work identically against agents deployed via either method.
# MAGIC
# MAGIC For full details, see: Author an AI agent and deploy it on Databricks Apps (<a href="https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent" target="_blank">AWS</a> | <a href="https://docs.databricks.com/azure/en/generative-ai/agent-framework/author-agent" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Migrating from Model Serving to Databricks Apps
# MAGIC
# MAGIC If you have an agent deployed via `agents.deploy()` and want to migrate to Databricks Apps, the process involves restructuring your code and configuration.
# MAGIC
# MAGIC ### Migration steps
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:nowrap;font-family:sans-serif;">
# MAGIC   <div style="background:#2272B4;color:#fff;padding:10px 14px;border-radius:8px 0 0 8px;font-weight:600;font-size:13px;text-align:center;min-width:120px;">1. Download<br/><span style="font-weight:400;font-size:12px;">Extract MLflow<br/>artifacts</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:10px 14px;font-weight:600;font-size:13px;text-align:center;min-width:120px;">2. Restructure<br/><span style="font-weight:400;font-size:12px;">Class methods to<br/>decorated functions</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:10px 14px;font-weight:600;font-size:13px;text-align:center;min-width:120px;">3. Configure<br/><span style="font-weight:400;font-size:12px;">Create<br/>databricks.yml</span></div>
# MAGIC   <div style="color:#999;font-size:18px;padding:0 4px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:10px 14px;border-radius:0 8px 8px 0;font-weight:600;font-size:13px;text-align:center;min-width:120px;">4. Deploy<br/><span style="font-weight:400;font-size:12px;">bundle deploy</span></div>
# MAGIC </div>
# MAGIC
# MAGIC ### What changes
# MAGIC
# MAGIC | Aspect | Model Serving | Databricks Apps |
# MAGIC |---|---|---|
# MAGIC | **Agent code** | Class-based `ResponsesAgent` with `predict()` / `predict_stream()` | Module-level `@invoke()` and `@stream()` decorated functions |
# MAGIC | **Configuration** | MLmodel file + model config YAML | `databricks.yml` with resource declarations |
# MAGIC | **Execution model** | Synchronous | Async recommended (`async def`) |
# MAGIC | **Deployment command** | `agents.deploy()` | `databricks bundle deploy` |
# MAGIC
# MAGIC ### What stays the same
# MAGIC
# MAGIC - Core agent logic (tool definitions, LLM calls, prompt templates)
# MAGIC - MLflow tracing and evaluation integration
# MAGIC - Unity Catalog tool registration
# MAGIC - Request/response payload format
# MAGIC
# MAGIC For the full migration guide, see: Migrate an agent from Model Serving to Databricks Apps (<a href="https://docs.databricks.com/aws/en/generative-ai/agent-framework/migrate-agent-to-apps" target="_blank">AWS</a> | <a href="https://docs.databricks.com/azure/en/generative-ai/agent-framework/migrate-agent-to-apps" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/generative-ai/agent-framework/migrate-agent-to-apps" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. The Agent Deployment Spectrum
# MAGIC
# MAGIC Databricks provides multiple levels of abstraction for building and deploying agents:
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:10px;margin:18px 0;flex-wrap:wrap;font-family:sans-serif;">
# MAGIC   <div style="flex:1;min-width:180px;background:#2272B4;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:700;font-size:15px;">Agent Bricks</div>
# MAGIC     <div style="font-size:12px;margin-top:6px;opacity:0.9;">Fully declarative.<br/>Knowledge Assistants,<br/>Supervisor Agents.<br/>No code required.</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:180px;background:#00A972;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:700;font-size:15px;">Supervisor API</div>
# MAGIC     <div style="font-size:12px;margin-top:6px;opacity:0.9;">Programmatic control.<br/>Databricks manages the<br/>agent loop. You define<br/>model, tools, instructions.</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:180px;background:#FF5F46;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:700;font-size:15px;">Databricks Apps</div>
# MAGIC     <div style="font-size:12px;margin-top:6px;opacity:0.9;">Full custom agents.<br/>Git-based, async,<br/>any framework.<br/>Recommended for production.</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:180px;background:#1B5162;color:#fff;padding:14px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:700;font-size:15px;">Unity Gateway APIs</div>
# MAGIC     <div style="font-size:12px;margin-top:6px;opacity:0.9;">Maximum control.<br/>Write your own agent loop.<br/>Databricks provides<br/>LLM inference only.</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="text-align:center;font-size:13px;color:#666;margin-top:4px;">← More managed &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; More control →</div>
# MAGIC
# MAGIC Regardless of where your agent sits on this spectrum, the evaluation techniques from this course apply. MLflow tracing, scorers, and `mlflow.genai.evaluate()` work across all deployment patterns.
# MAGIC
# MAGIC For more on the Supervisor API, see: Supervisor API (<a href="https://docs.databricks.com/aws/en/generative-ai/agent-bricks/supervisor-api" target="_blank">AWS</a> | <a href="https://docs.databricks.com/azure/en/generative-ai/agent-bricks/supervisor-api" target="_blank">Azure</a> | <a href="https://docs.databricks.com/gcp/en/generative-ai/agent-bricks/supervisor-api" target="_blank">GCP</a>)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC
# MAGIC