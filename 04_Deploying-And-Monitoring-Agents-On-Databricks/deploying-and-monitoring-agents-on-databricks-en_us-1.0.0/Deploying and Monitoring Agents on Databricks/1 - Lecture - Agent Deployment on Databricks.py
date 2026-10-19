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

# MAGIC %md
# MAGIC # Lecture - Agent Deployment on Databricks
# MAGIC
# MAGIC ## Overview
# MAGIC This lecture covers the architecture and tooling required to move a generative AI agent from a notebook prototype into a production Databricks App. The focus is on the agent lifecycle, Databricks Apps as the deployment target, and Declarative Automation Bundles (DABs) as the packaging and deployment mechanism.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lecture you will be able to:
# MAGIC - Outline the four stages of the agent lifecycle on Databricks
# MAGIC - Differentiate Databricks Apps deployment from model serving endpoint deployment
# MAGIC - Explain how Declarative Automation Bundles structure and deploy an agent application
# MAGIC - Describe how resource injection decouples agent code from infrastructure configuration
# MAGIC - Identify the role of Unity AI Gateway as the governance layer for deployed agents

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. The Agent Lifecycle

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. From Prototype to Production
# MAGIC
# MAGIC Building an agent in a notebook is only the beginning. A production-grade agent must be **deployed** to infrastructure that can serve requests, **observed** so you can understand its behavior, **evaluated** to measure quality, and **monitored** continuously to catch regressions before users notice them.
# MAGIC
# MAGIC On Databricks, this lifecycle maps to four stages:
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:120px;">Deployment<br/><span style="font-weight:400;font-size:14px;">DABs + Apps</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#2574B5;color:#fff;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:120px;">Observability<br/><span style="font-weight:400;font-size:14px;">MLflow Tracing</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#02A36F;color:#fff;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:120px;">Evaluation<br/><span style="font-weight:400;font-size:14px;">Scorers + Judges</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:10px 16px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:120px;">Monitoring<br/><span style="font-weight:400;font-size:14px;">Backfill + Archival</span></div>
# MAGIC </div>
# MAGIC
# MAGIC - **Deployment**: Package the agent and deploy it as a Databricks App using Declarative Automation Bundles (DABs). The app runs as a managed web application with its own compute, service principal, and URL.
# MAGIC - **Observability**: Integrate MLflow Tracing so every request produces a structured trace capturing inputs, outputs, latency, and execution flow.
# MAGIC - **Evaluation**: Attach scorers to the MLflow Experiment that automatically assess trace quality using LLM judges, guidelines, or custom Python functions.
# MAGIC - **Monitoring**: Manage scorer lifecycles, backfill historical traces with new scorers, and archive traces to Delta tables for long-term analysis.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">These stages are not strictly sequential. You can add tracing before deployment, attach scorers to an agent that is already live, or backfill evaluation data onto traces logged weeks ago. The lifecycle is designed to be incremental.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Databricks Apps key concepts ([AWS](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/key-concepts) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/key-concepts) | [GCP](https://docs.databricks.com/gcp/en/dev-tools/databricks-apps/key-concepts))

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Deployment Architecture

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Databricks Apps
# MAGIC
# MAGIC **Databricks Apps** provide a managed hosting environment for web applications running on the Databricks platform. When you deploy an agent as a Databricks App, the platform handles the infrastructure concerns so you can focus on the application logic.
# MAGIC
# MAGIC Key characteristics of a Databricks App:
# MAGIC - **Managed compute**: The platform provisions and manages the compute resources. You do not configure clusters, VMs, or containers.
# MAGIC - **Service principal**: Each app runs under its own service principal identity, which determines what Databricks resources the app can access (serving endpoints, UC objects, experiments).
# MAGIC - **Unique URL**: Every deployed app gets a workspace-scoped URL that users or downstream services can call directly.
# MAGIC - **Workspace integration**: Apps are first-class workspace objects visible on the **Apps** page in the workspace sidebar, where you can monitor status, view logs, and trigger redeployments.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">In the app-based deployment pattern, the agent code itself is the deployment artifact. The app calls a model serving endpoint for LLM inference, but the orchestration logic, tool integrations, and observability wiring all live in your application source.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Declarative Automation Bundles
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC A **Declarative Automation Bundle (DAB)** is the project structure that describes your app and its resources. The project folder contains two categories of files:
# MAGIC
# MAGIC **User-authored source files** (you write these):
# MAGIC - `agent.py`: The agent logic (using the OpenAI Agents SDK pointed at a Databricks AI Model Serving endpoint)
# MAGIC - `server.py`: A FastAPI server that wraps the agent and exposes HTTP endpoints
# MAGIC - `pyproject.toml`: Python dependency specification
# MAGIC
# MAGIC **Configuration files** (defined by the DAB/Apps spec):
# MAGIC - `databricks.yml`: Bundle declaration: app name, resources (serving endpoints, MLflow experiments), and permissions
# MAGIC - `app.yaml`: Runtime manifest: the startup command and environment variable injection via its top-level `env:` section
# MAGIC
# MAGIC Deployment is a **two-step process**. First, run `databricks bundle deploy` from the CLI to create or update the app resource and its configuration in the workspace. Then, deploy the application source code by running `databricks apps deploy <app-name>` from the CLI or through the **Apps** page in the workspace sidebar. The bundle deploy step manages resource declarations; the apps deploy step pushes your source code to the running infrastructure.
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Two-step CLI deployment</summary>
# MAGIC <div class="cb"><pre><span class="cm"># Step 1: Deploy the bundle (creates/updates resources)</span>
# MAGIC $ databricks bundle deploy
# MAGIC
# MAGIC <span class="cm"># Step 2: Deploy the app source code</span>
# MAGIC $ databricks apps deploy my-agent-app \
# MAGIC     --source-code-path /Workspace/Users/&lt;your-username&gt;/my-agent-app</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">The two steps are independent. You can redeploy app source code without changing resource declarations, and you can update resource configuration without redeploying code. This separation makes iterative development faster.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC DABs app deployment tutorial ([AWS](https://docs.databricks.com/aws/en/dev-tools/bundles/apps-tutorial) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/bundles/apps-tutorial) | [GCP](https://docs.databricks.com/gcp/en/dev-tools/bundles/apps-tutorial))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B3. Comparing Deployment Patterns
# MAGIC
# MAGIC Traditionally, deploying an agent on Databricks meant registering a model with MLflow and Unity Catalog, then deploying it to a **model serving endpoint**. With **Databricks Apps**, the deployment model is fundamentally different. The agent code itself is the deployment artifact, not a registered model.
# MAGIC
# MAGIC <div style="display:flex;justify-content:center;gap:16px;margin:18px 0;font-family:system-ui,sans-serif;flex-wrap:wrap;">
# MAGIC   <div style="flex:1;min-width:200px;border:2px solid #999;border-radius:10px;padding:14px;text-align:center;">
# MAGIC     <div style="font-size:14px;font-weight:600;color:#999;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:10px;">Model Serving Pattern</div>
# MAGIC     <div style="background:#F9F7F4;border:1px solid #1B5162;border-radius:6px;padding:8px;font-size:14px;font-weight:600;margin-bottom:6px;">Registered Model (UC)</div>
# MAGIC     <div style="color:#999;font-size:16px;">&#x2193;</div>
# MAGIC     <div style="background:#1B5162;color:#fff;border-radius:6px;padding:8px;font-size:14px;font-weight:600;">Model Serving Endpoint</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:200px;border:2px solid #FF5F46;border-radius:10px;padding:14px;text-align:center;">
# MAGIC     <div style="font-size:14px;font-weight:600;color:#FF5F46;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:10px;">App-Based Pattern</div>
# MAGIC     <div style="background:#FF5F46;color:#fff;border-radius:6px;padding:8px;font-size:14px;font-weight:600;margin-bottom:6px;">Databricks App (DAB)</div>
# MAGIC     <div style="color:#999;font-size:16px;">&#x2193; calls</div>
# MAGIC     <div style="background:#F9F7F4;border:1px solid #1B5162;border-radius:6px;padding:8px;font-size:14px;font-weight:600;">Model Serving Endpoint</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC | Concern | Model Serving Endpoint | Databricks App |
# MAGIC |---|---|---|
# MAGIC | **What is deployed** | A registered model (MLflow/UC) | Application code (DAB) |
# MAGIC | **Inference** | The endpoint runs the model directly | The app calls a serving endpoint for inference |
# MAGIC | **Scaling** | Managed by the serving infrastructure | Managed by the Apps platform |
# MAGIC | **Customization** | Limited to model configuration | Full control: custom UI, middleware, routing logic |
# MAGIC | **Governance** | UC model permissions | App-level ACLs + UC permissions on accessed resources |
# MAGIC | **Tool integration** | Defined in the registered model | MCP servers, direct API calls, any Python library |
# MAGIC
# MAGIC In the app-based pattern, the serving endpoint is still present. It hosts the foundation model that the agent calls for inference. But the agent orchestration, tool integrations, and observability wiring all live in the app code. This separation gives you full control over the application layer while Databricks manages model hosting.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Registering the agent model in Unity Catalog is optional with app-based deployment. UC still governs the data and resources the app touches (serving endpoints, catalog objects), but the app itself is not a UC securable. Governance splits between workspace-level app ACLs and UC permissions on downstream resources.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Agent framework and deployment patterns ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/deploying-and-monitoring-agents-on-databricks-en_us-1.0.0/images/20261010T172734Z/Deploying and Monitoring Agents on Databricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to learn more about <strong>agent deployment patterns</strong>? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/deploying-and-monitoring-agents-on-databricks-en_us-1.0.0/images/20261010T172734Z/Deploying and Monitoring Agents on Databricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-query" style="flex: 1;">What are the differences between deploying an agent as a Databricks App versus a model serving endpoint?</span>
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

# MAGIC %md-sandbox
# MAGIC ### B4. Resource Injection
# MAGIC
# MAGIC <style>
# MAGIC .cb{margin-top:12px;border-radius:8px;overflow:hidden;border:0.5px solid #dde0e8}
# MAGIC .cb pre{background:#f7f8fc;padding:12px 14px;font-family:monospace;font-size:16px;line-height:1.7;overflow-x:auto;color:#0b2026}
# MAGIC .kw{color:#a855f7}.fn{color:#3b82f6}.st{color:#16a34a}.nm{color:#ea580c}.cm{color:#9ca3af}.dc{color:#d97706}
# MAGIC </style>
# MAGIC
# MAGIC The `databricks.yml` bundle file declares every resource the app depends on. The `app.yaml` file controls how those resources are injected into the running app as **environment variables** via its top-level `env:` section. This means agent code reads configuration from the environment rather than hardcoding workspace-specific values.
# MAGIC
# MAGIC Here is how the two files work together:
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="background:#F9F7F4;border:2px solid #1B5162;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;"><strong>databricks.yml</strong><br/><span style="font-size:14px;color:#555;">Declares resources</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#F9F7F4;border:2px solid #1B5162;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;"><strong>app.yaml</strong><br/><span style="font-size:14px;color:#555;">Injects as env vars</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;border-radius:8px;padding:10px 14px;text-align:center;font-size:14px;font-weight:600;">Agent Code<br/><span style="font-weight:400;font-size:14px;">os.environ[...]</span></div>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: databricks.yml resource declaration</summary>
# MAGIC <div class="cb"><pre><span class="cm"># databricks.yml -- declares the app and its resources</span>
# MAGIC bundle:
# MAGIC   name: my-agent-app
# MAGIC
# MAGIC resources:
# MAGIC   apps:
# MAGIC     my_agent:
# MAGIC       name: my-agent-app
# MAGIC       source_code_path: ./src
# MAGIC       config:
# MAGIC         command: [<span class="st">"uv"</span>, <span class="st">"run"</span>, <span class="st">"start-server"</span>]
# MAGIC         env:
# MAGIC           - name: <span class="st">DATABRICKS_HOST</span>
# MAGIC             value: <span class="st">${workspace.host}</span>
# MAGIC           - name: <span class="st">SERVING_ENDPOINT</span>
# MAGIC             value: <span class="st">my-llm-endpoint</span>
# MAGIC           - name: <span class="st">MLFLOW_EXPERIMENT_NAME</span>
# MAGIC             value: <span class="st">/Users/me/my-experiment</span></pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: app.yaml environment variable injection</summary>
# MAGIC <div class="cb"><pre><span class="cm"># app.yaml -- runtime manifest with env var injection</span>
# MAGIC <span class="cm"># valueFrom references a resource key defined in the app's resources block</span>
# MAGIC command: [<span class="st">"uv"</span>, <span class="st">"run"</span>, <span class="st">"start-server"</span>]
# MAGIC env:
# MAGIC   - name: <span class="st">SERVING_ENDPOINT</span>
# MAGIC     valueFrom: <span class="st">"my-serving-endpoint"</span>
# MAGIC   - name: <span class="st">SECRET_KEY</span>
# MAGIC     valueFrom: <span class="st">"my-secret"</span>
# MAGIC   - name: <span class="st">MLFLOW_EXPERIMENT_NAME</span>
# MAGIC     value: <span class="st">"/Users/me/my-experiment"</span></pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <details style="margin-top:12px;">
# MAGIC <summary style="cursor:pointer;font-weight:600;font-size:15px;color:#378ADD;">Example: Reading environment variables in agent code</summary>
# MAGIC <div class="cb"><pre><span class="cm"># Reading injected resources in agent code</span>
# MAGIC <span class="kw">import</span> os
# MAGIC
# MAGIC serving_endpoint = os.environ[<span class="st">"SERVING_ENDPOINT"</span>]
# MAGIC experiment_name = os.environ[<span class="st">"MLFLOW_EXPERIMENT_NAME"</span>]
# MAGIC workspace_host = os.environ[<span class="st">"DATABRICKS_HOST"</span>]</pre></div>
# MAGIC </details>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Resource injection decouples agent logic from infrastructure configuration. The same agent code can run in development, staging, and production environments by changing only the <code>databricks.yml</code> resource declarations. In <code>databricks.yml</code>, environment variables are nested under <code>config.env</code>. In <code>app.yaml</code>, they use a top-level <code>env:</code> section. The <code>valueFrom</code> field in <code>app.yaml</code> references resource keys defined in the app's <code>resources</code> block. Note that <code>DATABRICKS_HOST</code> is automatically available in all Databricks apps, so it only needs explicit injection when using bundle substitution in <code>databricks.yml</code>.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC App environment variables and resource injection ([AWS](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/environment-variables) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/environment-variables) | [GCP](https://docs.databricks.com/gcp/en/dev-tools/databricks-apps/environment-variables))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B5. Unity AI Gateway
# MAGIC
# MAGIC <div style="border-left: 4px solid #f57c00; background: #fff3e0; padding: 12px 16px; border-radius: 4px; margin: 0 0 16px 0;">
# MAGIC <strong style="color: #e65100;">Beta</strong> — The unified AI Gateway (with traffic splitting, coding agent governance, and MCP server support) is currently in Beta. Account admins can enable access from the Previews page. APIs and behavior may change.
# MAGIC </div>
# MAGIC
# MAGIC When a deployed agent calls a model serving endpoint for inference, that request passes through the **Unity AI Gateway** — the centralized AI governance layer on Databricks. The gateway sits between your application and the serving endpoints, providing guardrails, rate limiting, usage tracking, and traffic management without requiring changes to your agent code.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:24px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:14px 18px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:130px;">Databricks App<br/><span style="font-weight:400;font-size:14px;">Agent Code</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="border:2px solid #02A36F;border-radius:10px;padding:16px 20px;background:#fff;text-align:center;min-width:200px;">
# MAGIC     <div style="font-size:14px;font-weight:600;color:#02A36F;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:10px;">Unity AI Gateway</div>
# MAGIC     <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
# MAGIC       <div style="background:#E8F5E9;color:#2E7D32;padding:4px 10px;border-radius:4px;font-size:13px;font-weight:600;">Guardrails</div>
# MAGIC       <div style="background:#E8F5E9;color:#2E7D32;padding:4px 10px;border-radius:4px;font-size:13px;font-weight:600;">Rate Limits</div>
# MAGIC       <div style="background:#E8F5E9;color:#2E7D32;padding:4px 10px;border-radius:4px;font-size:13px;font-weight:600;">Usage Tracking</div>
# MAGIC       <div style="background:#E8F5E9;color:#2E7D32;padding:4px 10px;border-radius:4px;font-size:13px;font-weight:600;">Traffic Splitting</div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:14px 18px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:130px;">Serving Endpoint<br/><span style="font-weight:400;font-size:14px;">Foundation Model</span></div>
# MAGIC </div>
# MAGIC
# MAGIC Key capabilities of the AI Gateway:
# MAGIC
# MAGIC | Capability | Description |
# MAGIC |---|---|
# MAGIC | **Guardrails** | Enforce safety and content policies on requests and responses at the gateway level |
# MAGIC | **Rate limiting** | Configure consumption limits per endpoint to manage capacity and control costs |
# MAGIC | **Usage tracking** | Analyze costs by endpoint, model, principal, and tags via billable usage system tables |
# MAGIC | **Traffic splitting** | Distribute requests across multiple model backends for A/B testing or gradual rollouts |
# MAGIC | **Inference tables** | Log requests and responses to Unity Catalog Delta tables for auditing and analysis |
# MAGIC
# MAGIC The gateway governs more than just agent traffic. It provides a unified governance layer across LLM endpoints, MCP servers, and coding agents (such as Cursor, Claude Code, and Codex CLI), giving platform teams a single place to manage AI access across the organization.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">The AI Gateway is transparent to your agent code. Your app calls the serving endpoint as usual; the gateway intercepts and governs the request automatically. No code changes are needed to benefit from rate limiting, guardrails, or usage tracking — these are configured at the platform level.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC Unity AI Gateway overview ([AWS](https://docs.databricks.com/aws/en/ai-gateway/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/) | [GCP](https://docs.databricks.com/gcp/en/ai-gateway/))

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC