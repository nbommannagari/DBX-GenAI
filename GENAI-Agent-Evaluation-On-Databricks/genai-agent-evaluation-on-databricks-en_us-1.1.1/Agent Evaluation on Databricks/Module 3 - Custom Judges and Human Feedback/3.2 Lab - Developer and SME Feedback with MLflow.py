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
# MAGIC # Lab - Developer and SME Feedback with MLflow
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This lab provides hands-on experience with MLflow's human feedback capabilities, focusing on collecting and managing assessments from different reviewer personas. You'll learn to implement feedback workflows that support development iteration, domain expert evaluation, and end-user input collection. The lab covers MLflow Assessments, which allow you to attach structured feedback, scores, and ground truth to traces and spans for quality evaluation and improvement.
# MAGIC
# MAGIC Human feedback is critical for improving AI applications, as it provides qualitative insights that complement automated metrics. This lab demonstrates how to systematically collect, organize, and analyze feedback from multiple stakeholder types to drive continuous improvement of your AI systems.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC - Implement developer feedback workflows using MLflow Assessments
# MAGIC - Configure SME (Subject Matter Expert) feedback collection through the Chat UI
# MAGIC - Distinguish between feedback and expectation assessment types
# MAGIC - Navigate MLflow's tracing interface to add and review assessments
# MAGIC - Review SME feedback programmatically in a Databricks Notebook
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Prerequisites</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"> This lab uses the agent created in <strong>Demo - Agent Setup</strong>. Please ensure you have completed that notebook before proceeding.</p>
# MAGIC </div>
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
# MAGIC ## A. Classroom Setup
# MAGIC Run the following cell to configure your environment.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-6

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Developer Feedback Implementation
# MAGIC
# MAGIC First, you will explore how developers can provide feedback during the development phase. MLflow Tracing allows you to add feedback or expectations directly to traces during development, giving you a quick way to record quality issues, mark successful examples, or add notes for future reference.
# MAGIC
# MAGIC The classroom setup script has created traces for you to evaluate. During setup, the agent was queried with 3 sample questions that are now available for assessment.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Navigate to Feedback Experiment
# MAGIC
# MAGIC Follow these steps to access the feedback experiment and provide developer assessments:
# MAGIC
# MAGIC 1. Navigate to and click on **Experiments** on the left side menu
# MAGIC 2. Locate the experiment called `airbnb_experiment` and click on it
# MAGIC     - You might be met with a message to choose the **experiment type**. Choose **GenAI apps & agents**. 
# MAGIC 3. We have created a session called **feedback-session-001**: click on **Traces** to see this session (Check under **Session** Column), then click on it.
# MAGIC 4. On the left, you will see **Turn 1** through **Turn 3** (see screenshot below)
# MAGIC
# MAGIC ![mlflow-assessment3.png](https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/feedback evaluation/mlflow-assessment3.png "mlflow-assessment3.png")
# MAGIC
# MAGIC Here we can add assessments at both the session-level and trace-level.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Viewing Trace-Level Assessment
# MAGIC
# MAGIC Now we'll add a trace-level assessment to evaluate tool usage:
# MAGIC
# MAGIC 1. Click on **Turn 1** and select **View full trace** or **Evaluate trace**
# MAGIC 2. You will see **Assessment** on the right of the trace. Click on it.
# MAGIC
# MAGIC ![mlflow-assessment1.png](https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/feedback evaluation/mlflow-assessment1.png "mlflow-assessment1.png")
# MAGIC
# MAGIC 3. Click on **Add Feedback** or **Add Expectation** on the right of the screen
# MAGIC 4. This will create a submenu with options to select **Feedback** or **Expectations** from **Assessment Type**, enter the **Assessment Name**, and select **String**, **Boolean**, or **Number** from **Data Type**.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B3. Configure Tool Usage Assessment
# MAGIC
# MAGIC Based on the lab setup, you will see that a tool was used and it was indeed the correct one. Fill out the assessment with the following values:
# MAGIC
# MAGIC - **Assessment Type**: _Expectation_
# MAGIC - **Assessment Name**:
# MAGIC <div style="display:inline-flex;align-items:center;gap:8px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:6px 12px;font-family:monospace;font-size:14pt;color:#0b2026;margin:4px 0;"><span id="b3-name">tool_usage</span><button onclick="var t=document.createElement('textarea');t.value=document.getElementById('b3-name').innerText;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.innerText='Copied!';var b=this;setTimeout(function(){b.innerText='Copy'},1500)" style="background:#FF5F46;color:#fff;border:none;border-radius:4px;padding:2px 10px;font-size:12pt;cursor:pointer;">Copy</button></div>
# MAGIC
# MAGIC - **Data Type**: _Boolean_
# MAGIC - **Value**: _True_
# MAGIC - **Rationale**:
# MAGIC <div style="display:inline-flex;align-items:center;gap:8px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:6px 12px;font-family:monospace;font-size:14pt;color:#0b2026;margin:4px 0;"><span id="b3-rationale">The tool was used correctly.</span><button onclick="var t=document.createElement('textarea');t.value=document.getElementById('b3-rationale').innerText;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.innerText='Copied!';var b=this;setTimeout(function(){b.innerText='Copy'},1500)" style="background:#FF5F46;color:#fff;border:none;border-radius:4px;padding:2px 10px;font-size:12pt;cursor:pointer;">Copy</button></div>
# MAGIC
# MAGIC Click **Create** and close the trace.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B4. Add Session-Level Assessment
# MAGIC
# MAGIC Now let's add a session-level assessment:
# MAGIC
# MAGIC 1. You can evaluate the other 2 turns using the same process as Turn 1
# MAGIC 2. Return to the main session menu and click on **Add Feedback** under **Session assessments** > **Human Feedback**
# MAGIC 3. Fill out the following values:
# MAGIC    - **Assessment Type**: Feedback
# MAGIC    - **Assessment Name**:
# MAGIC <div style="display:inline-flex;align-items:center;gap:8px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:6px 12px;font-family:monospace;font-size:14pt;color:#0b2026;margin:4px 0;"><span id="b4-name">ready_for_sme_feedback</span><button onclick="var t=document.createElement('textarea');t.value=document.getElementById('b4-name').innerText;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.innerText='Copied!';var b=this;setTimeout(function(){b.innerText='Copy'},1500)" style="background:#FF5F46;color:#fff;border:none;border-radius:4px;padding:2px 10px;font-size:12pt;cursor:pointer;">Copy</button></div>
# MAGIC
# MAGIC
# MAGIC    - **Data Type**: _String_
# MAGIC    - **Value**: _True_
# MAGIC    - **Rationale**:
# MAGIC <div style="display:inline-flex;align-items:center;gap:8px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:6px 12px;font-family:monospace;font-size:14pt;color:#0b2026;margin:4px 0;"><span id="b4-rationale">Based on the review of the outputs from each turn, the results are correct and testing is complete and ready for SME review.</span><button onclick="var t=document.createElement('textarea');t.value=document.getElementById('b4-rationale').innerText;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.innerText='Copied!';var b=this;setTimeout(function(){b.innerText='Copy'},1500)" style="background:#FF5F46;color:#fff;border:none;border-radius:4px;padding:2px 10px;font-size:12pt;cursor:pointer;">Copy</button></div>
# MAGIC
# MAGIC 4. Click **Create**
# MAGIC
# MAGIC **NOTE:** You are now ready to send to your SMEs for review, which we will cover in the next section.

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. SME Feedback Implementation
# MAGIC
# MAGIC Next, we'll implement SME (Subject Matter Expert) feedback. Suppose you are now acting as a SME for review. There are two approaches for SME feedback collection:
# MAGIC
# MAGIC 1. **Interactive testing with the Chat UI**
# MAGIC 2. **Labeling existing traces**
# MAGIC
# MAGIC In this lab, you will use the Chat UI approach.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. Deploy Agent Using Agent Framework
# MAGIC
# MAGIC Next, we need a deployed model to a model serving endpoint. We will use the `databricks` SDK class `agent.deploy()` method. All we need is the `model_name` and `uc_model_info`, both of which were created as part of the classroom setup.
# MAGIC
# MAGIC **NOTE:** In the interest of time, this has been performed for you as part of running **Demo - Agent Setup**.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Deployment Recommendation</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">This course uses <code>agents.deploy()</code> to deploy to Model Serving endpoints. For new production use cases, Databricks recommends deploying agents as <strong>Databricks Apps</strong>. See the <strong>Appendix - Agent Deployment Patterns</strong> for details on migration and the deployment spectrum.</p>
# MAGIC </div>
# MAGIC
# MAGIC Here is a code snippet showing how to deploy the agent:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.tracking import MlflowClient
# MAGIC from databricks import agents
# MAGIC
# MAGIC model_name = f"{catalog_name}.{schema_name}.{agent_name}"  # UC FQN
# MAGIC alias = "Champion"
# MAGIC
# MAGIC client = MlflowClient()
# MAGIC mv = client.get_model_version_by_alias(model_name, alias)
# MAGIC
# MAGIC # Pass the version, not the ModelVersion object
# MAGIC deployment = agents.deploy(
# MAGIC     model_name=model_name,
# MAGIC     model_version=int(mv.version),  # str also works; cast to int for clarity
# MAGIC     scale_to_zero=True
# MAGIC )
# MAGIC print("Endpoint:", deployment.endpoint_name)
# MAGIC ```
# MAGIC
# MAGIC <div style="
# MAGIC   border-left: 4px solid #ff9800;
# MAGIC   background: #fff3e0;
# MAGIC   padding: 14px 18px;
# MAGIC   border-radius: 4px;
# MAGIC   margin: 16px 0;
# MAGIC ">
# MAGIC   <strong style="display:block; color:#e65100; margin-bottom:6px; font-size: 1.1em;">
# MAGIC     Warning
# MAGIC   </strong>
# MAGIC   <div style="color:#333;">
# MAGIC <code>scale_to_zero=True</code> means that your agent endpoint will have zero replicas when idle. This means that the first query after a period of inactivity may experience cold-start latency.
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Interactive Testing with Chat UI
# MAGIC
# MAGIC We will use the [`get_review_app()`](https://mlflow.org/docs/latest/api_reference/_modules/mlflow/genai/labeling.html#get_review_app) method to get or create the review app.
# MAGIC
# MAGIC **Warning:** **Domain experts** need the following permissions to use the Review App's Chat UI:
# MAGIC
# MAGIC - **Account access**: Must be provisioned in your Databricks account, but does *not* require workspace access
# MAGIC - **Endpoint access**: `CAN_QUERY` permission on the model serving endpoint
# MAGIC - **MLflow access**: Workspace access with `CAN_EDIT` permissions on the MLflow Experiment
# MAGIC
# MAGIC For users without workspace access, account admins can:
# MAGIC - Use account-level SCIM provisioning to sync users from your identity provider
# MAGIC - Manually register users and groups in Databricks
# MAGIC
# MAGIC See [User and group management](https://docs.databricks.com/aws/en/admin/users-groups/scim/) for details.

# COMMAND ----------

from mlflow.genai.labeling import get_review_app

review_app = get_review_app()

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Configure Review App
# MAGIC
# MAGIC Let's connect our review app to our agent. The output from the next cell will contain the Review App URL. If you click on it, you will be able to:
# MAGIC
# MAGIC - Access the chat interface through their web browser
# MAGIC - Interact with your application by typing questions
# MAGIC - Provide feedback after each response using the built-in feedback controls
# MAGIC - Continue the conversation to test multiple interactions
# MAGIC
# MAGIC **Alternative UI approach:** You can also navigate to **Serving** on the left side menu, click on your deployed agent, and select the dropdown menu next to **Use** and select **Open review app**.

# COMMAND ----------

# MAGIC %md
# MAGIC In the next code snippet, we are using `agent_name` and `model_username_string` string variables that were created as a part of our lab setup.

# COMMAND ----------

review_app.add_agent(
    agent_name = agent_name,
    model_serving_endpoint = model_username_string
)

print(f"Share this URL: {review_app.url}/chat")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C4. Test Query and Provide Feedback
# MAGIC
# MAGIC Next, let's pass a query and provide feedback. For your convenience, a **Copy to clipboard** button has been added - click it to copy the query. Paste it in the review app. Below is a screenshot of what your screen should look like:
# MAGIC
# MAGIC ![feedback-app.png](https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/feedback evaluation/feedback-app.png "feedback-app.png")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Paste the following query into the review app chat:
# MAGIC
# MAGIC <div style="display:flex;align-items:center;gap:10px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:10px 14px;font-family:monospace;font-size:14pt;color:#0b2026;margin:8px 0;">
# MAGIC <span id="c4-query" style="flex:1;">How many rooms in Mission have a private room?</span>
# MAGIC <button onclick="var t=document.createElement('textarea');t.value=document.getElementById('c4-query').innerText;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.innerText='Copied!';var b=this;setTimeout(function(){b.innerText='Copy'},1500)" style="background:#FF5F46;color:#fff;border:none;border-radius:4px;padding:4px 12px;font-size:13pt;cursor:pointer;white-space:nowrap;">Copy</button>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C5. Submit Feedback
# MAGIC
# MAGIC Let's add some feedback. The result looks good and you should clearly see that a tool was used. Select **Yes** in the **Awaiting Feedback** window as shown in the previous screenshot, then paste the following feedback and click **Done**.
# MAGIC
# MAGIC ![feedback-app2.png](https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/feedback evaluation/feedback-app2.png "feedback-app2.png")
# MAGIC
# MAGIC <div style="display:flex;align-items:center;gap:10px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:10px 14px;font-family:monospace;font-size:14pt;color:#0b2026;margin:8px 0;">
# MAGIC <span id="c5-feedback" style="flex:1;">The agent should end with "Is there anything else I can help with?"</span>
# MAGIC <button onclick="var t=document.createElement('textarea');t.value=document.getElementById('c5-feedback').innerText;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.innerText='Copied!';var b=this;setTimeout(function(){b.innerText='Copy'},1500)" style="background:#FF5F46;color:#fff;border:none;border-radius:4px;padding:4px 12px;font-size:13pt;cursor:pointer;white-space:nowrap;">Copy</button>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### C6. View Submitted Feedback
# MAGIC
# MAGIC Navigate back to the model serving endpoint in **Serving** and click on **Traces** and select the latest trace (you might need to click the reload icon if you don't see your trace immediately). In the trace screen, you will see the familiar output, but you will also see your feedback submitted (see screenshot below).
# MAGIC
# MAGIC ![feedback-app3.png](https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/feedback evaluation/feedback-app3.png "feedback-app3.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C7. Viewing SME Feedback in a Notebook
# MAGIC
# MAGIC The MLflow trace is available using the UI, but we can also view the assessment from the SME programmatically. The code below retrieves traces from `airbnb_experiment` using `search_traces()`. After running the next cell, scroll and expand the assessment for the record returned.

# COMMAND ----------

import pandas

mlflow.set_tracking_uri("databricks")

## Resolve experiment ID
exp = mlflow.get_experiment_by_name(experiment_name)
exp_id = exp.experiment_id
print("Experiment ID:", exp_id)

## Query traces using string experiment ID; limit to top-level fields for safe display
traces_df = mlflow.search_traces(
    locations=[str(exp_id)],            ## must be str
    include_spans=True,                ## include full span trees for assessment details
    return_type="pandas"
)
traces_df.head(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Conclusion
# MAGIC
# MAGIC In this lab, you have successfully implemented human feedback workflows using MLflow across different reviewer personas:
# MAGIC
# MAGIC - **Developer feedback** through direct trace annotation in the MLflow UI
# MAGIC - **SME feedback** using the interactive Chat UI for real-time evaluation
# MAGIC
# MAGIC You've learned to distinguish between feedback and expectation assessment types, navigate MLflow's tracing interface, and understand the permissions and access patterns required for different reviewer types. These skills are essential for building robust AI applications that continuously improve through structured human feedback collection.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC