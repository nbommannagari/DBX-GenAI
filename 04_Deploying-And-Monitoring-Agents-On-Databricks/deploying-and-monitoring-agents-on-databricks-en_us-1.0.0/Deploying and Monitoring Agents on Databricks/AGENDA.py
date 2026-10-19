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
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Deploying and Monitoring Agents on Databricks
# MAGIC
# MAGIC This course covers the end-to-end lifecycle for deploying and monitoring generative AI agents on Databricks. Participants will learn how to deploy agents as Databricks Apps using Declarative Automation Bundles (DABs), integrate tools via the Model Context Protocol (MCP), instrument agents with MLflow Tracing, and evaluate production quality using scorers, multi-turn judges, and online evaluation. Through hands-on demos and labs, participants will gain practical experience building, observing, and monitoring production-grade AI agents on the Databricks platform.
# MAGIC
# MAGIC ----
# MAGIC
# MAGIC ## Prerequisites
# MAGIC The content was developed for participants with these skills/knowledge/abilities:
# MAGIC - Familiarity with Databricks workspace and notebooks
# MAGIC - Familiarity with Unity Catalog
# MAGIC - Experience building agents using the OpenAI Agents SDK
# MAGIC - Basic knowledge of MLflow and Python
# MAGIC - Familiarity with GenAI agent concepts (LLM calls, tool invocation, retrieval)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Course Agenda
# MAGIC The following modules are part of the **Generative AI Application Deployment and Monitoring** course by **Databricks Academy**.
# MAGIC
# MAGIC | # | Type | Notebook | Description |
# MAGIC |---|------|----------|-------------|
# MAGIC | 1 | Lecture | [Agent Deployment on Databricks]($./1 - Lecture - Agent Deployment on Databricks) | Agent lifecycle, Databricks Apps, DABs, deployment patterns, resource injection |
# MAGIC | 2 | Demo | [Deploying an Observable Agent]($./2 - Demo - Deploying an Observable Agent) | Progressively build an agent: deploy as an App, add MLflow tracing, integrate MCP tools |
# MAGIC | 3 | Lecture | [Tool Integration and Observability]($./3 - Lecture - Tool Integration and Observability) | Model Context Protocol, MCP server types, MCP lifecycle, MLflow Tracing foundations |
# MAGIC | 4 | Demo | [Tracing for Production Agents]($./4 - Demo - Tracing for Production Agents) | Manual spans, custom attributes, multi-turn session grouping, programmatic trace search |
# MAGIC | 5 | Lecture | [Production Evaluation and Monitoring]($./5 - Lecture - Production Evaluation and Monitoring) | Scorer types, custom scorers, multi-turn judges, sampling strategies, scorer lifecycle |
# MAGIC | 6 | Lab | [Online Evaluation with Multi-Turn Conversations]($./6 - Lab - Online Evaluation with Multi-Turn Conversations) | Deploy judges, create custom scorers, tune sampling rates, run the metrics computation job |
# MAGIC | 7 | Lab | [Backfilling and Archiving Multi-Turn Conversations]($./7 - Lab - Backfilling and Archiving Multi-Turn Conversations) | Backfill historical traces, enable/disable trace archival to Delta, scorer cleanup |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * A Databricks workspace with serverless compute (Version 5)
# MAGIC * Access to a Foundation Model serving endpoint (e.g., `databricks-claude-sonnet-4`)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC