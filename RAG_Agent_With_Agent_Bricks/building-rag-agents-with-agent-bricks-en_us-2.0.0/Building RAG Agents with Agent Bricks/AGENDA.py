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
# MAGIC ## Building RAG Agents with Agent Bricks
# MAGIC
# MAGIC This course provides hands-on training for building retrieval agents on Databricks. Participants will learn to create Genie Agents for unstructured data, perform information retrieval with Lakebase Search, parse and transform unstructured documents into structured data using AI Functions, chunk text for semantic retrieval, and build AI Search indexes with multiple search modes.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Basic SQL knowledge for querying tables and using built-in functions
# MAGIC - Basic Python programming experience
# MAGIC - Familiarity with the Databricks workspace UI (e.g. navigating Unity Catalog)
# MAGIC - Understanding of Unity Catalog concepts including catalogs, schemas, and volumes
# MAGIC - Familiarity with fundamental LLM concepts (e.g. embeddings, retrieval-augmented generation)
# MAGIC - Understanding of structured vs unstructured data
# MAGIC
# MAGIC ---
# MAGIC ## Course Agenda
# MAGIC The following notebooks are part of the **Building RAG Agents with Agent Bricks** course by **Databricks Academy**.
# MAGIC
# MAGIC | # | Type | Notebook | Duration |
# MAGIC | - | ---- | -------- | -------- |
# MAGIC | 1 | Lecture | [Retrieval Agents and Lakebase Search on Databricks]($./01 Lecture - Retrieval Agents and Lakebase Search on Databricks) | 30 min |
# MAGIC | 2 | Demo | [Creating a Genie Agent for Unstructured Data]($./02 Demo - Creating a Genie Agent for Unstructured Data) | 30 min |
# MAGIC | 3 | Lab | [Information Retrieval with Lakebase Search]($./03 Lab - Information Retrieval with Lakebase Search) | 30 min |
# MAGIC | 4 | Lecture | [Document Parsing and Chunking Strategies]($./04 Lecture - Document Parsing and Chunking Strategies) | 30 min |
# MAGIC | 5 | Demo | [Transforming PDFs to Structured Data]($./05 Demo - Transforming PDFs to Structured Data) | 35 min |
# MAGIC | 6 | Lecture | [AI Search on Databricks]($./06 Lecture - AI Search on Databricks) | 20 min |
# MAGIC | 7 | Lab | [Chunking PDFs and AI Search]($./07 Lab - Chunking PDFs and AI Search) | 45 min |
# MAGIC | 8 | Lecture | [APPENDIX]($./08 Lecture - APPENDIX) | Reference |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Technical Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the course:
# MAGIC
# MAGIC - Use **Serverless Compute (environment version 6)** for running all demo and lab notebooks
# MAGIC - Access to **`ai_parse_document()`**, **`ai_prep_search()`**, **`ai_classify()`**, and **`ai_extract()`** functions (Beta features)
# MAGIC - Access to **Databricks Agent Bricks** (Genie Agents)
# MAGIC - A pre-provisioned **Lakebase Postgres** project and branch (provisioned during workspace setup)
# MAGIC - A pre-created **AI Search endpoint** (`vs_all_demo`, provisioned during workspace setup)
# MAGIC - Access to **Foundation Model APIs** for managed embeddings (`databricks-gte-large-en`)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>