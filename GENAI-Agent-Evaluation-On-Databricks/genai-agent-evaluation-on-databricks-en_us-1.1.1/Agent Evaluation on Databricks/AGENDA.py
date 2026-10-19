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
# MAGIC ## Agent Evaluation on Databricks
# MAGIC
# MAGIC This course teaches students how to systematically evaluate AI agents using MLflow's evaluation framework, addressing the unique challenges of non-deterministic AI systems that traditional software testing cannot handle. Students learn to implement various evaluation approaches including built-in judges for common criteria like correctness and safety, guideline judges for business-specific requirements, and custom judges for specialized needs. The course covers both offline evaluation using curated datasets and online production monitoring, with hands-on experience using MLflow's tracing capabilities to understand agent execution patterns and collect human feedback from different stakeholder types. Through practical demonstrations and labs, students develop skills in creating evaluation workflows that drive continuous quality improvements throughout the AI agent development lifecycle.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC ### Python-Specific Skills
# MAGIC - Basic Python syntax and data structures (lists, dictionaries)
# MAGIC - Understanding of functions, classes, and object-oriented programming concepts
# MAGIC - Experience with Python package management and imports
# MAGIC - Familiarity with JSON data handling and file operations
# MAGIC - Basic understanding of lambda functions and list comprehensions
# MAGIC
# MAGIC ### SQL-Specific Skills
# MAGIC - Basic SQL query syntax (SELECT, FROM, WHERE)
# MAGIC - Understanding of table joins and aggregations
# MAGIC - Knowledge of SQL functions and data types
# MAGIC - Experience with Unity Catalog SQL functions and procedures
# MAGIC
# MAGIC ### Databricks-Specific Skills
# MAGIC - Understanding of Databricks workspace navigation and notebook interface
# MAGIC - Knowledge of Unity Catalog structure (catalogs, schemas, tables, volumes)
# MAGIC - Experience with Databricks compute resources and serverless computing
# MAGIC - Familiarity with MLflow experiment tracking and model registry
# MAGIC - Understanding of Databricks model serving endpoints and deployment
# MAGIC - Awareness of knowledge of Unity Catalog managed tables
# MAGIC
# MAGIC ### GenAI/Agent-Specific Skills
# MAGIC - Basic understanding of Large Language Models (LLMs) and their capabilities
# MAGIC - Knowledge of prompt engineering and system prompts
# MAGIC - Understanding of tool-calling agents and function calling concepts
# MAGIC - Somewhat familiar with evaluation metrics for AI systems (correctness, safety, relevance)
# MAGIC - Basic knowledge of MLflow tracing and agent evaluation frameworks
# MAGIC - Somewhat familiar with the notion of human feedback collection and assessment workflows
# MAGIC
# MAGIC ### Other/Optional Skills
# MAGIC - Experience with YAML configuration files can be helpful
# MAGIC - Basic understanding of REST APIs and HTTP requests can be helpful
# MAGIC - Understanding of version control and model lifecycle management for ML models can be helpful
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Course Agenda
# MAGIC
# MAGIC The following modules are part of the **AI Agent Evaluation with MLflow on Databricks** course by **Databricks Academy**.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC | # | Module Name | Lesson Name |
# MAGIC |---|-------------|-------------|
# MAGIC | 1 | [Module 1 - AI Agent Evaluation Fundamentals]($./Module 1 - AI Agent Evaluation Fundamentals) | • *Lecture:* [The Challenge of Evaluating AI Agents]($./Module 1 - AI Agent Evaluation Fundamentals/1.1 Lecture - The Challenge of Evaluating AI Agents)  <br> • *Demo:* [Agent Setup]($./Module 1 - AI Agent Evaluation Fundamentals/1.2 Demo - Agent Setup) <br> • *Lecture:* [MLflow's Evaluation Framework]($./Module 1 - AI Agent Evaluation Fundamentals/1.3 Lecture - MLflow's Evaluation Framework) |
# MAGIC | 2 | [Module 2 - Built-In and Guideline Judges]($./Module 2 - Built-In and Guideline Judges) | • *Lecture:* [Built-In Judges]($./Module 2 - Built-In and Guideline Judges/2.1 Lecture - Built-In Judges) <br> • *Demo:* [Using MLflow Built-In Judges]($./Module 2 - Built-In and Guideline Judges/2.2 Demo - Using MLflow Built-In Judges) <br> • *Lecture:* [Guideline Judges]($./Module 2 - Built-In and Guideline Judges/2.3 Lecture - Guideline Judges) <br> • *Demo:* [Guideline Judges with MLflow]($./Module 2 - Built-In and Guideline Judges/2.4 Demo - Guideline Judges with MLflow) <br> • *Lecture:* [Custom Judges and Feedback]($./Module 2 - Built-In and Guideline Judges/2.5 Lecture - Custom Judges and Feedback) <br> • *Demo:* [Custom Judges with MLflow]($./Module 2 - Built-In and Guideline Judges/2.6 Demo - Custom Judges with MLflow) <br> • *Lab:* [Applying Agent Evaluation]($./Module 2 - Built-In and Guideline Judges/2.7 Lab - Applying Agent Evaluation) |
# MAGIC | 3 | [Module 3 - Custom Judges and Human Feedback]($./Module 3 - Custom Judges and Human Feedback) | • *Lecture:* [Offline vs. Online Evaluation Strategies]($./Module 3 - Custom Judges and Human Feedback/3.1 Lecture - Offline vs. Online Evaluation Strategies) <br> • *Lab:* [Developer and SME Feedback with MLflow]($./Module 3 - Custom Judges and Human Feedback/3.2 Lab - Developer and SME Feedback with MLflow) |
# MAGIC | | **Appendix** | • [Agent Deployment Patterns]($./Appendix - Agent Deployment Patterns) |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use **Serverless compute (version 5)**, which is enabled by default.
# MAGIC * This course requires access to MLflow's evaluation framework and Unity Catalog for agent registration and governance.
# MAGIC * Some demonstrations require model serving endpoints for agent deployment and evaluation.
# MAGIC
# MAGIC The course is structured into three logical modules:
# MAGIC
# MAGIC 1. **Module 1** introduces the fundamental challenges of AI agent evaluation and MLflow's framework
# MAGIC 2. **Module 2** covers the core evaluation approaches using built-in and guideline judges
# MAGIC 3. **Module 3** explores advanced topics including custom judges, human feedback, and evaluation strategies
# MAGIC
# MAGIC Each module combines lectures for conceptual understanding with hands-on demos and labs for practical implementation. The progression moves from basic concepts to advanced techniques, ensuring students build a comprehensive understanding of AI agent evaluation using MLflow on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC