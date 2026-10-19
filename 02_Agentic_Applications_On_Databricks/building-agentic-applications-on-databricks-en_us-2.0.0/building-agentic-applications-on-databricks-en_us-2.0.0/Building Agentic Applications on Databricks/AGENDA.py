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
# MAGIC ## Building Agentic Applications on Databricks
# MAGIC
# MAGIC This course teaches students how to build production-grade agentic applications on Databricks. Students learn to create governed agent tools using Unity Catalog and MCP, build single and multi-agent systems with the OpenAI Agents SDK, and build agents with other popular frameworks such as LangChain and DSPy. The course covers the full progression from tool prototyping to production deployment, with hands-on experience using MLflow tracing to observe agent execution. Through practical demonstrations and labs, students develop skills in building, orchestrating, and governing AI agents on the Databricks platform.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC ### Python-Specific Skills
# MAGIC - Basic Python syntax and data structures (lists, dictionaries)
# MAGIC - Understanding of functions, classes, and decorators
# MAGIC - Experience with Python package management and imports
# MAGIC - Familiarity with JSON data handling
# MAGIC - Basic understanding of async/await patterns
# MAGIC
# MAGIC ### SQL-Specific Skills
# MAGIC - Basic SQL query syntax (SELECT, FROM, WHERE)
# MAGIC - Understanding of SQL functions and user-defined functions
# MAGIC - Experience with Unity Catalog SQL functions
# MAGIC
# MAGIC ### Databricks-Specific Skills
# MAGIC - Understanding of Databricks workspace navigation and notebook interface
# MAGIC - Knowledge of Unity Catalog structure (catalogs, schemas, tables, volumes, functions)
# MAGIC - Experience with Databricks compute resources and serverless computing
# MAGIC - Familiarity with MLflow experiment tracking
# MAGIC - Understanding of Databricks model serving endpoints
# MAGIC
# MAGIC ### GenAI/Agent-Specific Skills
# MAGIC - Basic understanding of Large Language Models (LLMs) and their capabilities
# MAGIC - Knowledge of prompt engineering and system prompts
# MAGIC - Familiarity with tool-calling agents and function calling concepts
# MAGIC - Basic awareness of the Model Context Protocol (MCP)
# MAGIC
# MAGIC ### Other/Optional Skills
# MAGIC - Experience with YAML configuration files can be helpful
# MAGIC - Familiarity with LangChain or the OpenAI SDK can be helpful
# MAGIC - Basic understanding of REST APIs and HTTP requests can be helpful
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Course Agenda
# MAGIC
# MAGIC The following lessons are part of the **Building Agentic Applications on Databricks** course by **Databricks Academy**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Course Learning Objectives
# MAGIC
# MAGIC By the end of this course, you will be able to:
# MAGIC
# MAGIC 1. **Define AI agents and their architectures** — Explain what AI agents are, identify their core components (reasoning, planning, tool use), and determine when to use a single-agent vs. multi-agent architecture.
# MAGIC 2. **Create governed agent tools in Unity Catalog** — Build and register both SQL and Python functions in Unity Catalog, equipping them with descriptive names, type hints, and structured docstrings that enable reliable agent tool selection.
# MAGIC 3. **Assess and improve tool documentation for agent use** — Identify documentation gaps that cause agents to misuse tools, and rewrite function signatures and docstrings to improve agent behavior.
# MAGIC 4. **Expose and discover tools via the Model Context Protocol (MCP)** — Use Databricks Managed MCP servers to connect agents to Unity Catalog functions, AI Search indexes, and other Databricks services at runtime.
# MAGIC 5. **Build single-agent systems with the OpenAI Agents SDK** — Configure the SDK with Databricks model routing, define custom tools with `@function_tool`, and build a functional agent that reasons over and calls tools.
# MAGIC 6. **Orchestrate multi-agent systems with handoffs** — Build supervisor-worker architectures using the OpenAI Agents SDK, route requests to specialized worker agents via handoffs, and refine routing through YAML-based instructions.
# MAGIC 7. **Build agents with LangChain and DSPy on Databricks** — Apply the same tool-creation-to-agent workflow using LangChain's `@tool` decorator and `create_agent`, as well as DSPy signatures and `dspy.ReAct` modules.
# MAGIC 8. **Observe agent execution with MLflow tracing** — Enable MLflow autologging, inspect hierarchical span traces across single- and multi-agent flows, and use tracing to debug and improve agent quality.
# MAGIC 9. **Apply Unity Gateway governance to AI agents** — Understand how Unity Gateway provides centralized access control, rate limiting, spend tracking, and fallback routing for model services across agent applications.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC | # | Type | Lesson Name |
# MAGIC |---|------|-------------|
# MAGIC | 01 | Lecture | [Agents, MCP, and AI Governance on Databricks]($./01 Lecture - Agents, MCP, and AI Governance on Databricks) |
# MAGIC | 02 | Demo | [Building Agent Tools on Databricks]($./02 Demo - Building Agent Tools on Databricks) |
# MAGIC | 03 | Lab | [Assessing and Fixing Agent Tools]($./03 Lab - Assessing and Fixing Agent Tools) |
# MAGIC | 04 | Lecture | [Building Agents with Popular Frameworks on Databricks]($./04 Lecture - Building Agents with Popular Frameworks on Databricks) |
# MAGIC | 05 | Demo | [Introduction to OpenAI Agents SDK on Databricks]($./05 Demo - Introduction to OpenAI Agents SDK on Databricks) |
# MAGIC | 06 | Lab | [Multi-Agent Orchestration with the OpenAI Agents SDK on Databricks]($./06 Lab - Multi-Agent Orchestration with The OpenAI Agents SDK on Databricks) |
# MAGIC | 07 | Demo | [Introduction to LangChain on Databricks]($./07 Demo - Introduction to LangChain on Databricks) |
# MAGIC | 08 | Lab | [Building Agents with DSPy on Databricks]($./08 Lab - Building Agents with DSPy on Databricks) |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use **Serverless compute (version 6)**, which is enabled by default.
# MAGIC * This course requires access to Unity Catalog for tool registration and governance.
# MAGIC * The demos and labs call Databricks-hosted foundation models through Unity Gateway.
# MAGIC
# MAGIC The course is structured into three logical sections:
# MAGIC
# MAGIC 1. **Foundations (01-03)** introduces agents, MCP, and AI governance on Databricks, then walks through building and assessing agent tools registered to Unity Catalog
# MAGIC 2. **Building Agents with the OpenAI Agents SDK (04-06)** surveys popular agent frameworks and rapid tool prototyping, builds a single agent with MCP tools and MLflow tracing, then has you build a multi-agent system in a hands-on lab
# MAGIC 3. **Other Frameworks (07-08)** applies the same patterns with LangChain, then has you build a tool-calling agent with DSPy in a hands-on lab
# MAGIC
# MAGIC Each section combines lectures for conceptual understanding with hands-on demos and labs for practical implementation. The progression moves from tool creation to single agents to multi-agent orchestration, and then to other frameworks, ensuring students build a comprehensive understanding of agentic application development on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>