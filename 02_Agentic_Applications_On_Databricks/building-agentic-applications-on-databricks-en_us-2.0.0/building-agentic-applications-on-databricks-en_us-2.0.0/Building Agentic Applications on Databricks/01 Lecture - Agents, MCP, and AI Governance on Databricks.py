# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Lecture Overview and Learning Objectives
# MAGIC %md
# MAGIC # Lecture - Agents, MCP, and AI Governance on Databricks
# MAGIC #### Duration: 20 minutes
# MAGIC ## Overview
# MAGIC This lecture establishes the foundations for building AI agents on Databricks. We cover what agents are, how they use tools registered in Unity Catalog, how the Model Context Protocol (MCP) enables tool discovery and execution, and how Unity Gateway provides governance across the stack.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lecture, you will be able to:
# MAGIC 1. Define what AI agents are and identify their core components
# MAGIC 2. Explain how Unity Catalog functions serve as governed agent tools
# MAGIC 3. Describe how MCP enables agents to discover and call tools at runtime
# MAGIC 4. Identify common tool patterns: structured retrieval, unstructured retrieval, code interpreter, and external connections
# MAGIC 5. Understand the role of Unity Gateway for governance and access control

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. What are AI Agents?

# COMMAND ----------

# DBTITLE 1,A1 - Single Agents
# MAGIC %md-sandbox
# MAGIC ### A1. Single Agents
# MAGIC
# MAGIC An **AI agent** is an intelligent software system that can perceive its environment, make decisions, and take actions to achieve specific goals. Unlike traditional AI systems that require continuous inputs from users, AI agents are autonomous systems that can:
# MAGIC
# MAGIC - **Reason** about complex problems and situations
# MAGIC - **Plan** sequences of actions to achieve objectives
# MAGIC - **Adapt** their behavior based on new information
# MAGIC - **Interact** with external systems and data sources
# MAGIC - **Learn** from experience to improve future performance
# MAGIC
# MAGIC A very simple example of agent framework is shown below. Here, a system prompt and the user's prompt are passed to the agent, where it then uses tools to help plan and process information to return back to the user.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 680 385" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Agent Framework</title>
# MAGIC   <desc>Agent Framework containing Generative AI models, Tools, and Sequence of decisions, taking System Prompt and User Prompt as input and producing Response and Output.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="af-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#618794" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <rect x="25" y="15" width="176" height="74" rx="8" fill="#ffffff" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="113" y="40" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="600" fill="#0b2026">System Prompt</text>
# MAGIC   <text x="113" y="64" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="600" fill="#0b2026">+ User Prompt</text>
# MAGIC   <rect x="479" y="15" width="176" height="74" rx="8" fill="#ffffff" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="567" y="52" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="600" fill="#0b2026">Response/Output</text>
# MAGIC   <path d="M113 89 L113 107" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#af-arrow)"/>
# MAGIC   <path d="M567 107 L567 89" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#af-arrow)"/>
# MAGIC   <rect x="15" y="110" width="650" height="265" rx="14" fill="none" stroke="#618794" stroke-width="1.5" stroke-opacity="0.4"/>
# MAGIC   <text x="34" y="137" text-anchor="start" font-size="14" font-weight="700" fill="#FF5F46">Agent Framework</text>
# MAGIC   <rect x="32" y="154" width="278" height="181" rx="10" fill="#ffffff"/>
# MAGIC   <rect x="48" y="188" width="504" height="70" rx="12" fill="#2272B4" fill-opacity="0.10"/>
# MAGIC   <text x="171" y="176" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="600" fill="#0b2026">Generative AI</text>
# MAGIC   <foreignObject x="86" y="194" width="58" height="58">
# MAGIC     <div xmlns="http://www.w3.org/1999/xhtml" style="width:100%;height:100%;">
# MAGIC       <img src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/agent-icon.png" style="width:100%;height:100%;display:block;" alt="AI brain icon"/>
# MAGIC     </div>
# MAGIC   </foreignObject>
# MAGIC   <foreignObject x="210" y="194" width="58" height="58">
# MAGIC     <div xmlns="http://www.w3.org/1999/xhtml" style="width:100%;height:100%;">
# MAGIC       <img src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/agent-icon.png" style="width:100%;height:100%;display:block;" alt="AI brain icon"/>
# MAGIC     </div>
# MAGIC   </foreignObject>
# MAGIC   <text x="387" y="223" text-anchor="middle" dominant-baseline="central" font-size="50" font-weight="500" font-style="italic" fill="#0b2026">fx</text>
# MAGIC   <rect x="438" y="202" width="54" height="42" rx="3" fill="#ffffff" stroke="#0b2026" stroke-width="1.5"/>
# MAGIC   <line x1="438" y1="213" x2="492" y2="213" stroke="#0b2026" stroke-width="1"/>
# MAGIC   <circle cx="444" cy="208" r="1.2" fill="#0b2026"/>
# MAGIC   <circle cx="449" cy="208" r="1.2" fill="#0b2026"/>
# MAGIC   <circle cx="454" cy="208" r="1.2" fill="#0b2026"/>
# MAGIC   <circle cx="465" cy="229" r="10" fill="none" stroke="#0b2026" stroke-width="1.3"/>
# MAGIC   <ellipse cx="465" cy="229" rx="10" ry="3.5" fill="none" stroke="#0b2026" stroke-width="1"/>
# MAGIC   <line x1="465" y1="219" x2="465" y2="239" stroke="#0b2026" stroke-width="1"/>
# MAGIC   <rect x="48" y="188" width="504" height="70" rx="12" fill="none" stroke="#2272B4" stroke-width="1.5" stroke-dasharray="6 3"/>
# MAGIC   <rect x="32" y="154" width="278" height="181" rx="10" fill="none" stroke="#618794" stroke-width="1.2"/>
# MAGIC   <text x="542" y="204" text-anchor="end" dominant-baseline="central" font-size="14" font-weight="600" fill="#0b2026">Tools</text>
# MAGIC   <text x="115" y="273" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#0b2026">Image</text>
# MAGIC   <text x="115" y="289" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#0b2026">Generation</text>
# MAGIC   <text x="239" y="273" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#0b2026">Language</text>
# MAGIC   <text x="239" y="289" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="500" fill="#0b2026">Model</text>
# MAGIC   <text x="171" y="318" text-anchor="middle" dominant-baseline="central" font-size="14" fill="#0b2026">(GenAI Examples)</text>
# MAGIC   <path d="M438 262 L438 274" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#af-arrow)"/>
# MAGIC   <path d="M452 274 L452 262" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#af-arrow)"/>
# MAGIC   <rect x="334" y="278" width="310" height="90" rx="14" fill="#EEEDE9" stroke="#618794" stroke-width="1.5"/>
# MAGIC   <rect x="354" y="292" width="272" height="12" rx="6" fill="#618794"/>
# MAGIC   <rect x="354" y="310" width="272" height="12" rx="6" fill="#618794"/>
# MAGIC   <rect x="354" y="328" width="272" height="12" rx="6" fill="#00A972"/>
# MAGIC   <text x="489" y="356" text-anchor="middle" dominant-baseline="central" font-size="14" fill="#0b2026">Sequence of decisions</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC What makes a software system an AI agent?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p>A plain LLM call maps one prompt to one response. An <strong>agent</strong> wraps that model in a loop that can take actions and react to their results:</p>
# MAGIC <ul style="margin: 12px 0; padding-left: 24px;">
# MAGIC <li><strong>Reason &amp; plan</strong> &mdash; break a goal into steps instead of answering in one shot.</li>
# MAGIC <li><strong>Use tools</strong> &mdash; call functions, query data, or reach external systems to get real information.</li>
# MAGIC <li><strong>Observe &amp; adapt</strong> &mdash; read each tool result and decide the next action, looping until the goal is met.</li>
# MAGIC <li><strong>Stay grounded</strong> &mdash; rely on retrieved, governed data rather than only the model's frozen training knowledge.</li>
# MAGIC </ul>
# MAGIC <p>On Databricks, the tools an agent calls are <strong>Unity Catalog functions</strong> &mdash; governed, discoverable, and permission-checked &mdash; which is what makes agent actions auditable and safe in production.</p>
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# DBTITLE 1,A2 - Multi-Agent Supervisors
# MAGIC %md-sandbox
# MAGIC ### A2. Multi-Agent Supervisors
# MAGIC
# MAGIC An example of a common multi-agent architecture is the **supervisor-worker** (also called router-delegate) pattern:
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 680 310" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Supervisor-Worker Multi-Agent Architecture</title>
# MAGIC   <desc>A Supervisor Agent receives user requests and delegates to specialized worker agents: Data Agent, Research Agent, and General Agent.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="sw-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#618794" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <rect x="15" y="15" width="650" height="280" rx="14" fill="none" stroke="#618794" stroke-width="1.5" stroke-opacity="0.4"/>
# MAGIC   <text x="34" y="42" text-anchor="start" font-size="14" font-weight="700" fill="#FF5F46">Multi-Agent System</text>
# MAGIC   <rect x="240" y="60" width="200" height="60" rx="10" fill="#ffffff" stroke="#FF5F46" stroke-width="2"/>
# MAGIC   <text x="340" y="82" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Supervisor Agent</text>
# MAGIC   <text x="340" y="104" text-anchor="middle" font-size="11" fill="#618794">Routes and synthesizes</text>
# MAGIC   <path d="M270 120 L152 165" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#sw-arrow)"/>
# MAGIC   <path d="M340 120 L340 165" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#sw-arrow)"/>
# MAGIC   <path d="M410 120 L528 165" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#sw-arrow)"/>
# MAGIC   <text x="275" y="143" text-anchor="middle" font-size="12" font-style="italic" fill="#618794">Delegates to</text>
# MAGIC   <rect x="48" y="168" width="584" height="110" rx="12" fill="#2272B4" fill-opacity="0.10"/>
# MAGIC   <rect x="48" y="168" width="584" height="110" rx="12" fill="none" stroke="#2272B4" stroke-width="1.5" stroke-dasharray="6 3"/>
# MAGIC   <text x="622" y="183" text-anchor="end" font-size="14" font-weight="600" fill="#0b2026">Workers</text>
# MAGIC   <rect x="68" y="194" width="160" height="66" rx="8" fill="#EBF3FA" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC   <text x="148" y="220" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Data Agent</text>
# MAGIC   <text x="148" y="242" text-anchor="middle" font-size="11" fill="#618794">UC functions, SQL</text>
# MAGIC   <rect x="260" y="194" width="160" height="66" rx="8" fill="#EBF3FA" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC   <text x="340" y="220" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Research Agent</text>
# MAGIC   <text x="340" y="242" text-anchor="middle" font-size="11" fill="#618794">AI Search, docs</text>
# MAGIC   <rect x="452" y="194" width="160" height="66" rx="8" fill="#EBF3FA" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC   <text x="532" y="220" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">General Agent</text>
# MAGIC   <text x="532" y="242" text-anchor="middle" font-size="11" fill="#618794">Knowledge, reasoning</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. The **supervisor** receives the user's request and analyzes intent
# MAGIC 2. It **delegates** to the appropriate worker agent (or handles simple requests itself)
# MAGIC 3. The worker executes with its specialized tools and returns a result
# MAGIC 4. The supervisor **synthesizes** the response and returns it to the user
# MAGIC
# MAGIC For example, in the OpenAI Agents SDK, this delegation is implemented using **handoffs**:
# MAGIC
# MAGIC ```python
# MAGIC from agents import Agent
# MAGIC
# MAGIC supervisor = Agent(
# MAGIC     name="Supervisor",
# MAGIC     instructions="Route requests to the appropriate specialist.",
# MAGIC     handoffs=[data_agent, business_agent],
# MAGIC )
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,A3 - Single-Agent vs. Multi-Agent
# MAGIC %md-sandbox
# MAGIC ### A3. Single-Agent vs. Multi-Agent
# MAGIC
# MAGIC Factor                 | Single Agent                                              | Multi-Agent
# MAGIC ---------------------- | --------------------------------------------------------- | -----------------------------------------------
# MAGIC Number of tools        | Works best when total tools is modest (≈ up to 8–10); beyond that, tool selection and prompt management start to degrade (heuristic, not a hard limit). | Better when you have many tools spanning distinct capabilities (often >10) and need clear separation between them.
# MAGIC Domain complexity      | One focused or tightly related domain.                   | Multiple distinct domains or verticals.
# MAGIC Instruction length     | Shorter, coherent system prompt that a single agent can reliably follow. | Would require a very long, fragmented, or conflicting prompt if kept in one agent.
# MAGIC Failure isolation      | Failures are acceptable as one unit; easier to reason about as a whole. | Need to isolate, debug, and replace individual agent capabilities without changing the whole system.
# MAGIC Team ownership         | Single team owns most or all behaviors.                  | Different teams own different domains, tools, or behaviors and need independent evolution.
# MAGIC Latency                | Minimal overhead; typically fewer model calls and simpler orchestration. | Extra routing/orchestration calls add latency and token cost.
# MAGIC Scalability / parallelism | Scales fine while domain and tools remain bounded; limited parallelism within one agent’s context window. | Better for parallelizing work across domains, distributing context, and scaling capabilities, at the cost of more coordination complexity.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Start with a single agent. Only decompose into multiple agents when you observe tool selection errors, instruction conflicts, or when different teams need to own different agent capabilities independently.
# MAGIC </p>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# DBTITLE 1,A4 - Building Agents on Databricks
# MAGIC %md-sandbox
# MAGIC ### A4. Agents on Databricks
# MAGIC
# MAGIC Now that we understand single and multi-agent architectures, let's look at how to build agents on Databricks. The platform offers a spectrum of approaches — from fully managed, no-code options to fully custom, code-first development.
# MAGIC
# MAGIC <div style="display: flex; gap: 18px; flex-wrap: wrap; justify-content: center; margin: 20px 0; font-family: sans-serif;">
# MAGIC
# MAGIC   <div style="flex: 1 1 300px; max-width: 340px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC     <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #5A6F77;"></div>
# MAGIC     <div style="display: inline-block; background: #5A6F77; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px;">LEGACY</div>
# MAGIC     <div style="font-size: 17px; font-weight: 700; color: #0b2026;">Knowledge Assistant</div>
# MAGIC     <div style="font-size: 14px; color: #5A6F77; margin-top: 6px; line-height: 1.5;">Bring your own AI Search index, choose your chunking strategy, and attach multiple source types (up to 10). Full control over the retrieval pipeline.</div>
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="flex: 1 1 300px; max-width: 340px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC     <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #00A972;"></div>
# MAGIC     <div style="display: inline-block; background: #00A972; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px;">MANAGED</div>
# MAGIC     <div style="font-size: 17px; font-weight: 700; color: #0b2026;">Genie Agent</div>
# MAGIC     <div style="font-size: 14px; color: #5A6F77; margin-top: 6px; line-height: 1.5;">Declarative agents that understand your data. Attach tables for structured queries or UC volumes for document search — Databricks handles tool orchestration and query generation.</div>
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="flex: 1 1 300px; max-width: 340px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC     <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #00A972;"></div>
# MAGIC     <div style="display: inline-block; background: #00A972; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px;">CODE-FIRST</div>
# MAGIC     <div style="font-size: 17px; font-weight: 700; color: #0b2026;">Custom Agent</div>
# MAGIC     <div style="font-size: 14px; color: #5A6F77; margin-top: 6px; line-height: 1.5;">Write your own agent logic, tools, and orchestration. Use the OpenAI Agents SDK, LangGraph, or any Python framework, then deploy it as a Databricks App, the recommended way to host agents.</div>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC **Knowledge Assistants** offer customization (bring your own AI Search index) but are a legacy path. **Genie Agents** are the recommended declarative option for data-aware agents: you declare *what* data or documents your agent should work with, and the platform optimizes *how*. **Custom agents** are code-first: you own the agent logic, tools, and orchestration end to end, and Databricks Apps is the recommended way to host them.
# MAGIC
# MAGIC #### The Frontier: Platform Capabilities for Custom Agents
# MAGIC
# MAGIC Custom agents don't have to be built from scratch. Databricks is adding platform capabilities that take care of common agent infrastructure for you. Here's what the frontier currently looks like:
# MAGIC
# MAGIC <div style="display: flex; gap: 18px; flex-wrap: wrap; justify-content: center; margin: 20px 0 12px 0; font-family: sans-serif;">
# MAGIC
# MAGIC   <div style="flex: 1 1 300px; max-width: 340px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC     <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #FF5F46;"></div>
# MAGIC     <div style="display: inline-block; background: #FF5F46; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px;">NEW · BETA</div>
# MAGIC     <div style="font-size: 17px; font-weight: 700; color: #0b2026;">Agent developer CLI</div>
# MAGIC     <div style="font-size: 14px; color: #5A6F77; margin-top: 6px; line-height: 1.5;">Create a “hello world” custom agent in under 10 minutes with a handful of commands, pre-configured with the capabilities below.</div>
# MAGIC     <div style="font-size: 13px; margin-top: 10px; color: #5A6F77;"><a href="https://docs.databricks.com/aws/en/agents/custom-agents/agent-bricks-cli" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/agents/custom-agents/agent-bricks-cli" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/agents/custom-agents/agent-bricks-cli" style="color: #1976d2; text-decoration: underline;">GCP</a></div>
# MAGIC   </div>
# MAGIC   <div style="flex: 1 1 300px; max-width: 340px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC     <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #FF5F46;"></div>
# MAGIC     <div style="display: inline-block; background: #FF5F46; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px;">NEW · BETA</div>
# MAGIC     <div style="font-size: 17px; font-weight: 700; color: #0b2026;">Managed memory and sessions</div>
# MAGIC     <div style="font-size: 14px; color: #5A6F77; margin-top: 6px; line-height: 1.5;">Help agents get the right context at the right time, within a conversation and across sessions.</div>
# MAGIC     <div style="font-size: 13px; margin-top: 10px; color: #5A6F77;"><a href="https://docs.databricks.com/aws/en/agents/custom-agents/stateful-agents" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/agents/custom-agents/stateful-agents" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/agents/custom-agents/stateful-agents" style="color: #1976d2; text-decoration: underline;">GCP</a></div>
# MAGIC   </div>
# MAGIC   <div style="flex: 1 1 300px; max-width: 340px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC     <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #FF5F46;"></div>
# MAGIC     <div style="display: inline-block; background: #FF5F46; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px;">NEW · BETA</div>
# MAGIC     <div style="font-size: 17px; font-weight: 700; color: #0b2026;">Agentic sandboxes</div>
# MAGIC     <div style="font-size: 14px; color: #5A6F77; margin-top: 6px; line-height: 1.5;">Let agents access data and run code in an isolated, secure environment.</div>
# MAGIC     <div style="font-size: 13px; margin-top: 10px; color: #5A6F77;"><a href="https://docs.databricks.com/aws/en/compute/serverless/sandbox" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/sandbox" style="color: #1976d2; text-decoration: underline;">Azure</a></div>
# MAGIC   </div>
# MAGIC   <div style="flex: 1 1 300px; max-width: 340px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC     <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #00A972;"></div>
# MAGIC     <div style="display: inline-block; background: #00A972; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px;">GA</div>
# MAGIC     <div style="font-size: 17px; font-weight: 700; color: #0b2026;">Agent hosting</div>
# MAGIC     <div style="font-size: 14px; color: #5A6F77; margin-top: 6px; line-height: 1.5;">Host your agent on Databricks serverless infrastructure.</div>
# MAGIC     <div style="font-size: 13px; margin-top: 10px; color: #5A6F77;"><a href="https://docs.databricks.com/aws/en/agents/custom-agents/author-agent" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/agents/custom-agents/author-agent" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/agents/custom-agents/author-agent" style="color: #1976d2; text-decoration: underline;">GCP</a></div>
# MAGIC   </div>
# MAGIC   <div style="flex: 1 1 300px; max-width: 340px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC     <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #00A972;"></div>
# MAGIC     <div style="display: inline-block; background: #00A972; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px;">GA</div>
# MAGIC     <div style="font-size: 17px; font-weight: 700; color: #0b2026;">Tracing and evaluation</div>
# MAGIC     <div style="font-size: 14px; color: #5A6F77; margin-top: 6px; line-height: 1.5;">End-to-end tracing and evaluation for agents with MLflow. You will use MLflow tracing throughout this course.</div>
# MAGIC     <div style="font-size: 13px; margin-top: 10px; color: #5A6F77;"><a href="https://docs.databricks.com/aws/en/mlflow3/genai/agent-observability-and-quality" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/agent-observability-and-quality" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/mlflow3/genai/agent-observability-and-quality" style="color: #1976d2; text-decoration: underline;">GCP</a></div>
# MAGIC   </div>
# MAGIC   <div style="flex: 1 1 300px; max-width: 340px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC     <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #00A972;"></div>
# MAGIC     <div style="display: inline-block; background: #00A972; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px;">GA</div>
# MAGIC     <div style="font-size: 17px; font-weight: 700; color: #0b2026;">Out of the box tools and MCPs</div>
# MAGIC     <div style="font-size: 14px; color: #5A6F77; margin-top: 6px; line-height: 1.5;">Pre-built first- and third-party MCP servers, plus pre-built tools for web search, code execution, and more.</div>
# MAGIC     <div style="font-size: 13px; margin-top: 10px; color: #5A6F77;"><a href="https://docs.databricks.com/aws/en/agents/mcp-tools/" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/agents/mcp-tools/" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/agents/mcp-tools/" style="color: #1976d2; text-decoration: underline;">GCP</a></div>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Beta features can change and may need to be enabled in your workspace before you can use them. Availability can also differ by cloud: Databricks Sandbox documentation is currently available for AWS and Azure only.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Tools and Common Patterns

# COMMAND ----------

# DBTITLE 1,B1 - Model Context Protocol (MCP)
# MAGIC %md-sandbox
# MAGIC ### B1. Model Context Protocol (MCP)
# MAGIC
# MAGIC On Databricks, [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro) is an open standard for connecting AI agents to tools and context such as data, functions, and external services. Databricks managed MCP servers ([AWS](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/managed-mcp) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/mcp/managed-mcp)) are Databricks-hosted, ready-to-use MCP endpoints for AI Search, Genie spaces, Databricks SQL, and Unity Catalog functions. They let agents connect to these tools through a standard interface while Unity Catalog permissions remain enforced and usage can be monitored and managed through Unity Gateway.
# MAGIC
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 680 375" role="img" style="font-family: sans-serif;">
# MAGIC   <title>MCP on Databricks</title>
# MAGIC   <desc>User query flows into an agent on Databricks that acts as an MCP client. The agent discovers and calls tools from managed MCP servers (Genie for structured data, AI Search for unstructured data, UC Functions for custom logic) and returns a response to the user. All traffic governed by Unity Catalog and Unity Gateway.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="mcp-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#618794" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC
# MAGIC   <rect x="25" y="15" width="176" height="74" rx="8" fill="#ffffff" stroke="#618794" stroke-width="1.5"/>
# MAGIC   <text x="113" y="52" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="600" fill="#0b2026">User Query</text>
# MAGIC
# MAGIC   <rect x="479" y="15" width="176" height="74" rx="8" fill="#ffffff" stroke="#618794" stroke-width="1.5"/>
# MAGIC   <text x="567" y="52" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="600" fill="#0b2026">Response</text>
# MAGIC
# MAGIC   <path d="M113 89 L113 107" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#mcp-arrow)"/>
# MAGIC   <path d="M567 107 L567 89" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#mcp-arrow)"/>
# MAGIC
# MAGIC   <rect x="15" y="110" width="650" height="250" rx="14" fill="none" stroke="#618794" stroke-width="1.5" stroke-opacity="0.4"/>
# MAGIC   <text x="34" y="137" text-anchor="start" font-size="14" font-weight="700" fill="#FF5F46">Databricks</text>
# MAGIC
# MAGIC   <rect x="32" y="154" width="210" height="160" rx="10" fill="#ffffff" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="137" y="176" text-anchor="middle" dominant-baseline="central" font-size="14" font-weight="600" fill="#0b2026">Agent</text>
# MAGIC   <foreignObject x="107" y="196" width="60" height="60">
# MAGIC     <div xmlns="http://www.w3.org/1999/xhtml" style="width:100%;height:100%;">
# MAGIC       <img src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/agent-icon.png" style="width:100%;height:100%;display:block;" alt="AI agent icon"/>
# MAGIC     </div>
# MAGIC   </foreignObject>
# MAGIC   <text x="137" y="275" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="500" fill="#0b2026">MCP Client</text>
# MAGIC   <text x="137" y="295" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#618794">Discovers and calls tools</text>
# MAGIC
# MAGIC   <rect x="262" y="174" width="386" height="140" rx="12" fill="#2272B4" fill-opacity="0.10"/>
# MAGIC   <rect x="262" y="174" width="386" height="140" rx="12" fill="none" stroke="#2272B4" stroke-width="1.5" stroke-dasharray="6 3"/>
# MAGIC   <text x="638" y="189" text-anchor="end" dominant-baseline="central" font-size="14" font-weight="600" fill="#0b2026">MCP Servers</text>
# MAGIC
# MAGIC   <rect x="274" y="204" width="114" height="90" rx="8" fill="#EBF3FA" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC   <text x="331" y="225" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="600" fill="#0b2026">Genie</text>
# MAGIC   <text x="331" y="258" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#618794">Structured</text>
# MAGIC   <text x="331" y="274" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#618794">data</text>
# MAGIC
# MAGIC   <rect x="398" y="204" width="114" height="90" rx="8" fill="#EBF3FA" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC   <text x="455" y="225" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="600" fill="#0b2026">AI Search</text>
# MAGIC   <text x="455" y="258" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#618794">Unstructured</text>
# MAGIC   <text x="455" y="274" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#618794">data</text>
# MAGIC
# MAGIC   <rect x="522" y="204" width="114" height="90" rx="8" fill="#EBF3FA" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC   <text x="579" y="225" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="600" fill="#0b2026">UC Functions</text>
# MAGIC   <text x="579" y="258" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#618794">Custom</text>
# MAGIC   <text x="579" y="274" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#618794">logic</text>
# MAGIC
# MAGIC   <path d="M242 222 L262 222" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#mcp-arrow)"/>
# MAGIC   <path d="M262 270 L242 270" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#mcp-arrow)"/>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC The main benefit of MCP is standardization. You can create a tool once and use it with different agent frameworks or clients that support MCP.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Learn More</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Read more in the Databricks docs for MCP on Databricks (<a href="https://docs.databricks.com/aws/en/generative-ai/mcp/" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/generative-ai/mcp/" style="color: #1976d2; text-decoration: underline;">GCP</a>) and managed MCP servers (<a href="https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp/" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/managed-mcp" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/generative-ai/mcp/managed-mcp" style="color: #1976d2; text-decoration: underline;">GCP</a>). For the broader protocol, see the <a href="https://modelcontextprotocol.io/docs/getting-started/intro" style="color: #1976d2; text-decoration: underline;">official MCP documentation</a>.
# MAGIC </p>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC Why does MCP matter for agents?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p><strong>Standardization.</strong> Without a shared protocol, every agent framework needs bespoke glue to reach each tool and data source &mdash; an N-frameworks &times; M-tools integration problem.</p>
# MAGIC <ul style="margin: 12px 0; padding-left: 24px;">
# MAGIC <li><strong>Define once, reuse anywhere</strong> &mdash; expose a tool through MCP and any MCP-aware client or framework can discover and call it.</li>
# MAGIC <li><strong>Runtime discovery</strong> &mdash; agents list the available tools and their schemas at runtime instead of hard-coding them.</li>
# MAGIC <li><strong>Governed by default</strong> &mdash; Databricks managed MCP servers surface Genie, AI Search, Databricks SQL, and Unity Catalog functions behind one interface, with UC permissions enforced and usage tracked through Unity Gateway.</li>
# MAGIC </ul>
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Genie Agents
# MAGIC
# MAGIC The diagram below frames a Genie Agent as a single agent whose main tool is text-to-SQL (optionally augmented with the trusted UC functions you registered) — and, just as important for the enterprise, the **governance model** around it. The user authenticates as themselves; the agent's **ACL** decides whether they can run it; and every query executes **on behalf of that user**, so the warehouse borrows the *author's* compute delegation while data access is enforced against the *end user's own* Unity Catalog permissions and audited back to them.
# MAGIC
# MAGIC <div style="width: 100%;">
# MAGIC <svg width="100%" viewBox="0 0 1240 460" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Genie Agent: single agent with tool calling and on-behalf-of governance</title>
# MAGIC   <desc>Left to right: a user authenticates as themselves; their identity is checked against the Genie Agent ACL (CAN VIEW / CAN RUN); the single managed agent plans and calls its built-in text-to-SQL tool or attached UC functions; SQL runs on a SQL warehouse under on-behalf-of execution, where compute comes from the author's delegation embedded at config time, data access uses the end user's own identity and Unity Catalog permissions, and every query is audited to the user. Unity Catalog beneath the warehouse enforces row filters, column masks, and per-user permissions. A grounded answer with sources returns along the bottom to the user.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="ga-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#618794" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC     <marker id="ga-arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#FF3621" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC
# MAGIC   <!-- User -->
# MAGIC   <rect x="16" y="132" width="140" height="86" rx="10" fill="#ffffff" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="86" y="168" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">User</text>
# MAGIC   <text x="86" y="189" text-anchor="middle" font-size="12" fill="#5A6F77">authenticates</text>
# MAGIC   <text x="86" y="205" text-anchor="middle" font-size="12" fill="#5A6F77">as themselves</text>
# MAGIC
# MAGIC   <!-- User -> ACL -->
# MAGIC   <path d="M156 175 L228 175" fill="none" stroke="#618794" stroke-width="2" marker-end="url(#ga-arrow)"/>
# MAGIC   <text x="192" y="165" text-anchor="middle" font-size="12" font-style="italic" fill="#5A6F77">identity</text>
# MAGIC
# MAGIC   <!-- ACL check -->
# MAGIC   <rect x="228" y="132" width="196" height="86" rx="10" fill="#ffffff" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="326" y="172" text-anchor="middle" font-size="14.5" font-weight="600" fill="#0b2026">Genie Agent ACL check</text>
# MAGIC   <text x="326" y="194" text-anchor="middle" font-size="12.5" fill="#5A6F77">CAN VIEW / CAN RUN?</text>
# MAGIC
# MAGIC   <!-- ACL -> Agent -->
# MAGIC   <path d="M424 175 L468 175" fill="none" stroke="#618794" stroke-width="2" marker-end="url(#ga-arrow)"/>
# MAGIC
# MAGIC   <!-- Genie Agent container -->
# MAGIC   <rect x="468" y="36" width="384" height="290" rx="14" fill="none" stroke="#618794" stroke-width="1.5" stroke-opacity="0.5"/>
# MAGIC   <text x="486" y="66" font-size="16" font-weight="700" fill="#FF5F46">Genie Agent (single managed agent)</text>
# MAGIC
# MAGIC   <!-- Plan & decide -->
# MAGIC   <rect x="486" y="80" width="348" height="56" rx="8" fill="#F9F7F4" stroke="#FF5F46" stroke-width="2"/>
# MAGIC   <text x="660" y="104" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">Plan &amp; decide which tool to call</text>
# MAGIC   <text x="660" y="124" text-anchor="middle" font-size="12.5" fill="#5A6F77">natural language &#8594; action</text>
# MAGIC
# MAGIC   <!-- Tools region -->
# MAGIC   <rect x="486" y="150" width="348" height="164" rx="10" fill="#EBF3FA" stroke="#2272B4" stroke-width="1.5" stroke-dasharray="6 3"/>
# MAGIC   <text x="820" y="172" text-anchor="end" font-size="13" font-weight="600" fill="#0b2026">Tools</text>
# MAGIC
# MAGIC   <rect x="500" y="182" width="320" height="58" rx="8" fill="#ffffff" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC   <text x="660" y="207" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">Text-to-SQL</text>
# MAGIC   <text x="660" y="227" text-anchor="middle" font-size="12" fill="#5A6F77">generates &amp; runs SQL (built-in tool)</text>
# MAGIC
# MAGIC   <rect x="500" y="250" width="320" height="58" rx="8" fill="#ffffff" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC   <text x="660" y="275" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">UC Functions</text>
# MAGIC   <text x="660" y="295" text-anchor="middle" font-size="12" fill="#5A6F77">attached trusted tools &#183; e.g. avg_neigh_price</text>
# MAGIC
# MAGIC   <!-- Agent -> Warehouse -->
# MAGIC   <path d="M852 175 L896 175" fill="none" stroke="#618794" stroke-width="2" marker-end="url(#ga-arrow)"/>
# MAGIC
# MAGIC   <!-- SQL Warehouse governance panel -->
# MAGIC   <rect x="896" y="36" width="320" height="290" rx="12" fill="#ffffff" stroke="#1B5162" stroke-width="2"/>
# MAGIC   <text x="912" y="66" font-size="15" font-weight="700" fill="#0b2026">SQL Warehouse</text>
# MAGIC   <text x="912" y="86" font-size="12" fill="#5A6F77">on-behalf-of execution</text>
# MAGIC
# MAGIC   <rect x="912" y="100" width="288" height="56" rx="6" fill="#EBF3FA" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC   <text x="924" y="124" font-size="13" font-weight="600" fill="#0b2026">COMPUTE &#8212; author's delegation</text>
# MAGIC   <text x="924" y="144" font-size="11" fill="#2272B4">embedded at config time</text>
# MAGIC
# MAGIC   <rect x="912" y="164" width="288" height="56" rx="6" fill="#FDECEA" stroke="#FF3621" stroke-width="1.2"/>
# MAGIC   <text x="924" y="188" font-size="13" font-weight="600" fill="#0b2026">DATA &#8212; end user's identity</text>
# MAGIC   <text x="924" y="208" font-size="11" fill="#FF3621">user's own UC permissions</text>
# MAGIC
# MAGIC   <rect x="912" y="228" width="288" height="56" rx="6" fill="#E7F6EF" stroke="#00A972" stroke-width="1.2"/>
# MAGIC   <text x="924" y="252" font-size="13" font-weight="600" fill="#0b2026">AUDIT &#8212; attributed to the user</text>
# MAGIC   <text x="924" y="272" font-size="11" fill="#00A972">every query is traceable</text>
# MAGIC
# MAGIC   <!-- Warehouse -> Unity Catalog (down) -->
# MAGIC   <path d="M1056 326 L1056 362" fill="none" stroke="#FF3621" stroke-width="2" marker-end="url(#ga-arrow-red)"/>
# MAGIC
# MAGIC   <!-- Unity Catalog -->
# MAGIC   <rect x="906" y="362" width="300" height="76" rx="10" fill="#ffffff" stroke="#FF3621" stroke-width="1.5"/>
# MAGIC   <text x="1056" y="392" text-anchor="middle" font-size="15" font-weight="700" fill="#0b2026">Unity Catalog</text>
# MAGIC   <text x="1056" y="414" text-anchor="middle" font-size="11.5" fill="#5A6F77">row filters &#183; column masks &#183; per-user enforcement</text>
# MAGIC
# MAGIC   <!-- Return: grounded answer + sources, along the bottom back to User -->
# MAGIC   <path d="M906 400 L86 400 L86 220" fill="none" stroke="#618794" stroke-width="2" marker-end="url(#ga-arrow)"/>
# MAGIC   <text x="480" y="390" text-anchor="middle" font-size="12.5" font-style="italic" fill="#5A6F77">grounded answer + sources</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC Why is a Genie Agent considered a single agent with tool calling?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p>A Genie Agent is a <strong>managed single agent</strong>. You do not write its orchestration loop, its prompt, or its retrieval code — Databricks does. But conceptually it behaves exactly like the single agents you already know:</p>
# MAGIC <ul style="margin: 12px 0; padding-left: 24px;">
# MAGIC <li><strong>It calls a tool.</strong> Its primary tool is <em>text-to-SQL</em>: it turns a natural-language question into a SQL query and runs it on a SQL warehouse. You can also attach <strong>trusted UC functions</strong> (like the ones from the previous demo) that it may call within that SQL.</li>
# MAGIC <li><strong>It grounds its answer in sources.</strong> Every response comes with the generated SQL and the rows it returned — the evidence behind the answer.</li>
# MAGIC <li><strong>It runs with governance built in.</strong> Access is gated by the agent's ACL (CAN VIEW / CAN RUN), and queries execute <em>on behalf of the requesting user</em>: the warehouse compute comes from the author's delegation configured on the agent, but data access is enforced against the <strong>end user's own Unity Catalog permissions</strong> — including row filters and column masks — and every query is audited to that user.</li>
# MAGIC <li><strong>It supports a human feedback loop.</strong> Users ask follow-ups in the same conversation (the agent keeps context) and rate answers, which curators use to improve the agent.</li>
# MAGIC </ul>
# MAGIC <p>The difference from the previous demo is <em>ownership</em>: there you wrote a code-first agent and wired tools via MCP; here the agent is managed, and your job as a developer is to <strong>configure it and drive it programmatically</strong>.</p>
# MAGIC
# MAGIC </div>
# MAGIC </details>
# MAGIC

# COMMAND ----------

# DBTITLE 1,Genie Code Callout
# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC
# MAGIC   <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #FF5F46;"></div>
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/lecture 1/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to know more about MCP on Databricks? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/lecture 1/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #F9F7F4; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-query" style="flex: 1;">I need a review of MCP on Databricks. Can I access MCP tools via the UI?</span>
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

# DBTITLE 1,B2 - Other Tools
# MAGIC %md-sandbox
# MAGIC ### B3. Other Tools
# MAGIC While we will focus on agent tools in UC, it's important to point out that there are other tools that exist outside those discussed in this course. 
# MAGIC
# MAGIC #### AI Search
# MAGIC Databricks AI Search is a vector search solution that is built into the platform and integrated with its governance and productivity tools. Vector search is a type of search optimized for retrieving embeddings.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Learn More</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC You can read more about AI Search here (<a href="https://docs.databricks.com/aws/en/ai-search/ai-search" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/ai-search/ai-search" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/ai-search/ai-search" style="color: #1976d2; text-decoration: underline;">GCP</a>).
# MAGIC </p>
# MAGIC </div>
# MAGIC
# MAGIC #### Common Tool Patterns
# MAGIC Below is a summary of some common tool patterns, as well as additional links for reading, that exist on Databricks today. 
# MAGIC | Tool pattern| Description|
# MAGIC |-------------|------------|
# MAGIC | **Structured data retrieval tools ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/structured-retrieval-tools) \| [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/structured-retrieval-tools) \| [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/structured-retrieval-tools))** | Query SQL tables, databases, and structured data sources.                                   |
# MAGIC | **Unstructured data retrieval tools ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/unstructured-retrieval-tools) \| [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools) \| [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/unstructured-retrieval-tools))** | Search document collections and perform retrieval-augmented generation.                    |
# MAGIC | **Code interpreter tools ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/code-interpreter-tools) \| [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/code-interpreter-tools) \| [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/code-interpreter-tools))**         | Allow agents to run Python code for calculations, data analysis, and dynamic processing.   |
# MAGIC | **External connection tools ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/external-connection-tools) \| [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/external-connection-tools) \| [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/external-connection-tools))**      | Connect to external services and APIs such as Slack.                                        |

# COMMAND ----------

# DBTITLE 1,C2 - Unity Gateway Spend Control
# MAGIC %md-sandbox
# MAGIC ## C. Governance and AI Management with Unity Gateway
# MAGIC <br>
# MAGIC <div style="max-width: 720px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 700 500" xmlns:xlink="http://www.w3.org/1999/xlink" style="font-family: sans-serif;">
# MAGIC   <defs>
# MAGIC     <marker id="b4arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#618794" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC     <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
# MAGIC       <feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="#1B3139" flood-opacity="0.16"/>
# MAGIC     </filter>
# MAGIC   </defs>
# MAGIC     <rect x="260" y="10" width="180" height="52" rx="12" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/>
# MAGIC   <text x="350" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Engineering Budget</text>
# MAGIC   <text x="350" y="50" text-anchor="middle" font-size="11" font-style="italic" fill="#5A6F77">$50,000 per month</text>
# MAGIC   <path d="M350 62 L350 92" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <rect x="30" y="96" width="640" height="152" rx="16" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.5" filter="url(#soft)"/>
# MAGIC   <text x="368" y="130" text-anchor="middle" font-size="18" font-weight="800" fill="#0b2026">Unity Gateway</text>
# MAGIC   <rect x="43" y="150" width="164" height="32" rx="16" fill="#FF5F46"/>
# MAGIC   <text x="125" y="171" text-anchor="middle" font-size="12.5" font-weight="600" fill="#ffffff">Spend tracking</text>
# MAGIC   <rect x="268" y="150" width="164" height="32" rx="16" fill="#FF5F46"/>
# MAGIC   <text x="350" y="171" text-anchor="middle" font-size="12.5" font-weight="600" fill="#ffffff">Budget policies</text>
# MAGIC   <rect x="493" y="150" width="164" height="32" rx="16" fill="#FF5F46"/>
# MAGIC   <text x="575" y="171" text-anchor="middle" font-size="12.5" font-weight="600" fill="#ffffff">Hard spend caps</text>
# MAGIC   <rect x="43" y="194" width="164" height="32" rx="16" fill="#FF5F46"/>
# MAGIC   <text x="125" y="215" text-anchor="middle" font-size="12.5" font-weight="600" fill="#ffffff">Rate limits</text>
# MAGIC   <rect x="268" y="194" width="164" height="32" rx="16" fill="#FF5F46"/>
# MAGIC   <text x="350" y="215" text-anchor="middle" font-size="12.5" font-weight="600" fill="#ffffff">Chargeback</text>
# MAGIC   <rect x="493" y="194" width="164" height="32" rx="16" fill="#FF5F46"/>
# MAGIC   <text x="575" y="215" text-anchor="middle" font-size="12.5" font-weight="600" fill="#ffffff">Usage attribution</text>
# MAGIC   <path d="M125 248 L125 286" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <path d="M350 248 L350 286" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <path d="M575 248 L575 286" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <rect x="40" y="290" width="170" height="66" rx="12" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/>
# MAGIC   <foreignObject x="75" y="304" width="100" height="22"><img xmlns="http://www.w3.org/1999/xhtml" src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/icons/cursor.png" style="width:100px;height:22px;object-fit:contain;display:block;"/></foreignObject>
# MAGIC   <text x="125" y="344" text-anchor="middle" font-size="11" fill="#5A6F77"><tspan fill="#00A972" font-weight="700">$5.5k</tspan> of $10k used</text>
# MAGIC   <rect x="265" y="290" width="170" height="66" rx="12" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/>
# MAGIC   <text x="350" y="322" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Omnigent</text>
# MAGIC   <text x="350" y="344" text-anchor="middle" font-size="11" fill="#5A6F77"><tspan fill="#E5A000" font-weight="700">$22k</tspan> of $25k used</text>
# MAGIC   <rect x="490" y="290" width="170" height="66" rx="12" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/>
# MAGIC   <text x="575" y="322" text-anchor="middle" font-size="13" font-weight="700" fill="#0b2026">Customer support</text>
# MAGIC   <text x="575" y="344" text-anchor="middle" font-size="11" fill="#5A6F77"><tspan fill="#FF3621" font-weight="700">$15k</tspan> of $15k used</text>
# MAGIC   <path d="M85 356 L85 412" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <circle cx="85" cy="438" r="24" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/><foreignObject x="69" y="422" width="32" height="32"><img xmlns="http://www.w3.org/1999/xhtml" src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/icons/openai.png" style="width:32px;height:32px;object-fit:contain;display:block;"/></foreignObject>
# MAGIC   <path d="M165 356 L165 412" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <circle cx="165" cy="438" r="24" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/><foreignObject x="149" y="422" width="32" height="32"><img xmlns="http://www.w3.org/1999/xhtml" src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/icons/claude-anthropic.png" style="width:32px;height:32px;object-fit:contain;display:block;"/></foreignObject>
# MAGIC   <path d="M310 356 L310 412" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <circle cx="310" cy="438" r="24" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/><foreignObject x="294" y="422" width="32" height="32"><img xmlns="http://www.w3.org/1999/xhtml" src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/icons/gemini.png" style="width:32px;height:32px;object-fit:contain;display:block;"/></foreignObject>
# MAGIC   <path d="M390 356 L390 412" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <circle cx="390" cy="438" r="24" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/><foreignObject x="374" y="422" width="32" height="32"><img xmlns="http://www.w3.org/1999/xhtml" src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/icons/genai-brain-icon.png" style="width:32px;height:32px;object-fit:contain;display:block;"/></foreignObject>
# MAGIC   <path d="M535 356 L535 412" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <g stroke="#FF3621" stroke-width="2.6" stroke-linecap="round"><line x1="529" y1="378" x2="541" y2="390"/><line x1="529" y1="390" x2="541" y2="378"/></g>
# MAGIC   <circle cx="535" cy="438" r="24" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/><foreignObject x="519" y="422" width="32" height="32"><img xmlns="http://www.w3.org/1999/xhtml" src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/icons/antigravity.png" style="width:32px;height:32px;object-fit:contain;display:block;"/></foreignObject>
# MAGIC   <path d="M615 356 L615 412" fill="none" stroke="#618794" stroke-width="1.6" stroke-dasharray="5 5" marker-end="url(#b4arrow)"/>
# MAGIC   <g stroke="#FF3621" stroke-width="2.6" stroke-linecap="round"><line x1="609" y1="378" x2="621" y2="390"/><line x1="609" y1="390" x2="621" y2="378"/></g>
# MAGIC   <circle cx="615" cy="438" r="24" fill="#ffffff" stroke="#C6DCEA" stroke-width="1.2" filter="url(#soft)"/><clipPath id="gclip615"><rect x="600" y="423" width="30" height="30" rx="4"/></clipPath><foreignObject x="600" y="423" width="80" height="30" clip-path="url(#gclip615)"><img xmlns="http://www.w3.org/1999/xhtml" src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/icons/databricks-genie.png" style="width:80px;height:30px;object-fit:cover;display:block;"/></foreignObject>
# MAGIC   <text x="575" y="487" text-anchor="middle" font-size="13.5" font-style="italic" font-weight="700" fill="#FF3621">Blocked!</text>
# MAGIC
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC What does Unity Gateway provide?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p style="margin-top: 10px;"><a href="https://www.databricks.com/blog/ai-gateway-governance-layer-agentic-ai" style="color: #1976d2; text-decoration: underline;">Unity Gateway</a> is the Databricks central AI governance layer for LLM endpoints (<a href="https://docs.databricks.com/aws/en/ai-gateway/#llms" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/#llms" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/ai-gateway/#llms" style="color: #1976d2; text-decoration: underline;">GCP</a>), MCP servers (<a href="https://docs.databricks.com/aws/en/ai-gateway/#mcps" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/#mcps" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/ai-gateway/#mcps" style="color: #1976d2; text-decoration: underline;">GCP</a>), and coding agents. Use Unity Gateway to analyze usage, configure permissions, enforce guardrails, and manage capacity across providers.</p>
# MAGIC
# MAGIC <ul style="margin: 12px 0; padding-left: 24px;">
# MAGIC <li><strong>Governance</strong>: Track which users, agents, and applications are making LLM calls, how agents use MCP servers and APIs, and applies consistent policies across models and tools
# MAGIC <ul style="margin: 4px 0; padding-left: 20px;"><li><strong>MCP Governance</strong>: On-behalf-of execution so agents act with the request user's exact permissions instead of a shared service account.</li></ul></li>
# MAGIC <li><strong>Rate limiting</strong>: Protect endpoints from overload with per-endpoint and per-identity (user or group) QPM/TPM limits (<a href="https://docs.databricks.com/aws/en/ai-gateway/rate-limits-beta" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/rate-limits-beta" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/ai-gateway/rate-limits-beta" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC <li><strong>Guardrails</strong>: Apply input/output safety filters and data-protection policies to block or mask harmful or sensitive content</li>
# MAGIC <li><strong>Usage tracking</strong>: Monitor token consumption and costs across Unity Gateway endpoints and coding agents using system tables</li>
# MAGIC <li><strong>Payload logging</strong>: Log request/response payloads to Unity Catalog Delta tables via inference tables for audit, debugging, and model improvement</li>
# MAGIC </ul>
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# DBTITLE 1,C1 - Fallback Example
# MAGIC %md-sandbox
# MAGIC ### C1. Fallback Example
# MAGIC <p>Without Unity Gateway, each agent or application would have to implement its own access control, guardrails, and logging, making safety, compliance, debugging, and cost control much harder to do reliably at scale (especially across multiple models and providers). The diagram below illustrates how Unity Gateway handles endpoint fallback routing.</p>
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 680 290" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Unity Gateway Endpoint fallback routing</title>
# MAGIC   <desc>Requests route from a client to Model 1, falling back to Model 2 or Model 3 on 429/5xx errors, with all outcomes logged to inference and usage tracking tables.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="fb-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#00A972" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC     <marker id="fb-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#FF3621" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC     <marker id="fb-gray" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#618794" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC
# MAGIC   <!-- Gateway container -->
# MAGIC   <rect x="120" y="20" width="540" height="258" rx="16" fill="none" stroke="#618794" stroke-width="1.5" stroke-dasharray="7 4" stroke-opacity="0.4"/>
# MAGIC   <text x="390" y="46" text-anchor="middle" font-size="13" fill="#618794">Unity Gateway Endpoint</text>
# MAGIC
# MAGIC   <!-- Log box -->
# MAGIC   <rect x="275" y="60" width="230" height="56" rx="8" fill="#EEEDE9" stroke="#00A972" stroke-width="1.5"/>
# MAGIC   <text x="390" y="84" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Request logged</text>
# MAGIC   <text x="390" y="102" text-anchor="middle" font-size="11" fill="#618794">Inference &amp; usage tracking table</text>
# MAGIC
# MAGIC   <!-- Routing group -->
# MAGIC   <rect x="135" y="146" width="520" height="102" rx="12" fill="#2272B4" fill-opacity="0.08" stroke="#2272B4" stroke-width="1.5" stroke-dasharray="6 3"/>
# MAGIC   <text x="649" y="162" text-anchor="end" font-size="11" font-style="italic" fill="#2272B4">Routing targets</text>
# MAGIC
# MAGIC   <!-- Model 1 -->
# MAGIC   <rect x="148" y="158" width="132" height="78" rx="8" fill="#ffffff" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="214" y="184" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Model 1</text>
# MAGIC   <rect x="155" y="193" width="118" height="18" rx="4" fill="#FFAB00"/>
# MAGIC   <text x="214" y="202" text-anchor="middle" font-size="10" font-weight="600" fill="#0b2026" dominant-baseline="central">Original request</text>
# MAGIC   <text x="214" y="224" text-anchor="middle" font-size="11" fill="#618794">Primary route</text>
# MAGIC
# MAGIC   <!-- Model 2 -->
# MAGIC   <rect x="340" y="178" width="130" height="56" rx="8" fill="#ffffff" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="405" y="202" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Model 2</text>
# MAGIC   <text x="405" y="221" text-anchor="middle" font-size="11" fill="#618794">First fallback</text>
# MAGIC
# MAGIC   <!-- Model 3 -->
# MAGIC   <rect x="530" y="178" width="110" height="56" rx="8" fill="#ffffff" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="585" y="202" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Model 3</text>
# MAGIC   <text x="585" y="221" text-anchor="middle" font-size="11" fill="#618794">Last fallback</text>
# MAGIC
# MAGIC   <!-- Green: Model 1 to Log -->
# MAGIC   <path d="M214 158 L310 116" fill="none" stroke="#00A972" stroke-width="1.8" marker-end="url(#fb-green)"/>
# MAGIC   <text x="218" y="131" text-anchor="end" font-size="11" fill="#00A972">200 (success)</text>
# MAGIC
# MAGIC   <!-- Red: Model 1 to Model 2 -->
# MAGIC   <path d="M280 205 L340 205" fill="none" stroke="#FF3621" stroke-width="1.8" marker-end="url(#fb-red)"/>
# MAGIC   <text x="310" y="198" text-anchor="middle" font-size="11" fill="#FF3621">429/5xx</text>
# MAGIC
# MAGIC   <!-- Green: Model 2 to Log -->
# MAGIC   <path d="M405 178 L390 116" fill="none" stroke="#00A972" stroke-width="1.8" marker-end="url(#fb-green)"/>
# MAGIC   <text x="406" y="131" text-anchor="start" font-size="11" fill="#00A972">200</text>
# MAGIC
# MAGIC   <!-- Red: Model 2 to Model 3 -->
# MAGIC   <path d="M470 205 L530 205" fill="none" stroke="#FF3621" stroke-width="1.8" stroke-dasharray="6 3" marker-end="url(#fb-red)"/>
# MAGIC   <text x="500" y="198" text-anchor="middle" font-size="11" fill="#FF3621">429/5xx</text>
# MAGIC
# MAGIC   <!-- Green: Model 3 to Log -->
# MAGIC   <path d="M585 178 L470 116" fill="none" stroke="#00A972" stroke-width="1.8" marker-end="url(#fb-green)"/>
# MAGIC   <text x="505" y="131" text-anchor="start" font-size="11" fill="#00A972">200</text>
# MAGIC
# MAGIC   <!-- Client -->
# MAGIC   <rect x="28" y="186" width="64" height="38" rx="8" fill="#ffffff" stroke="#618794" stroke-width="1.5"/>
# MAGIC   <text x="60" y="205" text-anchor="middle" font-size="13" fill="#0b2026" dominant-baseline="central">Client</text>
# MAGIC   <path d="M92 205 L148 205" fill="none" stroke="#618794" stroke-width="1.8" marker-end="url(#fb-gray)"/>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Learn More</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Databricks Foundation Model APIs integrate with Unity Gateway, but you enable Unity Gateway features (like usage tracking and inference tables) per endpoint. In the new Unity Gateway (Beta), usage tracking is on by default for Gateway endpoints, once the feature is enabled and the endpoint is created.
# MAGIC See: Unity Gateway documentation (<a href="https://docs.databricks.com/aws/en/ai-gateway/overview-beta" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/overview-beta" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/ai-gateway/overview-beta" style="color: #1976d2; text-decoration: underline;">GCP</a>)
# MAGIC </p>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# DBTITLE 1,C2. Workspace-Scoped Endpoints and UC-Native Services for LLM Use-Cases
# MAGIC %md-sandbox
# MAGIC ### C2. Workspace-Scoped Endpoints and UC-Native Services for LLM Use-Cases
# MAGIC
# MAGIC <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap; justify-content: center; margin: 16px 0; font-family: sans-serif;">
# MAGIC   <div style="border: 1.5px solid #618794; border-radius: 12px; padding: 20px 24px; width: 260px; background: #ffffff;">
# MAGIC     <div style="font-size: 16pt; font-weight: 700; color: #0b2026; text-align: center;">Legacy Model Serving Endpoints</div>
# MAGIC     <div style="font-size: 14pt; color: #618794; text-align: center; margin-top: 6px;">Workspace-scoped AI Gateway</div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 14px; font-size: 14pt; color: #0b2026;">Endpoint in Workspace A</div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 4px; font-size: 14pt; color: #0b2026;">Duplicate in Workspace B</div>
# MAGIC     <div style="border: 1px dashed #2272B4; border-radius: 6px; background: rgba(34,114,180,0.08); padding: 4px 10px; text-align: center; margin-top: 4px; font-size: 14pt; color: #0b2026;">Per-workspace permissions</div>
# MAGIC   </div>
# MAGIC   <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; min-width: 140px;">
# MAGIC     <div style="font-size: 14pt; color: #618794; text-align: center; line-height: 1.4;">Define once,<br/>govern centrally</div>
# MAGIC     <svg width="120" height="20" viewBox="0 0 120 20" style="display: block;">
# MAGIC       <defs><marker id="cmp-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
# MAGIC       <path d="M5 10 L110 10" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#cmp-arrow)"/>
# MAGIC     </svg>
# MAGIC   </div>
# MAGIC   <div style="border: 2px solid #FF5F46; border-radius: 12px; padding: 20px 24px; width: 280px; background: #F9F7F4;">
# MAGIC     <div style="font-size: 16pt; font-weight: 700; color: #FF5F46; text-align: center;">Model Services</div>
# MAGIC     <div style="font-size: 14pt; color: #618794; text-align: center; margin-top: 4px;">Unity Catalog securable object</div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 10px; font-size: 14pt; color: #0b2026;">Defined once in catalog.schema</div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 4px; font-size: 14pt; color: #0b2026;">Shared across the metastore</div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 4px; font-size: 14pt; color: #0b2026;">Discoverable in Catalog Explorer</div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 4px; font-size: 14pt; color: #0b2026;">Unity Gateway controls</div>
# MAGIC     <div style="border: 1px dashed #2272B4; border-radius: 6px; background: rgba(34,114,180,0.08); padding: 4px 10px; text-align: center; margin-top: 6px; font-size: 14pt; color: #0b2026;">Centralized UC privileges</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC When and why model services are recommended
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p>Model serving endpoints are not legacy for all AI workloads. However, for most governed LLM use cases, Databricks recommends moving to model services because model services shift LLM access from workspace-scoped endpoints into Unity Catalog, where the service can be defined once, shared across workspaces, permissioned centrally, and managed through Unity Gateway controls.</p>
# MAGIC
# MAGIC <ul style="margin: 12px 0; padding-left: 24px;">
# MAGIC <li>Standardize approved models across multiple workspaces</li>
# MAGIC <li>Centralize access control instead of managing permissions workspace by workspace</li>
# MAGIC <li>Apply governance controls such as rate limits, service policies, and guardrails consistently</li>
# MAGIC <li>Track usage, token consumption, and costs in a system-level way</li>
# MAGIC </ul>
# MAGIC
# MAGIC </div>
# MAGIC </details>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>