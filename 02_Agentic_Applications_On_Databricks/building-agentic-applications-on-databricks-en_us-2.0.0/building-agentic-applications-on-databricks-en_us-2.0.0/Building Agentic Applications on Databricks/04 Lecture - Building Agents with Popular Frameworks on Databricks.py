# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Lecture - Building Agents with Popular Frameworks and MLflow
# MAGIC %md
# MAGIC # Lecture - Building Agents with Popular Frameworks on Databricks
# MAGIC #### Duration: 20 minutes
# MAGIC ## Overview
# MAGIC Now that you know how to create and govern agent tools, this lecture surveys the most popular frameworks for building agents on Databricks. The discussion will be facilitated with a focus on the **OpenAI Agents SDK**. We cover its orchestration patterns, rapid tool prototyping across frameworks, MCP integration, and multi-agent concepts before briefly comparing other supported frameworks (LangChain, LangGraph, DSPy). This lecture closes with how MLflow provides observability across single-agent and multi-agent flows.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lecture, you will be able to:
# MAGIC 1. Identify the orchestration patterns supported by the OpenAI Agents SDK (handoffs, agents-as-tools, sequential, parallel, feedback loops)
# MAGIC 2. Prototype agent tools with `@function_tool` (OpenAI Agents SDK), `@tool` (LangChain), and plain functions (DSPy) before registering them to Unity Catalog
# MAGIC 3. Explain how `McpServer` connects agents to Unity Catalog functions via MCP
# MAGIC 4. Understand multi-agent concepts: supervisor agents, handoffs, and worker specialization
# MAGIC 5. Compare other supported frameworks (LangChain, LangGraph, DSPy) and when you might choose them
# MAGIC 6. Explain how MLflow tracing captures agent execution as hierarchical spans

# COMMAND ----------

# DBTITLE 1,A. OpenAI Agent Orchestration Patterns
# MAGIC %md
# MAGIC ## A. OpenAI Agent Orchestration Patterns
# MAGIC
# MAGIC [The OpenAI Agents SDK](https://openai.github.io/openai-agents-python/multi_agent/) supports multiple patterns for coordinating agents. These fall into two categories: **LLM-driven** (the model decides what happens next) and **code-driven** (your code controls the flow deterministically).

# COMMAND ----------

# DBTITLE 1,A1. LLM-Driven Orchestration
# MAGIC %md-sandbox
# MAGIC ### A1. LLM-Driven Orchestration
# MAGIC In this framework, an **agent is an LLM equipped with instructions, tools, and handoffs** that can autonomously determine how to complete an open-ended task.
# MAGIC
# MAGIC Agents can use tools to gather information and take actions, while **handoffs** allow them to delegate specific tasks to specialized sub-agents. For example, a research agent might use:
# MAGIC
# MAGIC * **Web search** to find information online
# MAGIC * **File search** to retrieve information from proprietary or connected data sources
# MAGIC * **Computer use** to interact with applications
# MAGIC * **Code execution** to analyze data
# MAGIC * **Handoffs** to delegate tasks to specialized agents for planning, writing, and other capabilities
# MAGIC
# MAGIC
# MAGIC **Click on each pattern** to see how it works.
# MAGIC
# MAGIC <style>
# MAGIC .orch-outer { max-width: 1100px; margin: 0 auto; font-family: sans-serif; font-size: 12pt; }
# MAGIC .orch-label { font-size: 12pt; font-weight: 600; color: #999; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 8px; }
# MAGIC .orch-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
# MAGIC .orch-node {
# MAGIC   flex: 1; min-width: 140px; background: #fff; border: 1.5px solid #C6DCEA; border-top: 5px solid transparent;
# MAGIC   border-radius: 14px; padding: 14px 12px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, box-shadow 0.15s;
# MAGIC   box-shadow: 0 2px 8px rgba(27,49,57,.07);
# MAGIC }
# MAGIC .orch-node:hover { transform: translateY(-2px); }
# MAGIC .orch-node.active { border-color: var(--oc); box-shadow: 0 4px 14px rgba(27,49,57,.14); }
# MAGIC .orch-node-title { display: block; font-size: 12pt; font-weight: 700; color: #0b2026; line-height: 1.3; pointer-events: none; }
# MAGIC .orch-node-sub { display: block; font-size: 12pt; color: #666; margin-top: 4px; pointer-events: none; }
# MAGIC .orch-detail-wrap { overflow: hidden; max-height: 0; opacity: 0; transition: max-height 0.35s ease, opacity 0.28s ease; }
# MAGIC .orch-detail-wrap.open { max-height: 1600px; opacity: 1; margin-top: 12px; }
# MAGIC .orch-detail-card { background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07); font-size: 12pt; line-height: 1.7; color: #333; }
# MAGIC .orch-detail-card strong { color: #0b2026; }
# MAGIC .orch-detail-card code { background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-size: 12pt; }
# MAGIC .orch-flow { display: flex; align-items: center; justify-content: center; gap: 0; margin: 14px 0; flex-wrap: wrap; }
# MAGIC .orch-flow-box { background: #1B3139; color: #fff; border-radius: 6px; padding: 8px 14px; font-size: 12pt; font-weight: 600; text-align: center; min-width: 90px; }
# MAGIC .orch-flow-arrow { color: #999; font-size: 20px; padding: 0 6px; }
# MAGIC .py-card { border:1px solid #DCE0E2; border-radius:8px; overflow:hidden; background:#F9F7F4; margin:12px 0; }
# MAGIC .py-card-hdr { display:flex; align-items:center; padding:0.45rem 0.75rem; background:#EEEDE9; border-bottom:1px solid #DCE0E2; }
# MAGIC .py-card-lang { display:inline-flex; align-items:center; gap:0.45rem; }
# MAGIC .py-card-lang span { font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#5A6F77; }
# MAGIC .py-card-body { margin:0; padding:1rem; font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace; font-size:0.82rem; line-height:1.55; color:#1B3139; white-space:pre-wrap; word-break:break-word; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="orch-outer" id="orch-a1">
# MAGIC
# MAGIC <div class="orch-row">
# MAGIC   <div class="orch-node" data-orch="handoffs" onclick="orchSelectA1('handoffs')" style="--oc:#FF5F46; border-top-color:#FF5F46;">
# MAGIC     <span class="orch-node-title">Handoffs</span>
# MAGIC     <span class="orch-node-sub">Specialist takes over</span>
# MAGIC   </div>
# MAGIC   <div class="orch-node" data-orch="tools" onclick="orchSelectA1('tools')" style="--oc:#4299E0; border-top-color:#4299E0;">
# MAGIC     <span class="orch-node-title">Agents as Tools</span>
# MAGIC     <span class="orch-node-sub">Manager calls specialists</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="orch-detail-wrap" id="orch-detail-wrap-a1">
# MAGIC   <div class="orch-detail-card" id="orch-detail-card-a1"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var ORCH_A1 = {
# MAGIC   handoffs: {
# MAGIC     color: '#FF5F46',
# MAGIC     title: 'Handoffs',
# MAGIC     html: '<strong>Pattern:</strong> A triage/supervisor agent routes the conversation to a specialist, which then <strong>becomes the active agent</strong> and responds directly to the user.<br/><br/><strong>Key behavior:</strong> Conversation ownership transfers. The specialist takes over; the supervisor does not post-process.<br/><br/><div class="orch-flow"><div class="orch-flow-box" style="background:transparent;color:#0b2026;border:2px solid #0b2026;">User</div><div class="orch-flow-arrow">&#x2192;</div><div class="orch-flow-box" style="background:#FF5F46;">Supervisor</div><div class="orch-flow-arrow">&#x2192;</div><div style="display:flex;flex-direction:column;gap:4px;"><div class="orch-flow-box" style="background:#2574B5;">Specialist A</div><div class="orch-flow-box" style="background:transparent;color:#2574B5;border:2px solid #2574B5;">Specialist B</div><div class="orch-flow-box" style="background:transparent;color:#2574B5;border:2px solid #2574B5;">Specialist C</div></div><div class="orch-flow-arrow">&#x2192;</div><div class="orch-flow-box" style="background:transparent;color:#0b2026;border:2px solid #0b2026;">User</div></div><strong>When to use:</strong> The specialist should respond directly, prompts need to stay focused, or you want to swap instructions and tool sets per route.<br/><br/><div class="py-card"><div class="py-card-hdr"><div class="py-card-lang"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span>Python</span></div></div><div class="py-card-body">Agent(handoffs=[agent_a, agent_b])</div></div>'
# MAGIC   },
# MAGIC   tools: {
# MAGIC     color: '#4299E0',
# MAGIC     title: 'Agents as Tools',
# MAGIC     html: '<strong>Pattern:</strong> A manager agent maintains conversation control and invokes specialist agents as tools via <code>Agent.as_tool()</code>. The manager <strong>synthesizes the results</strong> from multiple specialists into a final answer.<br/><br/><strong>Key behavior:</strong> The manager always owns the conversation. Specialists are called and their outputs are combined.<br/><br/><div class="orch-flow"><div class="orch-flow-box" style="background:transparent;color:#0b2026;border:2px solid #0b2026;">User</div><div class="orch-flow-arrow">&#x2192;</div><div class="orch-flow-box" style="background:#4299E0;">Manager</div><div class="orch-flow-arrow">&#x2194;</div><div style="display:flex;flex-direction:column;gap:4px;"><div class="orch-flow-box" style="background:#00A972;">Tool Agent A</div><div class="orch-flow-box" style="background:#FFAB00;">Tool Agent B</div></div><div class="orch-flow-arrow">&#x2192;</div><div class="orch-flow-box" style="background:transparent;color:#0b2026;border:2px solid #0b2026;">User</div></div><strong>When to use:</strong> You want one agent to own the final answer, combine outputs from multiple specialists, or maintain a single conversational thread.<br/><br/><div class="py-card"><div class="py-card-hdr"><div class="py-card-lang"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span>Python</span></div></div><div class="py-card-body">Agent(tools=[specialist.as_tool()])</div></div>'
# MAGIC   }
# MAGIC };
# MAGIC var orchCurrentA1 = null;
# MAGIC function orchSelectA1(id) {
# MAGIC   var wrap = document.getElementById('orch-detail-wrap-a1');
# MAGIC   var card = document.getElementById('orch-detail-card-a1');
# MAGIC   document.querySelectorAll('#orch-a1 .orch-node').forEach(function(n) { n.classList.toggle('active', n.dataset.orch === id); });
# MAGIC   if (orchCurrentA1 === id) { wrap.classList.remove('open'); document.querySelectorAll('#orch-a1 .orch-node').forEach(function(n) { n.classList.remove('active'); }); orchCurrentA1 = null; return; }
# MAGIC   orchCurrentA1 = id;
# MAGIC   var d = ORCH_A1[id];
# MAGIC   card.innerHTML = '<div style="height:5px;border-radius:5px;margin:-4px 0 12px;background:' + d.color + ';"></div><div style="font-size:16pt;font-weight:700;margin-bottom:12px;color:' + d.color + ';">' + d.title + '</div><div style="font-size:15pt;color:#0b2026;line-height:1.7;">' + d.html + '</div>';
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# DBTITLE 1,A2. Code-Driven Orchestration
# MAGIC %md-sandbox
# MAGIC ### A2. Code-Driven Orchestration
# MAGIC While LLM-based orchestration offers flexibility, **code-based orchestration provides greater predictability and control** over speed, cost, and performance. Common patterns include:
# MAGIC
# MAGIC - **Structured outputs:** Have an agent return structured data that code can evaluate and use to determine the next step, such as routing a task to the appropriate agent based on its classification.
# MAGIC - **Agent chaining:** Pass one agent’s output to the next, breaking complex tasks into predictable steps such as research → outline → draft → critique → revise.
# MAGIC - **Evaluation loops:** Repeatedly generate and evaluate an output until it meets defined criteria.
# MAGIC - **Parallel execution:** Run independent agents simultaneously to improve speed and efficiency.
# MAGIC
# MAGIC **Click on each pattern** to see how it works.
# MAGIC
# MAGIC <style>
# MAGIC .orch-outer { max-width: 1100px; margin: 0 auto; font-family: sans-serif; font-size: 12pt; }
# MAGIC .orch-label { font-size: 12pt; font-weight: 600; color: #999; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 8px; }
# MAGIC .orch-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
# MAGIC .orch-node {
# MAGIC   flex: 1; min-width: 140px; background: #fff; border: 1.5px solid #C6DCEA; border-top: 5px solid transparent;
# MAGIC   border-radius: 14px; padding: 14px 12px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, box-shadow 0.15s;
# MAGIC   box-shadow: 0 2px 8px rgba(27,49,57,.07);
# MAGIC }
# MAGIC .orch-node:hover { transform: translateY(-2px); }
# MAGIC .orch-node.active { border-color: var(--oc); box-shadow: 0 4px 14px rgba(27,49,57,.14); }
# MAGIC .orch-node-title { display: block; font-size: 12pt; font-weight: 700; color: #0b2026; line-height: 1.3; pointer-events: none; }
# MAGIC .orch-node-sub { display: block; font-size: 12pt; color: #666; margin-top: 4px; pointer-events: none; }
# MAGIC .orch-detail-wrap { overflow: hidden; max-height: 0; opacity: 0; transition: max-height 0.35s ease, opacity 0.28s ease; }
# MAGIC .orch-detail-wrap.open { max-height: 1600px; opacity: 1; margin-top: 12px; }
# MAGIC .orch-detail-card { background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07); font-size: 12pt; line-height: 1.7; color: #333; }
# MAGIC .orch-detail-card strong { color: #0b2026; }
# MAGIC .orch-detail-card code { background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-size: 12pt; }
# MAGIC .orch-flow { display: flex; align-items: center; justify-content: center; gap: 0; margin: 14px 0; flex-wrap: wrap; }
# MAGIC .orch-flow-box { background: #1B3139; color: #fff; border-radius: 6px; padding: 8px 14px; font-size: 12pt; font-weight: 600; text-align: center; min-width: 90px; }
# MAGIC .orch-flow-arrow { color: #999; font-size: 20px; padding: 0 6px; }
# MAGIC .py-card { border:1px solid #DCE0E2; border-radius:8px; overflow:hidden; background:#F9F7F4; margin:12px 0; }
# MAGIC .py-card-hdr { display:flex; align-items:center; padding:0.45rem 0.75rem; background:#EEEDE9; border-bottom:1px solid #DCE0E2; }
# MAGIC .py-card-lang { display:inline-flex; align-items:center; gap:0.45rem; }
# MAGIC .py-card-lang span { font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#5A6F77; }
# MAGIC .py-card-body { margin:0; padding:1rem; font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace; font-size:0.82rem; line-height:1.55; color:#1B3139; white-space:pre-wrap; word-break:break-word; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="orch-outer">
# MAGIC
# MAGIC <div class="orch-row">
# MAGIC   <div class="orch-node" data-orch="sequential" onclick="orchSelect('sequential')" style="--oc:#00A972; border-top-color:#00A972;">
# MAGIC     <span class="orch-node-title">Sequential Chaining</span>
# MAGIC     <span class="orch-node-sub">Pipeline of agents</span>
# MAGIC   </div>
# MAGIC   <div class="orch-node" data-orch="parallel" onclick="orchSelect('parallel')" style="--oc:#FFAB00; border-top-color:#FFAB00;">
# MAGIC     <span class="orch-node-title">Parallel Execution</span>
# MAGIC     <span class="orch-node-sub">Concurrent agents</span>
# MAGIC   </div>
# MAGIC   <div class="orch-node" data-orch="feedback" onclick="orchSelect('feedback')" style="--oc:#9B1D20; border-top-color:#9B1D20;">
# MAGIC     <span class="orch-node-title">Feedback Loops</span>
# MAGIC     <span class="orch-node-sub">Iterate until quality</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="orch-detail-wrap" id="orch-detail-wrap">
# MAGIC   <div class="orch-detail-card" id="orch-detail-card"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var ORCH = {
# MAGIC   sequential: {
# MAGIC     color: '#00A972',
# MAGIC     title: 'Sequential Chaining',
# MAGIC     html: '<strong>Pattern:</strong> Your code pipes the output of one agent as the input to the next. Each agent handles one step in a multi-step pipeline.<br/><br/><strong>Key behavior:</strong> Deterministic order. Each agent runs exactly once in sequence.<br/><br/><div class="orch-flow"><div class="orch-flow-box" style="background:#00A972;">Research</div><div class="orch-flow-arrow">&#x2192;</div><div class="orch-flow-box" style="background:#00A972;">Outline</div><div class="orch-flow-arrow">&#x2192;</div><div class="orch-flow-box" style="background:#00A972;">Draft</div><div class="orch-flow-arrow">&#x2192;</div><div class="orch-flow-box" style="background:#00A972;">Review</div></div><strong>When to use:</strong> Tasks decompose naturally into discrete steps, each requiring different instructions or tools. You want full control over the order of execution.<br/><br/><div class="py-card"><div class="py-card-hdr"><div class="py-card-lang"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span>Python</span></div></div><div class="py-card-body">result1 = await Runner.run(agent1, input)\nresult2 = await Runner.run(agent2, result1.to_input_list())</div></div><br/><br/>Use <code>result1.to_input_list()</code> rather than <code>result1.final_output</code> to carry the full conversation history forward, not just the final text string.'
# MAGIC   },
# MAGIC   parallel: {
# MAGIC     color: '#FFAB00',
# MAGIC     title: 'Parallel Execution',
# MAGIC     html: '<strong>Pattern:</strong> Run multiple independent agents simultaneously using <code>asyncio.gather</code>. Results are collected and combined after all agents finish.<br/><br/><strong>Key behavior:</strong> Concurrent execution. Faster than sequential when tasks are independent.<br/><br/><div class="orch-flow"><div class="orch-flow-box" style="background:transparent;color:#0b2026;border:2px solid #0b2026;">Input</div><div class="orch-flow-arrow">&#x2192;</div><div style="display:flex;flex-direction:column;gap:4px;"><div class="orch-flow-box" style="background:#FFAB00;">Agent A</div><div class="orch-flow-box" style="background:#FFAB00;">Agent B</div><div class="orch-flow-box" style="background:#FFAB00;">Agent C</div></div><div class="orch-flow-arrow">&#x2192;</div><div class="orch-flow-box" style="background:transparent;color:#0b2026;border:2px solid #0b2026;">Combine</div></div><strong>When to use:</strong> Multiple independent analyses of the same input, or different perspectives that can run concurrently without depending on each other.<br/><br/><div class="py-card"><div class="py-card-hdr"><div class="py-card-lang"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span>Python</span></div></div><div class="py-card-body">results = await asyncio.gather(\n    Runner.run(a1, q),\n    Runner.run(a2, q),\n    Runner.run(a3, q)\n)</div></div>'
# MAGIC   },
# MAGIC   feedback: {
# MAGIC     color: '#9B1D20',
# MAGIC     title: 'Feedback Loops',
# MAGIC     html: '<strong>Pattern:</strong> An agent runs in a loop, paired with an evaluator agent that provides feedback. The evaluator feedback is passed back to the generator each iteration. The loop continues until the evaluator approves the output or a max iteration count is reached.<br/><br/><strong>Key behavior:</strong> Iterative refinement. Quality improves with each pass because the generator receives explicit feedback, not just a retry signal.<br/><br/><div class="orch-flow"><div class="orch-flow-box" style="background:#9B1D20;">Generator</div><div class="orch-flow-arrow">&#x2192;</div><div class="orch-flow-box" style="background:#9B1D20;">Evaluator</div><div class="orch-flow-arrow">&#x21BA;</div><div class="orch-flow-box" style="background:transparent;color:#0b2026;border:2px solid #0b2026;">Pass?</div></div><strong>When to use:</strong> Tasks where quality can be measured programmatically or by an LLM judge, and iterative improvement is worth the extra cost (e.g., code generation, writing).<br/><br/><div class="py-card"><div class="py-card-hdr"><div class="py-card-lang"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span>Python</span></div></div><div class="py-card-body">approved = False\nfeedback = ""\nwhile not approved and iterations &lt; max_iter:\n    out = await Runner.run(gen, original_input + feedback)\n    eval_result = await Runner.run(eval, out.final_output)\n    approved = eval_result.final_output.approved\n    feedback = eval_result.final_output.feedback</div></div><br/><br/><code>Runner.run()</code> returns a <code>RunResult</code> object — always extract <code>.final_output</code> before passing output between agents. The evaluator agent requires an <code>output_type</code> (e.g. a Pydantic model with <code>approved: bool</code> and <code>feedback: str</code>) so its result can be parsed as structured data.'
# MAGIC   }
# MAGIC };
# MAGIC var orchCurrent = null;
# MAGIC function orchSelect(id) {
# MAGIC   var wrap = document.getElementById('orch-detail-wrap');
# MAGIC   var card = document.getElementById('orch-detail-card');
# MAGIC   document.querySelectorAll('.orch-node').forEach(function(n) { n.classList.toggle('active', n.dataset.orch === id); });
# MAGIC   if (orchCurrent === id) { wrap.classList.remove('open'); document.querySelectorAll('.orch-node').forEach(function(n) { n.classList.remove('active'); }); orchCurrent = null; return; }
# MAGIC   orchCurrent = id;
# MAGIC   var d = ORCH[id];
# MAGIC   card.innerHTML = '<div style="height:5px;border-radius:5px;margin:-4px 0 12px;background:' + d.color + ';"></div><div style="font-size:16pt;font-weight:700;margin-bottom:12px;color:' + d.color + ';">' + d.title + '</div><div style="font-size:15pt;color:#0b2026;line-height:1.7;">' + d.html + '</div>';
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# DBTITLE 1,B. Rapid Prototyping with Framework Tool Decorators
# MAGIC %md-sandbox
# MAGIC ## B. Rapid Prototyping with Framework Tool Decorators
# MAGIC
# MAGIC Before registering a tool to Unity Catalog, you usually want to try it out quickly: write a Python function, give it to an agent, and see whether the agent calls it correctly. Each framework has a lightweight way to do this inside a notebook. They all work the same way underneath: the framework reads the function's **name**, **type hints**, and **docstring** and turns them into the tool schema the LLM sees. That is why clear docstrings and typed parameters matter so much for tool selection.
# MAGIC
# MAGIC **Click each framework** to see the same `classify_price_tier` tool prototyped three ways.
# MAGIC
# MAGIC <style>
# MAGIC .rp-outer { max-width: 1100px; margin: 0 auto; font-family: sans-serif; font-size: 12pt; }
# MAGIC .rp-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
# MAGIC .rp-node {
# MAGIC   flex: 1; min-width: 140px; background: #fff; border: 1.5px solid #C6DCEA; border-top: 5px solid transparent;
# MAGIC   border-radius: 14px; padding: 14px 12px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, box-shadow 0.15s;
# MAGIC   box-shadow: 0 2px 8px rgba(27,49,57,.07);
# MAGIC }
# MAGIC .rp-node:hover { transform: translateY(-2px); }
# MAGIC .rp-node.active { border-color: var(--rc); box-shadow: 0 4px 14px rgba(27,49,57,.14); }
# MAGIC .rp-node-title { display: block; font-size: 12pt; font-weight: 700; color: #0b2026; line-height: 1.3; pointer-events: none; }
# MAGIC .rp-node-sub { display: block; font-size: 12pt; color: #666; margin-top: 4px; pointer-events: none; }
# MAGIC .rp-detail-wrap { overflow: hidden; max-height: 0; opacity: 0; transition: max-height 0.35s ease, opacity 0.28s ease; }
# MAGIC .rp-detail-wrap.open { max-height: 1600px; opacity: 1; margin-top: 12px; }
# MAGIC .rp-detail-card { background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07); font-size: 12pt; line-height: 1.7; color: #333; }
# MAGIC .rp-note code { background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-size: 12pt; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="rp-outer">
# MAGIC
# MAGIC <div class="rp-row">
# MAGIC   <div class="rp-node" data-rp="openai" onclick="rpSelect('openai')" style="--rc:#2272B4; border-top-color:#2272B4;">
# MAGIC     <span class="rp-node-title">OpenAI Agents SDK</span>
# MAGIC     <span class="rp-node-sub">@function_tool</span>
# MAGIC   </div>
# MAGIC   <div class="rp-node" data-rp="langchain" onclick="rpSelect('langchain')" style="--rc:#FF5F46; border-top-color:#FF5F46;">
# MAGIC     <span class="rp-node-title">LangChain</span>
# MAGIC     <span class="rp-node-sub">@tool</span>
# MAGIC   </div>
# MAGIC   <div class="rp-node" data-rp="dspy" onclick="rpSelect('dspy')" style="--rc:#00A972; border-top-color:#00A972;">
# MAGIC     <span class="rp-node-title">DSPy</span>
# MAGIC     <span class="rp-node-sub">No decorator</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="rp-detail-wrap" id="rp-detail-wrap">
# MAGIC   <div class="rp-detail-card" id="rp-detail-card"></div>
# MAGIC </div>
# MAGIC
# MAGIC <div id="rp-src-openai" data-color="#2272B4" data-title="OpenAI Agents SDK" style="display:none;"><div class="rp-note"><code>@function_tool</code> builds the tool schema from the function's name, type hints, and docstring. You used this in the <em>Introduction to OpenAI Agents SDK</em> demo.</div><div style="border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;background:#F9F7F4;margin:12px 0 0 0;"><div style="display:flex;align-items:center;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;"><div style="display:inline-flex;align-items:center;gap:0.45rem;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" style="display:inline-block;vertical-align:middle;"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">Python</span></div></div><div style="margin:0;padding:1rem;overflow-x:auto;font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;font-size:0.82rem;line-height:1.55;color:#1B3139;white-space:pre;">from agents import Agent, function_tool
# MAGIC
# MAGIC @function_tool
# MAGIC def classify_price_tier(price: float) -&gt; str:
# MAGIC     &quot;&quot;&quot;Classify a nightly SF Airbnb price as Budget, Mid-Range, Premium, or Luxury.
# MAGIC
# MAGIC     Args:
# MAGIC         price: Nightly listing price in USD.
# MAGIC     &quot;&quot;&quot;
# MAGIC     if price &lt; 100:
# MAGIC         return &quot;Budget&quot;
# MAGIC     if price &lt; 175:
# MAGIC         return &quot;Mid-Range&quot;
# MAGIC     if price &lt; 300:
# MAGIC         return &quot;Premium&quot;
# MAGIC     return &quot;Luxury&quot;
# MAGIC
# MAGIC agent = Agent(name=&quot;Data Analyst&quot;, tools=[classify_price_tier])</div></div></div>
# MAGIC <div id="rp-src-langchain" data-color="#FF5F46" data-title="LangChain" style="display:none;"><div class="rp-note"><code>@tool</code> works the same way: the name, type hints, and docstring become the schema the LLM reads. You will use this in the <em>Introduction to LangChain</em> demo.</div><div style="border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;background:#F9F7F4;margin:12px 0 0 0;"><div style="display:flex;align-items:center;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;"><div style="display:inline-flex;align-items:center;gap:0.45rem;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" style="display:inline-block;vertical-align:middle;"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">Python</span></div></div><div style="margin:0;padding:1rem;overflow-x:auto;font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;font-size:0.82rem;line-height:1.55;color:#1B3139;white-space:pre;">from langchain_core.tools import tool
# MAGIC from langchain.agents import create_agent
# MAGIC
# MAGIC @tool
# MAGIC def classify_price_tier(price: float) -&gt; str:
# MAGIC     &quot;&quot;&quot;Classify a nightly SF Airbnb price as Budget, Mid-Range, Premium, or Luxury.
# MAGIC
# MAGIC     Args:
# MAGIC         price: Nightly listing price in USD.
# MAGIC     &quot;&quot;&quot;
# MAGIC     if price &lt; 100:
# MAGIC         return &quot;Budget&quot;
# MAGIC     if price &lt; 175:
# MAGIC         return &quot;Mid-Range&quot;
# MAGIC     if price &lt; 300:
# MAGIC         return &quot;Premium&quot;
# MAGIC     return &quot;Luxury&quot;
# MAGIC
# MAGIC agent = create_agent(llm, tools=[classify_price_tier], system_prompt=system_prompt)</div></div></div>
# MAGIC <div id="rp-src-dspy" data-color="#00A972" data-title="DSPy" style="display:none;"><div class="rp-note">DSPy has no tool decorator. Any typed Python function with a docstring can be passed straight to <code>dspy.ReAct</code>, which wraps it in a <code>dspy.Tool</code>. You will use this in the <em>Building Agents with DSPy</em> lab.</div><div style="border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;background:#F9F7F4;margin:12px 0 0 0;"><div style="display:flex;align-items:center;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;"><div style="display:inline-flex;align-items:center;gap:0.45rem;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" style="display:inline-block;vertical-align:middle;"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">Python</span></div></div><pre style="margin:0;padding:1rem;overflow-x:auto;font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;font-size:0.82rem;line-height:1.55;color:#1B3139;">import dspy
# MAGIC &#8203;
# MAGIC def classify_price_tier(price: float) -&gt; str:
# MAGIC     &quot;&quot;&quot;Classify a nightly SF Airbnb price as Budget, Mid-Range, Premium, or Luxury.
# MAGIC &#8203;
# MAGIC     Args:
# MAGIC         price: Nightly listing price in USD.
# MAGIC     &quot;&quot;&quot;
# MAGIC     if price &lt; 100:
# MAGIC         return &quot;Budget&quot;
# MAGIC     if price &lt; 175:
# MAGIC         return &quot;Mid-Range&quot;
# MAGIC     if price &lt; 300:
# MAGIC         return &quot;Premium&quot;
# MAGIC     return &quot;Luxury&quot;
# MAGIC &#8203;
# MAGIC &#35; A plain typed function is already a tool: dspy.ReAct wraps it in dspy.Tool
# MAGIC agent = dspy.ReAct(&quot;question -&gt; answer&quot;, tools=[classify_price_tier])
# MAGIC &#8203;
# MAGIC &#35; Optional: wrap it yourself to override the name or description
# MAGIC price_tool = dspy.Tool(classify_price_tier, desc=&quot;Classify an SF nightly price.&quot;)</pre></div></div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var rpCurrent = null;
# MAGIC function rpSelect(id) {
# MAGIC   var wrap = document.getElementById('rp-detail-wrap');
# MAGIC   var card = document.getElementById('rp-detail-card');
# MAGIC   document.querySelectorAll('.rp-node').forEach(function(n) { n.classList.toggle('active', n.dataset.rp === id); });
# MAGIC   if (rpCurrent === id) { wrap.classList.remove('open'); document.querySelectorAll('.rp-node').forEach(function(n) { n.classList.remove('active'); }); rpCurrent = null; return; }
# MAGIC   rpCurrent = id;
# MAGIC   var src = document.getElementById('rp-src-' + id);
# MAGIC   var color = src.getAttribute('data-color');
# MAGIC   card.innerHTML = '<div style="height:5px;border-radius:5px;margin:-4px 0 12px;background:' + color + ';"></div><div style="font-size:16pt;font-weight:700;margin-bottom:12px;color:' + color + ';">' + src.getAttribute('data-title') + '</div><div style="font-size:15pt;color:#0b2026;line-height:1.7;">' + src.innerHTML + '</div>';
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# DBTITLE 1,B1. From Prototype to Governed Tool
# MAGIC %md-sandbox
# MAGIC ### B1. From Prototype to Governed Tool
# MAGIC
# MAGIC A prototyped tool exists only in your notebook session. Once it works, the recommended path is the same in every framework: register the logic as a **Unity Catalog function** so it is governed and reusable, then let agents reach it through a **Managed MCP server**.
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;gap:10px;flex-wrap:wrap;margin:16px 0;font-family:sans-serif;">
# MAGIC   <div style="flex: 1 1 180px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);"><div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #5A6F77;"></div><div style="font-size:17px;font-weight:700;color:#0b2026;">1. Prototype</div><div style="font-size:11pt;color:#5A6F77;margin-top:4px;">Decorator or plain function. Local to the notebook.</div></div>
# MAGIC   <div style="display:flex;align-items:center;font-size:18pt;color:#618794;">&#x2192;</div>
# MAGIC   <div style="flex: 1 1 180px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);"><div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #FF3621;"></div><div style="font-size:17px;font-weight:700;color:#0b2026;">2. Register to Unity Catalog</div><div style="font-size:11pt;color:#5A6F77;margin-top:4px;">Governed, permissioned, and reusable across agents.</div></div>
# MAGIC   <div style="display:flex;align-items:center;font-size:18pt;color:#618794;">&#x2192;</div>
# MAGIC   <div style="flex: 1 1 180px; background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);"><div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #2272B4;"></div><div style="font-size:17px;font-weight:700;color:#0b2026;">3. Expose through MCP</div><div style="font-size:11pt;color:#5A6F77;margin-top:4px;">Discoverable at runtime by any MCP-aware framework.</div></div>
# MAGIC </div>
# MAGIC
# MAGIC Each framework connects to the governed tool in its own way:
# MAGIC
# MAGIC | Framework | Prototype a tool | Use the governed UC tool |
# MAGIC |---|---|---|
# MAGIC | **OpenAI Agents SDK** | `@function_tool` | `McpServer` from `databricks_openai.agents`, passed as `mcp_servers=[...]` (see section C) |
# MAGIC | **LangChain** | `@tool` | Discover tools with `DatabricksMCPClient` and wrap each one as a LangChain `StructuredTool` |
# MAGIC | **DSPy** | A plain typed function, optionally wrapped in `dspy.Tool` | Convert MCP tools with `dspy.Tool.from_mcp_tool(...)`, or wrap a call to the UC function in a plain function |
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Tool definition is not where DSPy stands out. Its strength is <strong>optimizing</strong> the prompts and modules around your tools, which section E covers. For tools, think of DSPy as the lightest option: no decorator, just a well-documented function.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,C. Databricks MCP Tools
# MAGIC %md-sandbox
# MAGIC ## C. Databricks MCP Tools with `databricks_openai.agents.McpServer`
# MAGIC
# MAGIC Once a prototyped tool is registered to Unity Catalog, production agents built with the OpenAI Agents SDK reach it through `McpServer` ([source](https://github.com/databricks/databricks-ai-bridge/blob/main/integrations/openai/src/databricks_openai/agents/mcp_server.py)). It connects agents to Unity Catalog functions, AI Search indexes, and other Databricks services through the Model Context Protocol.
# MAGIC
# MAGIC **Click on each layer** to see how the pieces connect.
# MAGIC
# MAGIC <style>
# MAGIC .mcp-outer { max-width: 1100px; margin: 0 auto; font-family: sans-serif; font-size: 12pt; }
# MAGIC .mcp-diagram { display: flex; align-items: stretch; gap: 0; margin: 20px 0; }
# MAGIC .mcp-layer {
# MAGIC   flex: 1; padding: 14px 12px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, opacity 0.2s; border-right: 2px solid #fff;
# MAGIC }
# MAGIC .mcp-layer:last-child { border-right: none; }
# MAGIC .mcp-layer:hover { transform: translateY(-2px); }
# MAGIC .mcp-layer.active { opacity: 1; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
# MAGIC .mcp-layer-title { font-size: 12pt; font-weight: 700; color: #fff; }
# MAGIC .mcp-layer-sub { font-size: 10pt; color: rgba(255,255,255,0.8); margin-top: 4px; }
# MAGIC .mcp-arrow { display: flex; justify-content: space-around; margin: 4px 0; font-size: 14px; color: #999; }
# MAGIC .mcp-detail-wrap { overflow: hidden; max-height: 0; opacity: 0; transition: max-height 0.35s ease, opacity 0.28s ease; }
# MAGIC .mcp-detail-wrap.open { max-height: 2000px; opacity: 1; margin-top: 14px; }
# MAGIC .mcp-detail-card { background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07); font-size: 12pt; line-height: 1.7; color: #333; }
# MAGIC .mcp-detail-card strong { color: #0b2026; }
# MAGIC .mcp-detail-card code { background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-size: 11pt; font-family: 'Menlo','Consolas',monospace; }
# MAGIC .mcp-code { background: #F9F7F4; color: #1B3139; border-radius: 0 0 8px 8px; padding: 1rem; margin-top: 0; font-family: 'SF Mono','Cascadia Code','Fira Code',Consolas,monospace; font-size: 0.82rem; line-height: 1.55; overflow-x: auto; }
# MAGIC .mcp-code .kw { color: #7C3AED; } .mcp-code .fn { color: #2563EB; } .mcp-code .st { color: #059669; } .mcp-code .cm { color: #6B7280; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="mcp-outer">
# MAGIC
# MAGIC <div class="mcp-diagram">
# MAGIC   <div class="mcp-layer" style="background:#FF5F46;border-radius:8px 0 0 8px;" data-mcp="agent" onclick="mcpSelect('agent')">
# MAGIC     <div class="mcp-layer-title">Agent</div>
# MAGIC     <div class="mcp-layer-sub">OpenAI Agents SDK</div>
# MAGIC   </div>
# MAGIC   <div class="mcp-layer" style="background:#4299E0;" data-mcp="mcpserver" onclick="mcpSelect('mcpserver')">
# MAGIC     <div class="mcp-layer-title">McpServer</div>
# MAGIC     <div class="mcp-layer-sub">databricks_openai.agents</div>
# MAGIC   </div>
# MAGIC   <div class="mcp-layer" style="background:#1B5162;" data-mcp="protocol" onclick="mcpSelect('protocol')">
# MAGIC     <div class="mcp-layer-title">MCP Protocol</div>
# MAGIC     <div class="mcp-layer-sub">Streamable HTTP</div>
# MAGIC   </div>
# MAGIC   <div class="mcp-layer" style="background:#00A972;border-radius:0 8px 8px 0;" data-mcp="uc" onclick="mcpSelect('uc')">
# MAGIC     <div class="mcp-layer-title">Unity Catalog</div>
# MAGIC     <div class="mcp-layer-sub">Functions, AI Search</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="mcp-arrow">
# MAGIC   <span>&#x2194; mcp_servers=[...]</span>
# MAGIC   <span>&#x2194; tools/list, tools/call</span>
# MAGIC   <span>&#x2194; /api/2.0/mcp/functions/...</span>
# MAGIC </div>
# MAGIC
# MAGIC <div class="mcp-detail-wrap" id="mcp-detail-wrap">
# MAGIC   <div class="mcp-detail-card" id="mcp-detail-card"></div>
# MAGIC </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var MCP = {
# MAGIC   agent: {
# MAGIC     color: '#FF5F46',
# MAGIC     title: 'Agent Layer',
# MAGIC     html: '<strong>What it does:</strong> The agent receives <code>mcp_servers=[mcp_server]</code> and treats each MCP server as a source of tools. At runtime, the agent calls <code>tools/list</code> to discover available tools, then calls <code>tools/call</code> when the LLM decides to invoke one.<br/><br/><strong>Key point:</strong> The agent does not need to know what tools exist in advance. MCP provides runtime discovery.<br/><br/><div class="code-card-wrap" style="border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;margin:12px 0;"><div style="display:flex;align-items:center;justify-content:space-between;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;"><div style="display:inline-flex;align-items:center;gap:0.45rem;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" style="display:inline-block;vertical-align:middle;"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">Python</span></div></div><div class="mcp-code"><span class="cm"># Agent with MCP tools</span><br/><span class="kw">from</span> agents <span class="kw">import</span> Agent, Runner<br/><span class="kw">from</span> databricks_openai.agents <span class="kw">import</span> McpServer<br/><br/><span class="kw">async with</span> <span class="fn">McpServer</span>(url=mcp_url) <span class="kw">as</span> server:<br/>&nbsp;&nbsp;&nbsp;&nbsp;agent = <span class="fn">Agent</span>(<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;name=<span class="st">"my-agent"</span>,<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mcp_servers=[server],<br/>&nbsp;&nbsp;&nbsp;&nbsp;)<br/>&nbsp;&nbsp;&nbsp;&nbsp;result = <span class="kw">await</span> <span class="fn">Runner.run</span>(agent, messages)</div></div>'
# MAGIC   },
# MAGIC   mcpserver: {
# MAGIC     color: '#4299E0',
# MAGIC     title: 'McpServer Class',
# MAGIC     html: '<strong>What it does:</strong> <code>McpServer</code> extends the OpenAI Agents SDK <code>MCPServerStreamableHttp</code>. It adds Databricks authentication (<code>DatabricksOAuthClientProvider</code>), automatic URL construction, and MLflow tracing on every <code>call_tool</code>.<br/><br/><strong>Three factory methods:</strong><ul><li><code>McpServer(url=...)</code> -- direct URL to any MCP endpoint</li><li><code>McpServer.from_uc_function(catalog, schema, function_name)</code> -- builds URL: <code>/api/2.0/mcp/functions/{catalog}/{schema}/{function}</code></li><li><code>McpServer.from_vector_search(catalog, schema, index_name)</code> -- builds URL: <code>/api/2.0/mcp/vector-search/{catalog}/{schema}/{index}</code></li></ul><strong>Built-in tracing:</strong> Every tool call is wrapped with <code>@mlflow.trace(span_type=SpanType.TOOL)</code>, so MLflow automatically captures tool invocations as spans.<br/><br/><div class="code-card-wrap" style="border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;margin:12px 0;"><div style="display:flex;align-items:center;justify-content:space-between;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;"><div style="display:inline-flex;align-items:center;gap:0.45rem;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" style="display:inline-block;vertical-align:middle;"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">Python</span></div></div><div class="mcp-code"><span class="cm"># Factory method for UC functions</span><br/>server = <span class="fn">McpServer.from_uc_function</span>(<br/>&nbsp;&nbsp;&nbsp;&nbsp;catalog=<span class="st">"main"</span>,<br/>&nbsp;&nbsp;&nbsp;&nbsp;schema=<span class="st">"tools"</span>,<br/>&nbsp;&nbsp;&nbsp;&nbsp;function_name=<span class="st">"get_top_products"</span>,<br/>&nbsp;&nbsp;&nbsp;&nbsp;timeout=30.0,<br/>)</div></div>'
# MAGIC   },
# MAGIC   protocol: {
# MAGIC     color: '#1B5162',
# MAGIC     title: 'MCP Protocol Layer',
# MAGIC     html: '<strong>What it does:</strong> The protocol layer handles the actual communication. <code>McpServer</code> uses Streamable HTTP transport (via <code>streamablehttp_client</code> from the <code>mcp</code> library) to connect to the server.<br/><br/><strong>Connection lifecycle:</strong><ol><li><code>async with McpServer(...) as server:</code> opens the HTTP connection</li><li>Agent calls <code>tools/list</code> to discover available tools</li><li>Agent calls <code>tools/call(tool_name, arguments)</code> to invoke a tool</li><li>Context manager exit closes the connection</li></ol><strong>Authentication:</strong> <code>McpServer</code> injects a <code>DatabricksOAuthClientProvider</code> into the HTTP transport. This handles token refresh automatically using the <code>WorkspaceClient</code> credentials. No manual token management needed.'
# MAGIC   },
# MAGIC   uc: {
# MAGIC     color: '#00A972',
# MAGIC     title: 'Unity Catalog Backend',
# MAGIC     html: '<strong>What it does:</strong> Databricks hosts Managed MCP servers that expose Unity Catalog resources as tools. The MCP server translates <code>tools/list</code> and <code>tools/call</code> into UC operations.<br/><br/><strong>Supported backends:</strong><table style="width:100%;border-collapse:collapse;margin-top:10px;font-size:12pt;"><tr style="border-bottom:1px solid #ddd;"><th style="text-align:left;padding:6px;">Type</th><th style="text-align:left;padding:6px;">URL Pattern</th><th style="text-align:left;padding:6px;">What It Exposes</th></tr><tr style="border-bottom:1px solid #ddd;"><td style="padding:6px;">UC Functions</td><td style="padding:6px;"><code>/api/2.0/mcp/functions/{cat}/{schema}</code></td><td style="padding:6px;">SQL and Python UC functions as callable tools</td></tr><tr style="border-bottom:1px solid #ddd;"><td style="padding:6px;">AI Search</td><td style="padding:6px;"><code>/api/2.0/mcp/vector-search/{cat}/{schema}</code></td><td style="padding:6px;">AI Search indexes as retrieval tools</td></tr><tr><td style="padding:6px;">Genie</td><td style="padding:6px;"><code>/api/2.0/mcp/genie/{space_id}</code></td><td style="padding:6px;">Genie spaces as natural language query tools</td></tr></table><br/><strong>Governance:</strong> UC permissions are always enforced. Agents can only access tools the authenticated user (or service principal) is authorized to use.'
# MAGIC   }
# MAGIC };
# MAGIC function copyCodeCard(btn){var c=btn.closest('.code-card-wrap').querySelector('.mcp-code').textContent,ta=document.createElement('textarea');ta.value=c;ta.style.position='fixed';ta.style.opacity='0';document.body.appendChild(ta);ta.select();try{document.execCommand('copy')}catch(e){}document.body.removeChild(ta);var l=btn.querySelector('.copy-label');l.textContent='Copied!';btn.style.color='#2272B4';setTimeout(function(){l.textContent='Copy';btn.style.color='#5A6F77'},2000)}
# MAGIC var mcpCurrent = null;
# MAGIC function mcpSelect(id) {
# MAGIC   var wrap = document.getElementById('mcp-detail-wrap');
# MAGIC   var card = document.getElementById('mcp-detail-card');
# MAGIC   document.querySelectorAll('.mcp-layer').forEach(function(n) { n.classList.toggle('active', n.dataset.mcp === id); });
# MAGIC   if (mcpCurrent === id) { wrap.classList.remove('open'); document.querySelectorAll('.mcp-layer').forEach(function(n) { n.classList.remove('active'); }); mcpCurrent = null; return; }
# MAGIC   mcpCurrent = id;
# MAGIC   var d = MCP[id];
# MAGIC   card.innerHTML = '<div style="height:5px;border-radius:5px;margin:-4px 0 12px;background:' + d.color + ';"></div><div style="font-size:16pt;font-weight:700;margin-bottom:12px;color:' + d.color + ';">' + d.title + '</div><div style="font-size:15pt;color:#0b2026;line-height:1.7;">' + d.html + '</div>';
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# DBTITLE 1,D. Multi-Agent Concepts
# MAGIC %md-sandbox
# MAGIC ## D. Multi-Agent Concepts
# MAGIC
# MAGIC A single agent with tools is powerful, but many real-world workflows require **multiple specialized agents** working together. The OpenAI Agents SDK supports this natively through the `handoffs` parameter.
# MAGIC
# MAGIC **Click on each concept** to learn how multi-agent systems work.
# MAGIC
# MAGIC <style>
# MAGIC .ma-outer { max-width: 1100px; margin: 0 auto; font-family: sans-serif; font-size: 12pt; }
# MAGIC .ma-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
# MAGIC .ma-node {
# MAGIC   flex: 1; min-width: 160px; background: #fff; border: 1.5px solid #C6DCEA; border-top: 5px solid transparent;
# MAGIC   border-radius: 14px; padding: 14px 12px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, box-shadow 0.15s;
# MAGIC   box-shadow: 0 2px 8px rgba(27,49,57,.07);
# MAGIC }
# MAGIC .ma-node:hover { transform: translateY(-2px); }
# MAGIC .ma-node.active { border-color: var(--mc); box-shadow: 0 4px 14px rgba(27,49,57,.14); }
# MAGIC .ma-node-title { display: block; font-size: 12pt; font-weight: 700; color: #0b2026; line-height: 1.3; pointer-events: none; }
# MAGIC .ma-node-sub { display: block; font-size: 12pt; color: #666; margin-top: 4px; pointer-events: none; }
# MAGIC .ma-detail-wrap { overflow: hidden; max-height: 0; opacity: 0; transition: max-height 0.35s ease, opacity 0.28s ease; }
# MAGIC .ma-detail-wrap.open { max-height: 1600px; opacity: 1; margin-top: 12px; }
# MAGIC .ma-detail-card { background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07); font-size: 12pt; line-height: 1.7; color: #333; }
# MAGIC .py-card { border:1px solid #DCE0E2; border-radius:8px; overflow:hidden; background:#F9F7F4; margin:12px 0; }
# MAGIC .py-card-hdr { display:flex; align-items:center; padding:0.45rem 0.75rem; background:#EEEDE9; border-bottom:1px solid #DCE0E2; }
# MAGIC .py-card-lang { display:inline-flex; align-items:center; gap:0.45rem; }
# MAGIC .py-card-lang span { font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#5A6F77; }
# MAGIC .py-card-body { margin:0; padding:1rem; font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace; font-size:0.82rem; line-height:1.55; color:#1B3139; white-space:pre-wrap; word-break:break-word; }
# MAGIC .ma-detail-card strong { color: #0b2026; }
# MAGIC .ma-detail-card code { background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-size: 12pt; }
# MAGIC .ma-flow { display: flex; align-items: center; justify-content: center; gap: 0; margin: 14px 0; flex-wrap: wrap; }
# MAGIC .ma-flow-box { background: #1B3139; color: #fff; border-radius: 6px; padding: 8px 14px; font-size: 12pt; font-weight: 600; text-align: center; min-width: 90px; }
# MAGIC .ma-flow-arrow { color: #999; font-size: 20px; padding: 0 6px; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="ma-outer">
# MAGIC <div class="ma-row">
# MAGIC   <div class="ma-node" data-ma="supervisor" onclick="maSelect('supervisor')" style="--mc:#FF5F46; border-top-color:#FF5F46;">
# MAGIC     <span class="ma-node-title">Supervisor Pattern</span>
# MAGIC     <span class="ma-node-sub">Route queries to specialists</span>
# MAGIC   </div>
# MAGIC   <div class="ma-node" data-ma="workers" onclick="maSelect('workers')" style="--mc:#4299E0; border-top-color:#4299E0;">
# MAGIC     <span class="ma-node-title">Worker Specialization</span>
# MAGIC     <span class="ma-node-sub">Each agent has focused tools</span>
# MAGIC   </div>
# MAGIC   <div class="ma-node" data-ma="mcp-multi" onclick="maSelect('mcp-multi')" style="--mc:#00A972; border-top-color:#00A972;">
# MAGIC     <span class="ma-node-title">MCP-Backed Workers</span>
# MAGIC     <span class="ma-node-sub">UC tools per agent via MCP</span>
# MAGIC   </div>
# MAGIC   <div class="ma-node" data-ma="tracing-multi" onclick="maSelect('tracing-multi')" style="--mc:#9B1D20; border-top-color:#9B1D20;">
# MAGIC     <span class="ma-node-title">Distributed Tracing</span>
# MAGIC     <span class="ma-node-sub">Unified trace across agents</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="ma-detail-wrap" id="ma-detail-wrap">
# MAGIC   <div class="ma-detail-card" id="ma-detail-card"></div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var MA = {
# MAGIC   supervisor: {
# MAGIC     color: '#FF5F46',
# MAGIC     title: 'Supervisor Pattern',
# MAGIC     html: 'A supervisor agent receives all user queries and decides which worker to delegate to. The supervisor does not call tools directly. Instead, it uses <code>handoffs</code> to transfer the conversation to a specialist.<br/><br/><div class="ma-flow"><div class="ma-flow-box" style="background:transparent;color:#0b2026;border:2px solid #0b2026;">User</div><div class="ma-flow-arrow">&#x2192;</div><div class="ma-flow-box" style="background:#FF5F46;">Supervisor</div><div class="ma-flow-arrow">&#x2192;</div><div style="display:flex;flex-direction:column;gap:4px;"><div class="ma-flow-box" style="background:#4299E0;">Data Analyst</div><div class="ma-flow-box" style="background:#00A972;">Business Analyst</div></div><div class="ma-flow-arrow">&#x2192;</div><div class="ma-flow-box" style="background:transparent;color:#0b2026;border:2px solid #0b2026;">User</div></div><div class="py-card"><div class="py-card-hdr"><div class="py-card-lang"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span>Python</span></div></div><div class="py-card-body">Agent(name="Supervisor", handoffs=[data_analyst, business_analyst])</div></div>The supervisor reads each worker name and set of instructions to decide which one is best suited for the request.'
# MAGIC   },
# MAGIC   workers: {
# MAGIC     color: '#4299E0',
# MAGIC     title: 'Worker Specialization',
# MAGIC     html: 'Each worker agent has a focused role with its own instructions and tools. This keeps prompts concise and tool sets relevant.<br/><br/><strong>Example of specialized agents:</strong><ul><li><strong>Data Analyst</strong> - uses <code>classify_price_tier</code> to categorize Airbnb prices</li><li><strong>Business Analyst</strong> - uses <code>neighborhood_summary</code> for market insights</li></ul><strong>Why specialize?</strong> A single agent with many tools may struggle to select the right one. Specialized agents with 1-2 tools each make more accurate tool selections and produce more focused responses.'
# MAGIC   },
# MAGIC   'mcp-multi': {
# MAGIC     color: '#00A972',
# MAGIC     title: 'MCP-Backed Workers',
# MAGIC     html: 'Each worker agent connects to its own <code>McpServer</code> instance, scoped to the specific UC function it needs. This means each agent only sees the tools relevant to its role.<br/><br/><div class="py-card"><div class="py-card-hdr"><div class="py-card-lang"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span>Python</span></div></div><div class="py-card-body">da_mcp = McpServer.from_uc_function(catalog, schema, "classify_price_tier")\nba_mcp = McpServer.from_uc_function(catalog, schema, "neighborhood_summary")\n\ndata_analyst = Agent(mcp_servers=[da_mcp])\nbiz_analyst = Agent(mcp_servers=[ba_mcp])</div></div><strong>Key point:</strong> The <code>async with</code> context manager must wrap the entire multi-agent flow so all MCP connections stay open during execution.'
# MAGIC   },
# MAGIC   'tracing-multi': {
# MAGIC     color: '#9B1D20',
# MAGIC     title: 'Distributed Tracing',
# MAGIC     html: '<strong>Challenge:</strong> With <code>mlflow.openai.autolog()</code> alone, each agent invocation produces a separate trace. The supervisor call and each worker call appear as independent traces, making it hard to see the full orchestration.<br/><br/><strong>Solution:</strong> Wrap the entire multi-agent flow in a parent <code>@mlflow.trace</code> span. All child spans (routing, worker LLM calls, MCP tool calls) nest under this root, producing a single trace tree.<br/><br/><div class="py-card"><div class="py-card-hdr"><div class="py-card-lang"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span>Python</span></div></div><div class="py-card-body">@mlflow.trace(span_type="AGENT", name="supervisor_orchestration")\nasync def run_supervisor(question):\n    # All agent calls inside become child spans\n    result = await Runner.run(supervisor, question)\n    return result.final_output</div></div>'
# MAGIC   }
# MAGIC };
# MAGIC var maCurrent = null;
# MAGIC function maSelect(id) {
# MAGIC   var wrap = document.getElementById('ma-detail-wrap');
# MAGIC   var card = document.getElementById('ma-detail-card');
# MAGIC   document.querySelectorAll('.ma-node').forEach(function(n) { n.classList.toggle('active', n.dataset.ma === id); });
# MAGIC   if (maCurrent === id) { wrap.classList.remove('open'); document.querySelectorAll('.ma-node').forEach(function(n) { n.classList.remove('active'); }); maCurrent = null; return; }
# MAGIC   maCurrent = id;
# MAGIC   var d = MA[id];
# MAGIC   card.innerHTML = '<div style="height:5px;border-radius:5px;margin:-4px 0 12px;background:' + d.color + ';"></div><div style="font-size:16pt;font-weight:700;margin-bottom:12px;color:' + d.color + ';">' + d.title + '</div><div style="font-size:15pt;color:#0b2026;line-height:1.7;">' + d.html + '</div>';
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# DBTITLE 1,E. Other Supported Frameworks
# MAGIC %md-sandbox
# MAGIC ## E. Other Supported Frameworks
# MAGIC
# MAGIC While this course focuses on the OpenAI Agents SDK, Databricks supports multiple agent authoring frameworks. The key requirement is wrapping your agent with the MLflow `ResponsesAgent` interface for deployment. Some other popular frameworks are shown below. 
# MAGIC
# MAGIC **Click on each framework** to learn about its strengths and integration with Databricks.
# MAGIC
# MAGIC <style>
# MAGIC .fw-outer { max-width: 1100px; margin: 0 auto; font-family: sans-serif; font-size: 12pt; }
# MAGIC .fw-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
# MAGIC .fw-node {
# MAGIC   flex: 1; min-width: 140px; background: #fff; border: 1.5px solid #C6DCEA; border-top: 5px solid transparent;
# MAGIC   border-radius: 14px; padding: 14px 12px; text-align: center; cursor: pointer; user-select: none;
# MAGIC   transition: transform 0.12s, box-shadow 0.15s;
# MAGIC   box-shadow: 0 2px 8px rgba(27,49,57,.07);
# MAGIC }
# MAGIC .fw-node:hover { transform: translateY(-2px); }
# MAGIC .fw-node.active { border-color: var(--fc); box-shadow: 0 4px 14px rgba(27,49,57,.14); }
# MAGIC .fw-node-title { display: block; font-size: 12pt; font-weight: 700; color: #0b2026; line-height: 1.3; pointer-events: none; }
# MAGIC .fw-node-sub { display: block; font-size: 12pt; color: #666; margin-top: 4px; pointer-events: none; }
# MAGIC .fw-detail-wrap { overflow: hidden; max-height: 0; opacity: 0; transition: max-height 0.35s ease, opacity 0.28s ease; }
# MAGIC .fw-detail-wrap.open { max-height: 1600px; opacity: 1; margin-top: 12px; }
# MAGIC .fw-detail-card { background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07); font-size: 12pt; line-height: 1.7; color: #333; }
# MAGIC .fw-detail-card strong { color: #0b2026; }
# MAGIC .fw-detail-card code { background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-size: 12pt; }
# MAGIC .fw-code { background: #F9F7F4; color: #1B3139; border-radius: 0 0 8px 8px; padding: 1rem; margin-top: 0; font-family: 'SF Mono','Cascadia Code','Fira Code',Consolas,monospace; font-size: 0.82rem; line-height: 1.55; overflow-x: auto; }
# MAGIC .fw-code .kw { color: #7C3AED; } .fw-code .fn { color: #2563EB; } .fw-code .st { color: #059669; } .fw-code .cm { color: #6B7280; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="fw-outer">
# MAGIC <div class="fw-row">
# MAGIC   <div class="fw-node" data-fw="langchain" onclick="fwSelect('langchain')" style="--fc:#FF5F46; border-top-color:#FF5F46;">
# MAGIC     <span class="fw-node-title">LangChain</span>
# MAGIC     <span class="fw-node-sub">Composable chains and tools</span>
# MAGIC   </div>
# MAGIC   <div class="fw-node" data-fw="langgraph" onclick="fwSelect('langgraph')" style="--fc:#4299E0; border-top-color:#4299E0;">
# MAGIC     <span class="fw-node-title">LangGraph</span>
# MAGIC     <span class="fw-node-sub">Stateful graph workflows</span>
# MAGIC   </div>
# MAGIC   <div class="fw-node" data-fw="dspy" onclick="fwSelect('dspy')" style="--fc:#00A972; border-top-color:#00A972;">
# MAGIC     <span class="fw-node-title">DSPy</span>
# MAGIC     <span class="fw-node-sub">Prompt optimization</span>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="fw-detail-wrap" id="fw-detail-wrap">
# MAGIC   <div class="fw-detail-card" id="fw-detail-card"></div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <script>
# MAGIC var FW = {
# MAGIC   langchain: {
# MAGIC     color: '#FF5F46',
# MAGIC     title: 'LangChain',
# MAGIC     html: '<strong>What it is:</strong> A composable framework for building LLM applications using chains, agents, and tools.<br/><br/><strong>Install:</strong><br/><code>%pip install unitycatalog-langchain[databricks]</code><br/><code>%pip install databricks-langchain</code><br/><br/><strong>Databricks integration:</strong><ul><li><code>databricks-langchain</code> provides <strong>ChatDatabricks</strong> and <strong>UCFunctionToolkit</strong> to expose Unity Catalog functions as LangChain tools.</li><li>Use <strong>create_agent</strong> from <code>langchain.agents</code> to build the agent loop. It replaces the <code>AgentExecutor</code> + <code>create_tool_calling_agent</code> pattern from LangChain 0.x.</li><li>Use <strong><code>mlflow.langchain.autolog()</code></strong> to enable automatic tracing for the LangChain app.</li></ul><em>Note: Databricks now recommends MCP-based tools for most new agents. UCFunctionToolkit remains supported for UC function tools.</em><br/><br/><div class="code-card-wrap" style="border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;margin:12px 0;"><div style="display:flex;align-items:center;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;"><div style="display:inline-flex;align-items:center;gap:0.45rem;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">Python</span></div></div><div class="fw-code"><span class="kw">import</span> mlflow<br/><span class="kw">from</span> databricks_langchain <span class="kw">import</span> ChatDatabricks, UCFunctionToolkit<br/><span class="kw">from</span> langchain.agents <span class="kw">import</span> create_agent<br/><br/>LLM_ENDPOINT_NAME = <span class="st">"databricks-gpt-5-2"</span><br/><br/>llm = <span class="fn">ChatDatabricks</span>(endpoint=LLM_ENDPOINT_NAME)<br/>toolkit = <span class="fn">UCFunctionToolkit</span>(function_names=[<span class="st">"catalog.schema.my_func"</span>])<br/>tools = toolkit.tools<br/><br/>mlflow.langchain.<span class="fn">autolog</span>()<br/><br/>agent = <span class="fn">create_agent</span>(llm, tools=tools, system_prompt=<span class="st">"You are a helpful assistant."</span>)<br/>agent.<span class="fn">invoke</span>({<span class="st">"messages"</span>: [(<span class="st">"human"</span>, <span class="st">"What is the average price in Mission?"</span>)]})</div></div>'
# MAGIC   },
# MAGIC   langgraph: {
# MAGIC     color: '#4299E0',
# MAGIC     title: 'LangGraph',
# MAGIC     html: '<strong>What it is:</strong> An open-source library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Agents are defined as nodes in a directed graph with explicit state management and conditional edges.<br/><br/><strong>Databricks integration:</strong><ul><li>Use <strong>ChatDatabricks</strong> from <code>databricks-langchain</code> with <strong>create_agent</strong> from <code>langchain.agents</code>, which is built on LangGraph</li><li>Stateful, multi-actor agent graphs suitable for long-running workflows (LangGraph provides persistence/checkpointing features)</li><li>Tools can come from managed MCP servers via <code>DatabricksMCPServer</code> + <code>DatabricksMultiServerMCPClient</code>, then passed into <code>create_agent</code></li><li><code>mlflow.langchain.autolog()</code> traces LangGraph execution, capturing graph steps as spans</li></ul><div class="code-card-wrap" style="border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;margin:12px 0;"><div style="display:flex;align-items:center;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;"><div style="display:inline-flex;align-items:center;gap:0.45rem;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">Python</span></div></div><div class="fw-code"><span class="kw">import</span> mlflow<br/><span class="kw">from</span> langchain_core.tools <span class="kw">import</span> tool<br/><span class="kw">from</span> langchain.agents <span class="kw">import</span> create_agent<br/><span class="kw">from</span> databricks_langchain <span class="kw">import</span> ChatDatabricks<br/><br/><span class="cm"># Define a tool</span><br/>@tool<br/><span class="kw">def</span> <span class="fn">get_current_time</span>() -> str:<br/>&nbsp;&nbsp;<span class="st">"""Get the current date and time."""</span><br/>&nbsp;&nbsp;<span class="kw">from</span> datetime <span class="kw">import</span> datetime<br/>&nbsp;&nbsp;<span class="kw">return</span> datetime.<span class="fn">now</span>().<span class="fn">isoformat</span>()<br/><br/>mlflow.langchain.<span class="fn">autolog</span>()<br/><br/>llm = <span class="fn">ChatDatabricks</span>(endpoint=<span class="st">"databricks-gpt-5-2"</span>)<br/>agent = <span class="fn">create_agent</span>(llm, tools=[get_current_time])</div></div>'
# MAGIC   },
# MAGIC   dspy: {
# MAGIC     color: '#00A972',
# MAGIC     title: 'DSPy',
# MAGIC     html: '<strong>What it is:</strong> A framework for programmatically defining and optimizing generative AI agents. You specify input/output signatures and the DSPy compiler automatically optimizes prompts and, when applicable, model weights for your task and data.<br/><br/><strong>Databricks integration:</strong><ul><li><code>mlflow.dspy.autolog()</code> captures DSPy program, evaluation, and compilation traces into MLflow on Databricks</li><li>Databricks provides DSPy example notebooks for text classification, RAG over AI Search, multi-agent systems with Genie + DSPy, and migrating LangChain to DSPy</li><li>For production agents, use the <code>databricks-dspy</code> package alongside MLflow <code>ResponsesAgent</code></li></ul><div class="code-card-wrap" style="border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;margin:12px 0;"><div style="display:flex;align-items:center;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;"><div style="display:inline-flex;align-items:center;gap:0.45rem;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2C8 2 8 4.5 8 4.5V7h4v1H6.5S3 8 3 12.5 5.5 17 5.5 17H8v-3.5S7.8 11 10 11h4s2 .1 2-2V4.5S16 2 12 2zm-1 1.8a.9.9 0 110 1.8.9.9 0 010-1.8z" fill="#5A6F77"/><path d="M12 22c4 0 4-2.5 4-2.5V17h-4v-1h5.5s3.5 0 3.5-4.5S18.5 7 18.5 7H16v3.5s.2 2.5-2 2.5h-4s-2-.1-2 2v4.5S8 22 12 22zm1-1.8a.9.9 0 110-1.8.9.9 0 010 1.8z" fill="none" stroke="#5A6F77" stroke-width="1.5"/></svg><span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">Python</span></div></div><div class="fw-code"><span class="kw">import</span> mlflow<br/><span class="kw">import</span> dspy<br/><br/>mlflow.dspy.<span class="fn">autolog</span>()<br/>mlflow.<span class="fn">set_tracking_uri</span>(<span class="st">"databricks"</span>)<br/>mlflow.<span class="fn">set_experiment</span>(<span class="st">"/Shared/dspy-tracing-demo"</span>)<br/><br/>lm = dspy.<span class="fn">LM</span>(<span class="st">"databricks/databricks-gpt-5-2"</span>)<br/>dspy.<span class="fn">configure</span>(lm=lm)<br/><br/><span class="cm"># Define a signature, not a prompt</span><br/><span class="kw">class</span> <span class="fn">QA</span>(dspy.Signature):<br/>&nbsp;&nbsp;question: str = dspy.<span class="fn">InputField</span>()<br/>&nbsp;&nbsp;answer: str = dspy.<span class="fn">OutputField</span>()<br/><br/>predictor = dspy.<span class="fn">Predict</span>(QA)</div></div>'
# MAGIC   }
# MAGIC };
# MAGIC var fwCurrent = null;
# MAGIC function fwSelect(id) {
# MAGIC   var wrap = document.getElementById('fw-detail-wrap');
# MAGIC   var card = document.getElementById('fw-detail-card');
# MAGIC   document.querySelectorAll('.fw-node').forEach(function(n) { n.classList.toggle('active', n.dataset.fw === id); });
# MAGIC   if (fwCurrent === id) { wrap.classList.remove('open'); document.querySelectorAll('.fw-node').forEach(function(n) { n.classList.remove('active'); }); fwCurrent = null; return; }
# MAGIC   fwCurrent = id;
# MAGIC   var d = FW[id];
# MAGIC   card.innerHTML = '<div style="height:5px;border-radius:5px;margin:-4px 0 12px;background:' + d.color + ';"></div><div style="font-size:16pt;font-weight:700;margin-bottom:12px;color:' + d.color + ';">' + d.title + '</div><div style="font-size:15pt;color:#0b2026;line-height:1.7;">' + d.html + '</div>';
# MAGIC   wrap.classList.add('open');
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. MLflow Integration

# COMMAND ----------

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
# MAGIC     Want to know more about how MLflow integrates with the OpenAI Agent SDK on Databricks? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/lecture 1/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #F9F7F4; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-query" style="flex: 1;">What does set_trace_processors([]) from the OpenAI Agents SDK do? Why is it needed with MLflow?</span>
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
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### F1. Traditional ML vs Agents
# MAGIC
# MAGIC For robust operations with GenAI applications and agents, you often need deeper insight into **server-side behavior**, including **latency/cost metrics** and **API logging**, beyond what is typically required for traditional ML batch workloads.
# MAGIC
# MAGIC | | Traditional ML | Agents |
# MAGIC |---|---|---|
# MAGIC | **Shape** | Single call | Multi-step, often looped |
# MAGIC | **Visible by default** | Input, output | Input, final output |
# MAGIC | **Unit of observation** | The call | A **span** per step |
# MAGIC | **Needed to debug** | Features, latency | Per-step I/O, tokens, tool args |
# MAGIC | **MLflow captures** | Request / response | Full **trace** (tree of spans) |
# MAGIC
# MAGIC #### Agents need more insight
# MAGIC - Agents perform multiple intermediate steps (for example: retrieval, tool use, LLM calls), and you need to see each step, its inputs/outputs, and per-step latency/token usage to debug and improve quality.
# MAGIC - **MLflow Tracing** captures these as **traces** and **spans** automatically for supported libraries (such as the OpenAI SDK, LangChain, and other GenAI frameworks) and provides a UI and APIs to analyze them across development and production.
# MAGIC - In a **tracing system** such as OpenTelemetry and MLflow, a single operation is represented as a **span**. It records when the operation started and ended, along with the metadata, inputs, and outputs for that unit of work.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <div>
# MAGIC       <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC         MLflow spans are designed to be compatible with the
# MAGIC         <a href="https://opentelemetry.io/docs/concepts/signals/traces/" style="color: #1976d2; text-decoration: underline;">OpenTelemetry trace specification</a>, so additional information (such as token counts) is stored as key-value <em>attributes</em> on the span, rather than as new top-level fields.
# MAGIC       </p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### F2. Hierarchical Span Structure
# MAGIC
# MAGIC MLflow organizes trace data using a hierarchical span structure that mirrors agent execution, starting with a single root span that represents the overall request or workflow, with nested child spans for each sub-step.
# MAGIC
# MAGIC - **Parent spans**: High-level operations like "process user request"
# MAGIC - **Child spans**: Detailed steps like "call retrieval tool" or "generate response"
# MAGIC - **Span relationships**: Clear parent-child relationships that show execution flow, which should mimic your application's execution plan.
# MAGIC - **Span types**: Categorization of spans (`TOOL`, `CHAT_MODEL`, `RETRIEVER`) for better organization
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/binder/prod_main/building-agentic-applications-on-databricks-en_us-2.0.0/images/20261012T195948Z/Building Agentic Applications on Databricks/Includes/images/tracing-example.png" alt="tracing-example.png">
# MAGIC <p>
# MAGIC </p>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### F3. Registering MLflow Traces with Unity Catalog 
# MAGIC You can point an experiment at a catalog and schema, and every trace your app emits lands as a governed managed UC table with ready-to-query views. 
# MAGIC
# MAGIC <div class="mlt">
# MAGIC <style>
# MAGIC   .mlt{background:#ffffff;max-width:1600px;margin:0 auto;padding:24px 20px 40px;}
# MAGIC   .mlt svg{display:block;width:100%;height:auto;}
# MAGIC   .mlt svg text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;fill:#0b2026;}
# MAGIC   .mlt .h   {font-weight:800;}
# MAGIC   .mlt .b   {font-weight:700;}
# MAGIC   .mlt .m   {fill:#618794;font-weight:500;}
# MAGIC   .mlt .mono,.mlt .mono text{white-space:pre;font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-ligatures:none;font-feature-settings:"liga" 0,"calt" 0;}
# MAGIC   .mlt .pfx {fill:#FF5F46;}
# MAGIC </style>
# MAGIC <svg viewBox="0 0 1600 680" role="img" aria-label="MLflow traces registered with Unity Catalog. A GenAI app built on OpenAI, LangChain, LlamaIndex, DSPy, Anthropic or custom code is auto-instrumented by MLflow Tracing, which captures inputs and outputs, latency, token usage, tool calls, errors and span hierarchy. Binding the experiment to UnityCatalog with table_prefix oai_sa writes traces into the prod.agents schema as four Delta tables: oai_sa_otel_spans, oai_sa_otel_logs, oai_sa_otel_annotations and oai_sa_otel_metrics, plus two SQL views built over them: oai_sa_trace_unified and oai_sa_trace_metadata. The assets are governed by Unity Catalog and queried with Databricks SQL or mlflow.search_traces.">
# MAGIC   <defs>
# MAGIC     <!-- repo glyphs: product-icons/tableIcon.svg and product-icons/TableViewIcon.svg -->
# MAGIC     <symbol id="ic-table" viewBox="0 0 16 16">
# MAGIC       <path fill-rule="evenodd" clip-rule="evenodd" d="M1 1.75C1 1.33579 1.33579 1 1.75 1H14.25C14.6642 1 15 1.33579 15 1.75V14.25C15 14.6642 14.6642 15 14.25 15H1.75C1.33579 15 1 14.6642 1 14.25V1.75ZM2.5 2.5V5.5H13.5V2.5H2.5ZM2.5 13.5V7H5V13.5H2.5ZM6.5 13.5H9.5V7H6.5V13.5ZM11 7V13.5H13.5V7H11Z" fill="#0B2026"/>
# MAGIC     </symbol>
# MAGIC     <symbol id="ic-view" viewBox="0 0 16 16">
# MAGIC       <path fill-rule="evenodd" clip-rule="evenodd" d="M1.75 1C1.33579 1 1 1.33579 1 1.75V14.25C1 14.6642 1.33579 15 1.75 15H4V13.5H2.5V7H5V9H6.5V7H9.5V9H11V7H13.5V9H15V1.75C15 1.33579 14.6642 1 14.25 1H1.75ZM13.5 5.5V2.5H2.5V5.5H13.5Z" fill="#FF3621"/>
# MAGIC       <path fill-rule="evenodd" clip-rule="evenodd" d="M11.75 10C11.5511 10 11.3603 10.079 11.2197 10.2197C11.1398 10.2995 11.0798 10.3955 11.0429 10.5H9.95735C9.85439 10.2087 9.57658 10 9.25003 10L5.75 10C5.55108 10 5.36031 10.079 5.21966 10.2197C5.07901 10.3603 4.99999 10.5511 5 10.75L5.00003 12.5C5.00003 13.8807 6.11932 15 7.50003 15C8.88074 15 10 13.8807 10 12.5V12H11L11 12.5C11 13.8807 12.1193 15 13.5 15C14.8807 15 16 13.8807 16 12.5V10.7506C16 10.3364 15.6642 10 15.25 10L11.75 10ZM12.5 12.5L12.5 11.5L14.5 11.5V12.5C14.5 13.0523 14.0523 13.5 13.5 13.5C12.9477 13.5 12.5 13.0523 12.5 12.5ZM6.50002 11.5L6.50003 12.5C6.50003 13.0523 6.94774 13.5 7.50003 13.5C8.05231 13.5 8.50003 13.0523 8.50003 12.5V11.5001L6.50002 11.5Z" fill="#FF3621"/>
# MAGIC     </symbol>
# MAGIC     <marker id="ar" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#618794"/></marker>
# MAGIC     <marker id="ar-big" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#1B5162"/></marker>
# MAGIC     <filter id="sh" x="-10%" y="-10%" width="120%" height="125%"><feDropShadow dx="0" dy="4" stdDeviation="9" flood-color="#1B3139" flood-opacity="0.10"/></filter>
# MAGIC   </defs>
# MAGIC   <!-- ================= 1 · GenAI app ================= -->
# MAGIC   <rect x="40" y="20" width="320" height="620" rx="18" fill="#ffffff" stroke="#DCE0E2" stroke-width="1.5" filter="url(#sh)"/>
# MAGIC   <text class="m b" x="68" y="60" font-size="14" letter-spacing="1.6">1 · YOUR GENAI APP</text>
# MAGIC   <text class="h" x="68" y="90" font-size="22">Any LLM framework</text>
# MAGIC   <text class="m" x="68" y="116" font-size="15">one-line autologging</text>
# MAGIC   <g>
# MAGIC     <rect x="62" y="142" width="276" height="62" rx="10" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <text class="b" x="82" y="168" font-size="17">OpenAI</text>
# MAGIC     <text class="mono m" x="82" y="190" font-size="14">mlflow.openai.autolog()</text>
# MAGIC     <rect x="62" y="216" width="276" height="62" rx="10" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <text class="b" x="82" y="242" font-size="17">LangChain</text>
# MAGIC     <text class="mono m" x="82" y="264" font-size="14">mlflow.langchain.autolog()</text>
# MAGIC     <rect x="62" y="290" width="276" height="62" rx="10" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <text class="b" x="82" y="316" font-size="17">LlamaIndex</text>
# MAGIC     <text class="mono m" x="82" y="338" font-size="14">mlflow.llama_index.autolog()</text>
# MAGIC     <rect x="62" y="364" width="276" height="62" rx="10" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <text class="b" x="82" y="390" font-size="17">DSPy</text>
# MAGIC     <text class="mono m" x="82" y="412" font-size="14">mlflow.dspy.autolog()</text>
# MAGIC     <rect x="62" y="438" width="276" height="62" rx="10" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <text class="b" x="82" y="464" font-size="17">Anthropic</text>
# MAGIC     <text class="mono m" x="82" y="486" font-size="14">mlflow.anthropic.autolog()</text>
# MAGIC     <rect x="62" y="512" width="276" height="62" rx="10" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <text class="b" x="82" y="538" font-size="17">Custom code</text>
# MAGIC     <text class="mono m" x="82" y="560" font-size="14">@mlflow.trace</text>
# MAGIC   </g>
# MAGIC   <text class="m" x="200" y="612" font-size="14" text-anchor="middle">no changes to app logic</text>
# MAGIC   <!-- arrow 1 -> 2 -->
# MAGIC   <path d="M362,330 L414,330" fill="none" stroke="#618794" stroke-width="2.5" marker-end="url(#ar)"/>
# MAGIC   <!-- ================= 2 · MLflow Tracing ================= -->
# MAGIC   <rect x="420" y="20" width="320" height="620" rx="18" fill="#ffffff" stroke="#DCE0E2" stroke-width="1.5" filter="url(#sh)"/>
# MAGIC   <text class="m b" x="448" y="60" font-size="14" letter-spacing="1.6">2 · MLFLOW TRACING</text>
# MAGIC   <text class="h" x="448" y="90" font-size="22">Each trace captures</text>
# MAGIC   <g font-size="16">
# MAGIC     <circle cx="456" cy="130" r="4" fill="#4299E0"/><text x="470" y="136">Inputs &amp; outputs</text>
# MAGIC     <circle cx="456" cy="162" r="4" fill="#4299E0"/><text x="470" y="168">Latency per step</text>
# MAGIC     <circle cx="456" cy="194" r="4" fill="#4299E0"/><text x="470" y="200">Token usage</text>
# MAGIC     <circle cx="456" cy="226" r="4" fill="#4299E0"/><text x="470" y="232">Tool / function calls</text>
# MAGIC     <circle cx="456" cy="258" r="4" fill="#4299E0"/><text x="470" y="264">Errors &amp; retries</text>
# MAGIC     <circle cx="456" cy="290" r="4" fill="#4299E0"/><text x="470" y="296">Span hierarchy</text>
# MAGIC   </g>
# MAGIC   <text class="m b" x="448" y="350" font-size="14" letter-spacing="1.6">BIND TO UNITY CATALOG</text>
# MAGIC   <rect x="440" y="364" width="280" height="168" rx="10" fill="#303F47"/>
# MAGIC   <g class="mono" font-size="14">
# MAGIC     <text x="456" y="392" style="fill:#C4CCD6">mlflow.set_experiment(</text>
# MAGIC     <text x="456" y="414" style="fill:#C4CCD6">  experiment_name="…",</text>
# MAGIC     <text x="456" y="436" style="fill:#C4CCD6">  trace_location=UnityCatalog(</text>
# MAGIC     <text x="456" y="458" style="fill:#C4CCD6">    catalog_name=<tspan style="fill:#FFCC66">"prod"</tspan>,</text>
# MAGIC     <text x="456" y="480" style="fill:#C4CCD6">    schema_name=<tspan style="fill:#FFCC66">"agents"</tspan>,</text>
# MAGIC     <text x="456" y="502" style="fill:#C4CCD6">    table_prefix=<tspan style="fill:#FF8774">"oai_sa"</tspan>))</text>
# MAGIC   </g>
# MAGIC   <text class="m" x="448" y="570" font-size="14">Needs a SQL warehouse:</text>
# MAGIC   <text class="mono m" x="448" y="592" font-size="14">MLFLOW_TRACING_SQL_WAREHOUSE_ID</text>
# MAGIC   <!-- arrow 2 -> 3 -->
# MAGIC   <path d="M742,330 L794,330" fill="none" stroke="#618794" stroke-width="2.5" marker-end="url(#ar)"/>
# MAGIC   <!-- ================= 3 · Unity Catalog ================= -->
# MAGIC   <rect x="800" y="20" width="760" height="620" rx="18" fill="#ffffff" stroke="#1B5162" stroke-width="1.8" filter="url(#sh)"/>
# MAGIC   <path d="M818,20 H1542 A18,18 0 0 1 1560,38 V76 H800 V38 A18,18 0 0 1 818,20 Z" fill="#1B5162"/>
# MAGIC   <text x="826" y="56" font-size="22" font-weight="800" style="fill:#ffffff">3 · Unity Catalog</text>
# MAGIC   <text class="mono" x="1534" y="55" font-size="16" text-anchor="end" style="fill:#ffffff">prod.agents.<tspan style="fill:#FF8774">oai_sa</tspan>_*</text>
# MAGIC   <!-- tables section -->
# MAGIC   <use href="#ic-table" x="826" y="96" width="20" height="20"/>
# MAGIC   <text class="b" x="854" y="113" font-size="17">4 Delta tables</text>
# MAGIC   <text class="m" x="982" y="113" font-size="15">raw OpenTelemetry data, written as traces arrive</text>
# MAGIC   <g>
# MAGIC     <!-- spans -->
# MAGIC     <path d="M822,130 H1164 A10,10 0 0 1 1174,140 V212 A10,10 0 0 1 1164,222 H822 Z" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <rect x="822" y="130" width="5" height="92" fill="#1B5162"/>
# MAGIC     <use href="#ic-table" x="842" y="146" width="22" height="22"/>
# MAGIC     <text class="mono b" x="874" y="164" font-size="17"><tspan class="pfx">oai_sa</tspan>_otel_spans</text>
# MAGIC     <text class="m" x="842" y="192" font-size="15">One row per span: the call tree with</text>
# MAGIC     <text class="m" x="842" y="211" font-size="15">timing, inputs and outputs</text>
# MAGIC     <!-- logs -->
# MAGIC     <path d="M1186,130 H1528 A10,10 0 0 1 1538,140 V212 A10,10 0 0 1 1528,222 H1186 Z" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <rect x="1186" y="130" width="5" height="92" fill="#1B5162"/>
# MAGIC     <use href="#ic-table" x="1206" y="146" width="22" height="22"/>
# MAGIC     <text class="mono b" x="1238" y="164" font-size="17"><tspan class="pfx">oai_sa</tspan>_otel_logs</text>
# MAGIC     <text class="m" x="1206" y="192" font-size="15">OpenTelemetry log records emitted</text>
# MAGIC     <text class="m" x="1206" y="211" font-size="15">while the app runs</text>
# MAGIC     <!-- annotations -->
# MAGIC     <path d="M822,234 H1164 A10,10 0 0 1 1174,244 V316 A10,10 0 0 1 1164,326 H822 Z" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <rect x="822" y="234" width="5" height="92" fill="#1B5162"/>
# MAGIC     <use href="#ic-table" x="842" y="250" width="22" height="22"/>
# MAGIC     <text class="mono b" x="874" y="268" font-size="17"><tspan class="pfx">oai_sa</tspan>_otel_annotations</text>
# MAGIC     <text class="m" x="842" y="296" font-size="15">Assessments, feedback and tags</text>
# MAGIC     <text class="m" x="842" y="315" font-size="15">attached to traces</text>
# MAGIC     <!-- metrics -->
# MAGIC     <path d="M1186,234 H1528 A10,10 0 0 1 1538,244 V316 A10,10 0 0 1 1528,326 H1186 Z" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <rect x="1186" y="234" width="5" height="92" fill="#1B5162"/>
# MAGIC     <use href="#ic-table" x="1206" y="250" width="22" height="22"/>
# MAGIC     <text class="mono b" x="1238" y="268" font-size="17"><tspan class="pfx">oai_sa</tspan>_otel_metrics</text>
# MAGIC     <text class="m" x="1206" y="296" font-size="15">OpenTelemetry metric data points</text>
# MAGIC     <text class="m" x="1206" y="315" font-size="15">recorded alongside traces</text>
# MAGIC   </g>
# MAGIC   <!-- bus: tables -> views -->
# MAGIC   <g fill="none" stroke="#1B5162" stroke-width="2.2" stroke-linecap="round">
# MAGIC     <path d="M998,326 L998,364"/>
# MAGIC     <path d="M1362,326 L1362,364"/>
# MAGIC     <path d="M998,364 L1362,364"/>
# MAGIC     <path d="M998,364 L998,406" marker-end="url(#ar-big)"/>
# MAGIC     <path d="M1362,364 L1362,406" marker-end="url(#ar-big)"/>
# MAGIC   </g>
# MAGIC   <rect x="1056" y="350" width="252" height="28" rx="14" fill="#ffffff" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <use href="#ic-view" x="1068" y="356" width="16" height="16"/>
# MAGIC   <text class="b" x="1196" y="370" font-size="15" text-anchor="middle">2 SQL views over the tables</text>
# MAGIC   <!-- views -->
# MAGIC   <g>
# MAGIC     <path d="M822,412 H1164 A10,10 0 0 1 1174,422 V526 A10,10 0 0 1 1164,536 H822 Z" fill="#FFF6F4" stroke="#FABFBA"/>
# MAGIC     <rect x="822" y="412" width="5" height="124" fill="#FF5F46"/>
# MAGIC     <use href="#ic-view" x="842" y="428" width="22" height="22"/>
# MAGIC     <text class="mono b" x="874" y="446" font-size="17"><tspan class="pfx">oai_sa</tspan>_trace_unified</text>
# MAGIC     <text class="m" x="842" y="476" font-size="15">One row per trace: spans, metadata</text>
# MAGIC     <text class="m" x="842" y="495" font-size="15">and assessments together</text>
# MAGIC     <rect x="842" y="506" width="120" height="24" rx="12" fill="#FF5F46"/>
# MAGIC     <text x="902" y="523" font-size="14" font-weight="700" text-anchor="middle" style="fill:#ffffff">FULL DETAIL</text>
# MAGIC     <path d="M1186,412 H1528 A10,10 0 0 1 1538,422 V526 A10,10 0 0 1 1528,536 H1186 Z" fill="#FFF6F4" stroke="#FABFBA"/>
# MAGIC     <rect x="1186" y="412" width="5" height="124" fill="#FF5F46"/>
# MAGIC     <use href="#ic-view" x="1206" y="428" width="22" height="22"/>
# MAGIC     <text class="mono b" x="1238" y="446" font-size="17"><tspan class="pfx">oai_sa</tspan>_trace_metadata</text>
# MAGIC     <text class="m" x="1206" y="476" font-size="15">Tags, metadata and assessments only,</text>
# MAGIC     <text class="m" x="1206" y="495" font-size="15">no span payloads</text>
# MAGIC     <rect x="1206" y="506" width="156" height="24" rx="12" fill="#FF5F46"/>
# MAGIC     <text x="1284" y="523" font-size="14" font-weight="700" text-anchor="middle" style="fill:#ffffff">FAST FILTERING</text>
# MAGIC   </g>
# MAGIC   <!-- what you get -->
# MAGIC   <line x1="822" y1="556" x2="1538" y2="556" stroke="#EEEDE9" stroke-width="1.5"/>
# MAGIC   <g font-size="15">
# MAGIC     <rect x="822" y="574" width="230" height="44" rx="22" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <text x="937" y="601" text-anchor="middle"><tspan class="b">Govern</tspan><tspan class="m">: grants · lineage · audit</tspan></text>
# MAGIC     <rect x="1064" y="574" width="220" height="44" rx="22" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <text x="1174" y="601" text-anchor="middle"><tspan class="b">Query</tspan><tspan class="m">: Databricks SQL</tspan></text>
# MAGIC     <rect x="1296" y="574" width="242" height="44" rx="22" fill="#F9F7F4" stroke="#EEEDE9"/>
# MAGIC     <text x="1417" y="601" text-anchor="middle"><tspan class="b">Read</tspan><tspan class="m">: </tspan><tspan class="mono m" font-size="14">mlflow.search_traces()</tspan></text>
# MAGIC   </g>
# MAGIC </svg>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## G. Conclusion
# MAGIC
# MAGIC In this lecture, you learned the building blocks for agent development on Databricks:
# MAGIC
# MAGIC - **Orchestration patterns**: handoffs, agents-as-tools, sequential, parallel, and feedback loops
# MAGIC - **Rapid prototyping**: `@function_tool`, `@tool`, and plain DSPy functions let you try a tool before registering it to Unity Catalog
# MAGIC - **MCP integration**: `McpServer` connects agents to governed UC functions at runtime
# MAGIC - **Multi-agent concepts**: supervisor routing, worker specialization, and MCP-scoped tools
# MAGIC - **Other frameworks**: LangChain, LangGraph, and DSPy as alternative authoring options
# MAGIC - **MLflow tracing**: hierarchical spans that capture every step of agent execution
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>