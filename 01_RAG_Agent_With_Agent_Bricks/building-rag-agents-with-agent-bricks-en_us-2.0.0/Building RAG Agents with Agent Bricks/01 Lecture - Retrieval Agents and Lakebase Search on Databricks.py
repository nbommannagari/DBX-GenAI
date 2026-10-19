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
# MAGIC # Lecture - Retrieval Agents and Lakebase Search on Databricks
# MAGIC #### Duration: 30 minutes
# MAGIC ## Overview
# MAGIC This lecture introduces retrieval agents and AI engineering on Databricks. We begin with why prompting alone isn't enough and how Retrieval Augmented Generation (RAG) fills the gap. We then trace the evolution from prompt engineering to context engineering to harness engineering, and survey Databricks' retrieval approaches — from managed Genie Agents (recommended) and Knowledge Assistants (legacy), to fully custom code-first agents. Finally, we introduce Lakebase, Databricks' managed Postgres database, and Lakebase Search, which brings hybrid vector and keyword search directly into your operational database for co-located retrieval.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lecture, you will be able to:
# MAGIC 1. Explain what retrieval agents are and why they are needed
# MAGIC 2. Distinguish prompt engineering, context engineering, and harness engineering
# MAGIC 3. Describe Genie Agents as the simplest managed retrieval solution on Databricks
# MAGIC 4. Identify the knowledge source types available to a Knowledge Assistant
# MAGIC 5. Compare declarative vs. code-first approaches to building retrieval agents
# MAGIC 6. Explain what Lakebase is and its key capabilities for AI applications
# MAGIC 7. Describe Lakebase Search and when to use it vs. AI Search

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Introduction to Retrieval Agents

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. Why Retrieval Agents?
# MAGIC
# MAGIC Large Language Models (LLMs) are powerful, but they have hard limits. No amount of prompt refinement can overcome:
# MAGIC <br>
# MAGIC </br>
# MAGIC
# MAGIC <div class="cw">
# MAGIC <style>
# MAGIC .cw { font-family: sans-serif; max-width: 1100px; margin: 0 auto; color: #0b2026; }
# MAGIC .cw * { box-sizing: border-box; }
# MAGIC .ds-heading { font-size: 22pt; font-weight: 700; color: #0b2026; line-height: 1.2; margin: 0 0 8px; }
# MAGIC .ds-subhead { font-size: 15pt; color: #5E7077; line-height: 1.45; margin: 0 0 24px; }
# MAGIC .ds-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; align-items: stretch; }
# MAGIC .ds-row > * { min-width: 0; }
# MAGIC .ds-card { background: #F9F7F4; border-radius: 8px; box-shadow: 0 4px 16px rgba(27,49,57,0.13); padding: 24px 20px; border-top: 8px solid #FFAB00; }
# MAGIC .ds-card-title { font-size: 18pt; font-weight: 700; line-height: 1.25; margin: 0 0 10px; }
# MAGIC .ds-card-text { font-size: 14pt; color: #5E7077; line-height: 1.55; margin: 0; }
# MAGIC .tc { text-align: center; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="ds-row">
# MAGIC   <div class="ds-card tc">
# MAGIC     <div class="ds-card-title">Knowledge Cutoffs</div>
# MAGIC     <div class="ds-card-text">The model doesn't know about events after its training date.</div>
# MAGIC   </div>
# MAGIC   <div class="ds-card tc">
# MAGIC     <div class="ds-card-title">Hallucination</div>
# MAGIC     <div class="ds-card-text">The model fabricates plausible-sounding facts when it lacks real data.</div>
# MAGIC   </div>
# MAGIC   <div class="ds-card tc">
# MAGIC     <div class="ds-card-title">Missing Private Context</div>
# MAGIC     <div class="ds-card-text">No access to your organization's proprietary documents.</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <br>
# MAGIC </br>
# MAGIC
# MAGIC **Retrieval Augmented Generation (RAG)** addresses these limits by injecting relevant external data into the model's context at query time. 
# MAGIC
# MAGIC Instead of relying on frozen training data, a RAG system **retrieves** relevant documents, **augments** the prompt with them, and lets the model **generate** a grounded response.
# MAGIC
# MAGIC A **retrieval agent** is a concrete implementation of this pattern that handles query routing, retrieval orchestration, and context assembly within a real system.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 680 260" role="img" style="font-family: sans-serif;">
# MAGIC   <title>RAG Pipeline</title>
# MAGIC   <desc>User query flows through Retrieve, Augment, and Generate stages to produce a grounded response.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="rag-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC
# MAGIC   <!-- User Query -->
# MAGIC   <rect x="15" y="90" width="120" height="60" rx="8" fill="#ffffff" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="75" y="124" text-anchor="middle" font-size="18" font-weight="600" fill="#0b2026">User Query</text>
# MAGIC
# MAGIC   <!-- Arrow 1 -->
# MAGIC   <path d="M135 120 L175 120" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#rag-arrow)"/>
# MAGIC
# MAGIC   <!-- Retrieve -->
# MAGIC   <rect x="180" y="90" width="130" height="60" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="245" y="116" text-anchor="middle" font-size="18" font-weight="600" fill="#0b2026">Retrieve</text>
# MAGIC   <text x="245" y="134" text-anchor="middle" font-size="15" fill="#618794">Search</text>
# MAGIC
# MAGIC   <!-- Arrow 2 -->
# MAGIC   <path d="M310 120 L350 120" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#rag-arrow)"/>
# MAGIC
# MAGIC   <!-- Augment -->
# MAGIC   <rect x="355" y="90" width="130" height="60" rx="8" fill="#4299E0" fill-opacity="0.12" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="420" y="116" text-anchor="middle" font-size="18" font-weight="600" fill="#0b2026">Augment</text>
# MAGIC   <text x="420" y="134" text-anchor="middle" font-size="15" fill="#618794">Inject context</text>
# MAGIC
# MAGIC   <!-- Arrow 3 -->
# MAGIC   <path d="M485 120 L525 120" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#rag-arrow)"/>
# MAGIC
# MAGIC   <!-- Generate -->
# MAGIC   <rect x="530" y="90" width="130" height="60" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="595" y="116" text-anchor="middle" font-size="18" font-weight="600" fill="#0b2026">Generate</text>
# MAGIC   <text x="595" y="134" text-anchor="middle" font-size="15" fill="#618794">LLM response</text>
# MAGIC
# MAGIC   <!-- Knowledge Base below Retrieve -->
# MAGIC   <rect x="155" y="180" width="180" height="50" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.2"/>
# MAGIC   <text x="245" y="204" text-anchor="middle" font-size="16" fill="#0b2026">Knowledge Base</text>
# MAGIC   <text x="245" y="220" text-anchor="middle" font-size="13" fill="#618794">UC securables &amp; Lakebase</text>
# MAGIC   <path d="M245 180 L245 150" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#rag-arrow)"/>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Learn More</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC See the Databricks documentation on Retrieval Augmented Generation (<a href="https://docs.databricks.com/aws/en/generative-ai/retrieval-augmented-generation#what-is-retrieval-augmented-generation" style="color: #1976d2;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation#what-is-retrieval-augmented-generation" style="color: #1976d2;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/generative-ai/retrieval-augmented-generation#what-is-retrieval-augmented-generation" style="color: #1976d2;">GCP</a>) for a deeper dive.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### A2. The Common Retrieval Pipeline
# MAGIC
# MAGIC Regardless of the framework or product you choose, every RAG system follows the same four-stage workflow. 
# MAGIC
# MAGIC Understanding these stages helps you evaluate managed solutions and know what you're building when you go code-first.
# MAGIC
# MAGIC <svg width="100%" viewBox="0 0 1100 340" preserveAspectRatio="xMidYMin meet" style="display:block; margin:16px auto; max-width:1100px;"><foreignObject x="0" y="0" width="1100" height="340"><div xmlns="http://www.w3.org/1999/xhtml" style="display:flex; align-items:flex-start; gap:10px; flex-wrap:nowrap; justify-content:center; width:1100px; height:340px; box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
# MAGIC
# MAGIC   <!-- Box 1: Indexing -->
# MAGIC   <div style="border: 1.5px solid #618794; border-radius: 12px; padding: 16px; width: 200px; flex-shrink: 0; background: #ffffff;">
# MAGIC     <div style="font-size: 18px; font-weight: 700; color: #0b2026; text-align: center;">Indexing</div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 8px 10px; margin-top: 14px;">
# MAGIC       <div style="font-size: 14px; color: #0b2026; font-weight: 600;">Ingest &amp; Parse</div>
# MAGIC       <div style="font-size: 12px; color: #618794; margin-top: 2px; line-height: 1.35;">Documents → chunks + metadata</div>
# MAGIC     </div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 8px 10px; margin-top: 6px;">
# MAGIC       <div style="font-size: 14px; color: #0b2026; font-weight: 600;">Embed &amp; Index</div>
# MAGIC       <div style="font-size: 12px; color: #618794; margin-top: 2px; line-height: 1.35;">Embedding model → vector search index</div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC
# MAGIC   <!-- Arrow -->
# MAGIC   <div style="padding-top: 50px; flex-shrink: 0;">
# MAGIC     <div style="font-size: 22px; color: #1B3139;">&#10132;</div>
# MAGIC   </div>
# MAGIC
# MAGIC   <!-- Box 2: Retrieval -->
# MAGIC   <div style="border: 1.5px solid #2272B4; border-radius: 12px; padding: 16px; width: 200px; flex-shrink: 0; background: #ffffff;">
# MAGIC     <div style="font-size: 18px; font-weight: 700; color: #0b2026; text-align: center;">Retrieval</div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 8px 10px; margin-top: 14px;">
# MAGIC       <div style="font-size: 14px; color: #0b2026; font-weight: 600;">Receive Query</div>
# MAGIC       <div style="font-size: 12px; color: #618794; margin-top: 2px; line-height: 1.35;">User question hits the system</div>
# MAGIC     </div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 8px 10px; margin-top: 6px;">
# MAGIC       <div style="font-size: 14px; color: #0b2026; font-weight: 600;">Plan &amp; Route</div>
# MAGIC       <div style="font-size: 12px; color: #618794; margin-top: 2px; line-height: 1.35;">Select sources → tailored sub-queries</div>
# MAGIC     </div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 8px 10px; margin-top: 6px;">
# MAGIC       <div style="font-size: 14px; color: #0b2026; font-weight: 600;">Search &amp; Rank</div>
# MAGIC       <div style="font-size: 12px; color: #618794; margin-top: 2px; line-height: 1.35;">Vector search → rerank → evidence set</div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC
# MAGIC   <!-- Arrow -->
# MAGIC   <div style="padding-top: 50px; flex-shrink: 0;">
# MAGIC     <div style="font-size: 22px; color: #1B3139;">&#10132;</div>
# MAGIC   </div>
# MAGIC
# MAGIC   <!-- Box 3: Generation -->
# MAGIC   <div style="border: 2px solid #FF5F46; border-radius: 12px; padding: 16px; width: 200px; flex-shrink: 0; background: #F9F7F4;">
# MAGIC     <div style="font-size: 18px; font-weight: 700; color: #FF5F46; text-align: center;">Generation</div>
# MAGIC     <div style="border: 1px solid #FF5F46; border-radius: 6px; background: #ffffff; padding: 10px; margin-top: 14px;">
# MAGIC       <div style="font-size: 14px; color: #0b2026; font-weight: 600;">Generate Answer</div>
# MAGIC       <div style="font-size: 12px; color: #618794; margin-top: 2px; line-height: 1.35;">LLM grounds on retrieved chunks → cited answer</div>
# MAGIC       <div style="font-size: 11px; color: #618794; margin-top: 6px; font-style: italic;">with source citations</div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC
# MAGIC   <!-- Arrow -->
# MAGIC   <div style="padding-top: 50px; flex-shrink: 0;">
# MAGIC     <div style="font-size: 22px; color: #1B3139;">&#10132;</div>
# MAGIC   </div>
# MAGIC
# MAGIC   <!-- Box 4: Feedback Loop -->
# MAGIC   <div style="border: 1.5px dashed #618794; border-radius: 12px; padding: 16px; width: 200px; flex-shrink: 0; background: #ffffff;">
# MAGIC     <div style="font-size: 18px; font-weight: 700; color: #0b2026; text-align: center;">Feedback Loop</div>
# MAGIC     <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 8px 10px; margin-top: 14px;">
# MAGIC       <div style="font-size: 14px; color: #0b2026; font-weight: 600;">Observe &amp; Improve</div>
# MAGIC       <div style="font-size: 12px; color: #618794; margin-top: 2px; line-height: 1.35;">Traces + human feedback → refine retrieval &amp; generation</div>
# MAGIC       <div style="font-size: 11px; color: #FF5F46; margin-top: 6px; font-weight: 600;">continuous quality loop</div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div></foreignObject></svg>
# MAGIC
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC What are the four stages of a retrieval pipeline?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p style="margin-top: 10px;">Every RAG system — whether managed or custom — flows through four stages: two offline (preparing the index) and two online (handling the request), with a feedback loop that improves quality over time.</p>
# MAGIC
# MAGIC <ol style="margin: 12px 0; padding-left: 24px;">
# MAGIC <li><strong>Indexing</strong>: Ingest source documents, <strong>parse, chunk, embed, and index</strong> them into a searchable store (plus normalize metadata).</li>
# MAGIC <li><strong>Retrieval</strong>: Receive the user query, <strong>plan retrieval</strong> by selecting sources and crafting sub-queries, then <strong>search and rank</strong> results from one or more indexes.</li>
# MAGIC <li><strong>Generation</strong>: Use an LLM to <strong>generate a grounded answer</strong> from the retrieved context, including <strong>source citations</strong> back to the original documents.</li>
# MAGIC <li><strong>Feedback loop</strong>: Use tracing, evaluation (LLM judges, human review), and <strong>user feedback</strong> to continuously refine both retrieval behavior and answer quality.</li>
# MAGIC </ol>
# MAGIC
# MAGIC <p>In Section C, you'll see how <strong>Genie Agents</strong> and <strong>Knowledge Assistants</strong> handle these stages for you, and how <strong>custom agents</strong> let you own each stage directly.</p>
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Introduction to AI Engineering

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### B1. From Prompt Engineering to Context Engineering to Harness Engineering
# MAGIC
# MAGIC <div id="pche-wrap" style="position: relative; width: 100%; margin: 16px 0; font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;">
# MAGIC   <div style="display: flex; align-items: center; justify-content: space-between; gap: 0; flex-wrap: nowrap; width: 100%;">
# MAGIC     <!-- Prompt Engineering (standalone) -->
# MAGIC     <div id="pche-src1" style="border: 1.5px solid #618794; border-radius: 12px; padding: 20px 24px; flex: 3 1 0; min-width: 0; background: #ffffff;">
# MAGIC       <div style="font-size: 19px; font-weight: 700; color: #0b2026; text-align: center;">Prompt Engineering</div>
# MAGIC       <div style="font-size: 16px; color: #618794; text-align: center; margin-top: 6px;">The instruction text</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 14px; font-size: 15px; color: #0b2026;">System instructions</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Few-shot examples</div>
# MAGIC       <div style="border: 1px dashed #2272B4; border-radius: 6px; background: rgba(34,114,180,0.08); padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">User prompt</div>
# MAGIC     </div>
# MAGIC     <!-- Gap 1 (label) -->
# MAGIC     <div style="align-self: stretch; flex: 2 1 0; min-width: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; padding-top: 56px;">
# MAGIC       <div style="font-size: 15px; color: #618794; text-align: center; line-height: 1.4; font-weight: 700;">Manage the<br/>entire context<br/>state</div>
# MAGIC     </div>
# MAGIC     <!-- Context Engineering (standalone, contains embedded Prompt) -->
# MAGIC     <div id="pche-src2" style="border: 2px solid #FF5F46; border-radius: 12px; padding: 20px 22px; flex: 4 1 0; min-width: 0; background: #F9F7F4;">
# MAGIC       <div style="font-size: 19px; font-weight: 700; color: #FF5F46; text-align: center;">Context Engineering</div>
# MAGIC       <div style="font-size: 16px; color: #618794; text-align: center; margin-top: 4px;">The entire input environment</div>
# MAGIC       <div style="font-size: 14px; color: #0b2026; text-align: center; margin-top: 4px; font-style: italic;">"What does the model see?"</div>
# MAGIC       <!-- embedded prompt engineering (subset) -->
# MAGIC       <div id="pche-tgt1" style="border: 1.5px dashed #618794; border-radius: 8px; background: rgba(97,135,148,0.10); padding: 10px 10px; margin-top: 14px;">
# MAGIC         <div style="font-size: 15px; font-weight: 700; color: #0b2026; text-align: center;">Prompt Engineering</div>
# MAGIC         <div style="font-size: 14px; color: #0b2026; text-align: center; margin-top: 4px;">system instructions · few-shot · user prompt</div>
# MAGIC       </div>
# MAGIC       <div style="text-align: center; color: #618794; font-size: 18px; font-weight: 700; margin: 8px 0 6px 0; line-height: 1;">+</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Retrieval / RAG (AI + Vector Search)</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Memory across sessions (Lakebase)</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Tool context (MCP + UC functions)</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Governance (Unity Catalog)</div>
# MAGIC     </div>
# MAGIC     <!-- Gap 2 (label) -->
# MAGIC     <div style="align-self: stretch; flex: 2 1 0; min-width: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; padding-top: 56px;">
# MAGIC       <div style="font-size: 15px; color: #FF5F46; text-align: center; line-height: 1.4; font-weight: 700;">Scaffold the<br/>runtime around<br/>the model</div>
# MAGIC     </div>
# MAGIC     <!-- Harness Engineering (contains embedded Context Engineering + harness-specific components) -->
# MAGIC     <div style="border: 2px solid #1B5162; border-radius: 12px; padding: 20px 20px; flex: 5.4 1 0; min-width: 0; background: #F8F9FC;">
# MAGIC       <div style="font-size: 19px; font-weight: 700; color: #1B5162; text-align: center;">Harness Engineering</div>
# MAGIC       <div style="font-size: 16px; color: #618794; text-align: center; margin-top: 4px;">The runtime loop + scaffolding around the model</div>
# MAGIC       <div style="font-size: 14px; color: #0b2026; text-align: center; margin-top: 4px; font-style: italic;">"How does the system operate?"</div>
# MAGIC       <!-- embedded context engineering (nested) -->
# MAGIC       <div id="pche-tgt2" style="border: 1.5px dashed #FF5F46; border-radius: 8px; background: rgba(255,95,70,0.07); padding: 10px 10px; margin-top: 14px;">
# MAGIC         <div style="font-size: 15px; font-weight: 700; color: #FF5F46; text-align: center;">Context Engineering</div>
# MAGIC         <div style="font-size: 14px; color: #0b2026; text-align: center; margin-top: 4px;">prompt · retrieval · memory · tools · governance</div>
# MAGIC       </div>
# MAGIC       <div style="text-align: center; color: #618794; font-size: 18px; font-weight: 700; margin: 8px 0 6px 0; line-height: 1;">+</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Sandboxes &amp; execution environments (Serverless compute)</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Filesystem &amp; durable storage (UC Volumes + Lakebase)</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Feedback loops &amp; self-verification (MLflow / Agent Evaluation)</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Guardrails &amp; human-in-the-loop (Unity Gateway)</div>
# MAGIC       <div style="border: 1px solid #618794; border-radius: 6px; background: #ffffff; padding: 6px 10px; text-align: center; margin-top: 6px; font-size: 15px; color: #0b2026;">Observability &amp; logging (MLflow Tracing + Inference Tables)</div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <!-- connector overlay: measured right-middle -> embedded left-middle -->
# MAGIC   <svg id="pche-svg" style="position: absolute; top: 0; left: 0; pointer-events: none; overflow: visible; z-index: 0;">
# MAGIC     <defs>
# MAGIC       <marker id="pche-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7.5" markerHeight="7.5" orient="auto">
# MAGIC         <path d="M2 1 L8 5 L2 9" fill="none" stroke="#618794" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC       </marker>
# MAGIC     </defs>
# MAGIC     <path id="pche-path1" fill="none" stroke="#618794" stroke-width="1.4" stroke-dasharray="6 4" stroke-linecap="round" />
# MAGIC     <path id="pche-path2" fill="none" stroke="#FF5F46" stroke-width="1.4" stroke-dasharray="6 4" stroke-linecap="round" />
# MAGIC   </svg>
# MAGIC </div>
# MAGIC <script>
# MAGIC (function(){
# MAGIC   function draw(){
# MAGIC     var wrap=document.getElementById('pche-wrap');
# MAGIC     var svg=document.getElementById('pche-svg');
# MAGIC     if(!wrap||!svg){return;}
# MAGIC     var wr=wrap.getBoundingClientRect();
# MAGIC     svg.setAttribute('width',wr.width);
# MAGIC     svg.setAttribute('height',wr.height);
# MAGIC     svg.setAttribute('viewBox','0 0 '+wr.width+' '+wr.height);
# MAGIC     var pairs=[['pche-src1','pche-tgt1','pche-path1'],['pche-src2','pche-tgt2','pche-path2']];
# MAGIC     for(var i=0;i<pairs.length;i++){
# MAGIC       var s=document.getElementById(pairs[i][0]);
# MAGIC       var t=document.getElementById(pairs[i][1]);
# MAGIC       var p=document.getElementById(pairs[i][2]);
# MAGIC       if(!s||!t||!p){continue;}
# MAGIC       var sr=s.getBoundingClientRect(), tr=t.getBoundingClientRect();
# MAGIC       var x1=sr.right-wr.left, y1=sr.top-wr.top+sr.height/2;
# MAGIC       var x2=tr.left-wr.left,  y2=tr.top-wr.top+tr.height/2;
# MAGIC       var dx=(x2-x1)*0.5;
# MAGIC       var d='M '+x1+' '+y1+' C '+(x1+dx)+' '+y1+' '+(x2-dx)+' '+y2+' '+x2+' '+y2;
# MAGIC       p.setAttribute('d',d);
# MAGIC     }
# MAGIC   }
# MAGIC   function schedule(){ (window.requestAnimationFrame||function(f){setTimeout(f,16);})(draw); }
# MAGIC   if(document.readyState!=='loading'){ schedule(); } else { document.addEventListener('DOMContentLoaded',schedule); }
# MAGIC   window.addEventListener('load',schedule);
# MAGIC   window.addEventListener('resize',schedule);
# MAGIC   if(document.fonts&&document.fonts.ready){ document.fonts.ready.then(schedule); }
# MAGIC   setTimeout(schedule,300); setTimeout(schedule,800);
# MAGIC })();
# MAGIC </script>
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC What's the difference between prompt, context, and harness engineering?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p><strong>Prompt engineering</strong> focuses on crafting the instruction text sent to a model. It involves choosing the right words, formatting, and examples. It's tactical and operates at the level of a single query.</p>
# MAGIC
# MAGIC <p><strong>Context engineering</strong> is broader. It's the art and science of designing the <em>entire information environment</em> a model receives at inference time: not just the question, but all the supporting data, instructions, history, and constraints that shape the model's decision. It answers one question: <strong>"What does the model see?"</strong></p>
# MAGIC
# MAGIC <p><strong>Harness engineering</strong> is broader still. It's the <em>entire scaffolding around the model</em> — the runtime loop that executes each conversation turn, the source of truth codified in-repo, the deterministic guardrails, and the workflow automation that make an agent reliable in production. It answers a different question: <strong>"How does the system around the model operate?"</strong> Context engineering is one component of it — the Foundry framing calls this <strong>"Context Engineering++"</strong>: context engineering <em>plus</em> workflow engineering, spec-driven development, and deterministic enforcement of invariants.</p>
# MAGIC
# MAGIC <p>Think of it this way: prompt engineering is writing a good question on an exam. Context engineering is designing the entire exam room — the reference materials on the desk, the instructions on the board, the rules about what's allowed. Harness engineering is running the whole institution around it — writing the rules, invigilating, grading, and re-running the exam reliably every term. Context engineering makes the agent <em>smart</em>; harness engineering makes the agent <em>reliable and repeatable</em>.</p>
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Retrieval Approaches on Databricks
# MAGIC
# MAGIC Databricks offers a spectrum of retrieval approaches — from fully managed to fully custom. **Genie Agents are the recommended managed path for RAG**; Knowledge Assistants remain available but are a legacy option. The code-first path gives you complete control.
# MAGIC
# MAGIC <div style="display: flex; gap: 28px; flex-wrap: wrap; justify-content: center; margin: 20px 0;">
# MAGIC
# MAGIC   <div style="border: 2px solid #C9CDD2; border-radius: 12px; padding: 20px 24px; width: 320px; background: #F0F2F4; box-shadow: 0 4px 16px rgba(97,107,117,0.08); position: relative;">
# MAGIC     <div style="position: absolute; top: -12px; right: 16px; background: #5A6F77; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 12px; border-radius: 10px;">LEGACY</div>
# MAGIC     <div style="font-size: 12px; font-weight: 700; color: #5A6F77; letter-spacing: 0.08em; text-transform: uppercase;">Managed RAG</div>
# MAGIC     <div style="font-size: 19px; font-weight: 700; color: #0b2026; margin-top: 4px;">Knowledge Assistant</div>
# MAGIC     <div style="font-size: 15px; color: #618794; margin-top: 8px; line-height: 1.5;">Bring your own AI Search index, choose your chunking strategy, and attach multiple source types (up to 10). Full control over the retrieval pipeline.</div>
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="border: 2px solid #00A972; border-radius: 12px; padding: 20px 24px; width: 320px; background: #E6F5EE; box-shadow: 0 4px 16px rgba(0,169,114,0.12); position: relative;">
# MAGIC     <div style="position: absolute; top: -12px; right: 16px; background: #00A972; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 12px; border-radius: 10px;">RECOMMENDED</div>
# MAGIC     <div style="font-size: 12px; font-weight: 700; color: #00A972; letter-spacing: 0.08em; text-transform: uppercase;">Managed RAG</div>
# MAGIC     <div style="font-size: 19px; font-weight: 700; color: #0b2026; margin-top: 4px;">Genie Agent</div>
# MAGIC     <div style="font-size: 15px; color: #618794; margin-top: 8px; line-height: 1.5;">Attach a Unity Catalog volume. Databricks handles parsing, chunking, indexing, and search via Lakebase — you write zero pipeline code.</div>
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="border: 2px dashed #00A972; border-radius: 12px; padding: 20px 24px; width: 320px; background: #E6F5EE; box-shadow: 0 4px 16px rgba(0,169,114,0.12); position: relative;">
# MAGIC     <div style="position: absolute; top: -12px; right: 16px; background: #00A972; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; padding: 3px 12px; border-radius: 10px;">RECOMMENDED</div>
# MAGIC     <div style="font-size: 12px; font-weight: 700; color: #00A972; letter-spacing: 0.08em; text-transform: uppercase;">Full Control</div>
# MAGIC     <div style="font-size: 19px; font-weight: 700; color: #0b2026; margin-top: 4px;">Custom Agent</div>
# MAGIC     <div style="font-size: 15px; color: #618794; margin-top: 8px; line-height: 1.5;">Write your own retrieval logic, prompts, tools, and orchestration. Use LangGraph, custom Python, or any framework — essentially unlimited flexibility.</div>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC **Genie Agents** are the go-forward declarative option: you declare *what* you want retrieved, and the system optimizes *how*. 
# MAGIC
# MAGIC **Knowledge Assistants** offer additional customization (bring your own AI Search index) but are a **legacy path** — Genie Agents are recommended for new projects. 
# MAGIC
# MAGIC **Custom agents** are code-first: you own the retrieval logic end to end.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. Genie Agents: Attach a Volume, Get Search
# MAGIC
# MAGIC A **Genie Agent** is the fastest managed path for retrieval on Databricks. In the simplest case, you attach a Unity Catalog volume containing your documents and Databricks handles the rest. But the more complete picture is that a Genie Agent sits on top of **Unity Catalog governed data**: structured data works in both Chat and Agent mode, while unstructured volume files are available in Agent mode.
# MAGIC
# MAGIC <div style="max-width: 1200px; margin: 20px auto;">
# MAGIC <svg width="100%" viewBox="0 0 1200 660" role="img" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
# MAGIC   <title>Inside the Genie Agent</title>
# MAGIC   <defs>
# MAGIC     <filter id="genieGlowC1" x="-40%" y="-40%" width="180%" height="180%">
# MAGIC       <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#FF5F46" flood-opacity="0.22"/>
# MAGIC     </filter>
# MAGIC   </defs>
# MAGIC
# MAGIC
# MAGIC   <!-- User -->
# MAGIC   <rect x="89" y="20" width="190" height="64" rx="14" fill="#FFFFFF" stroke="#618794" stroke-width="2"/>
# MAGIC   <circle cx="139" cy="44" r="10" fill="none" stroke="#618794" stroke-width="2.4"/>
# MAGIC   <path d="M122 72c0-9.4 7.6-17 17-17s17 7.6 17 17" fill="none" stroke="#618794" stroke-width="2.4" stroke-linecap="round"/>
# MAGIC   <text x="165" y="46" font-size="18" font-weight="700" fill="#0b2026">User</text>
# MAGIC   <text x="165" y="68" font-size="15" fill="#5A6F77">session user</text>
# MAGIC
# MAGIC   <!-- User ↔ Genie (bidirectional) -->
# MAGIC   <line x1="184" y1="97" x2="184" y2="135" stroke="#618794" stroke-width="2.6"/>
# MAGIC   <polygon points="184,148 177,135 191,135" fill="#618794"/>
# MAGIC   <polygon points="184,84 177,97 191,97" fill="#618794"/>
# MAGIC   <text x="171" y="120" font-size="13" fill="#5A6F77" font-style="italic" text-anchor="end">asks a question</text>
# MAGIC   <text x="197" y="120" font-size="13" fill="#5A6F77" font-style="italic">result with cited source</text>
# MAGIC
# MAGIC   <!-- Genie Agent -->
# MAGIC   <rect x="44" y="148" width="280" height="98" rx="14" fill="#F9F7F4" stroke="#FF5F46" stroke-width="3" filter="url(#genieGlowC1)"/>
# MAGIC   <g transform="translate(65,163) scale(0.34)">
# MAGIC     <path d="M179.614 80.7871C167.525 100.208 149.376 126.933 141.272 133.79C133.167 140.647 121.149 153.009 93.2907 153.009C73.7741 153.009 56.9864 140.474 49.5352 122.498C49.5353 122.519 49.5356 122.54 49.5358 122.561C48.4799 119.938 45.9132 118.085 42.9121 118.085C38.969 118.085 35.7721 121.282 35.7721 125.225C35.7722 126.15 35.9484 126.997 36.2689 127.752C37.984 131.793 42.5469 131.965 42.5469 144.569C42.5469 143.103 42.5469 146.404 42.5469 144.569C30.6947 144.569 21.0865 134.96 21.0863 123.108C21.0863 111.256 30.6945 101.647 42.5469 101.647H45.4249V101.644H79.5153C82.062 101.644 84.5067 102.722 86.0978 104.71C88.8856 108.194 91.0222 111.438 92.5969 114.114C94.056 116.594 98.4255 116.618 99.8699 114.13C101.393 111.506 103.416 108.337 105.98 104.929C107.573 102.813 110.103 101.644 112.752 101.644H125.427C143.951 101.644 154.105 92.7623 159.289 86.3612C161.972 83.049 165.782 80.7871 170.045 80.7871H179.614Z" fill="#FABFBA"/>
# MAGIC     <path d="M96.3843 111.768C96.3842 87.8629 77.3194 68.4739 53.768 68.3843C77.3196 68.2949 96.3845 48.9049 96.3845 25C96.3845 25 96.3845 25 96.3845 25C96.3845 48.9602 115.538 68.3838 139.165 68.3838C115.538 68.3838 96.3844 87.8075 96.3843 111.768Z" fill="#FF5F46"/>
# MAGIC     <path d="M81.7909 159.789C75.0244 159.789 69.5392 166.599 69.5392 175H126.368C126.368 166.599 120.883 159.789 114.116 159.789H81.7909Z" fill="#FF5F46"/>
# MAGIC   </g>
# MAGIC   <foreignObject x="132" y="148" width="186" height="98">
# MAGIC     <div xmlns="http://www.w3.org/1999/xhtml" style="box-sizing:border-box;width:100%;height:100%;display:flex;flex-direction:column;align-items:flex-start;justify-content:center;padding:6px 8px;font-family:sans-serif;">
# MAGIC       <div style="font-size:18px;font-weight:800;color:#0b2026;margin-bottom:4px;">Genie Agent</div>
# MAGIC       <div style="font-size:13px;line-height:1.3;color:#5A6F77;">Natural-language interface over governed data</div>
# MAGIC     </div>
# MAGIC   </foreignObject>
# MAGIC
# MAGIC   <!-- Genie ↔ Unity Catalog (bidirectional) -->
# MAGIC   <path d="M184 259 V393" fill="none" stroke="#0b2026" stroke-width="2.4" stroke-linecap="round"/>
# MAGIC   <polygon points="184,406 176,393 192,393" fill="#0b2026"/>
# MAGIC   <polygon points="184,246 176,259 192,259" fill="#0b2026"/>
# MAGIC   <foreignObject x="29" y="306" width="310" height="40">
# MAGIC     <div xmlns="http://www.w3.org/1999/xhtml" style="box-sizing:border-box;width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#FFFFFF;border:1px solid #E2D9D5;border-radius:20px;padding:0 18px;font-family:sans-serif;">
# MAGIC       <span style="font-size:13px;color:#0b2026;white-space:nowrap;">Uses governed data to answer questions</span>
# MAGIC     </div>
# MAGIC   </foreignObject>
# MAGIC
# MAGIC   <!-- Unity Catalog -->
# MAGIC   <rect x="22" y="406" width="626" height="244" rx="16" fill="#FDF9F8" stroke="#FF3621" stroke-width="2"/>
# MAGIC   <svg x="40" y="418" width="38" height="38" viewBox="-5.315 -5.0 210.0 210.0" overflow="visible">
# MAGIC     <path d="M51.5066 84.0254H19.5312V115.974H51.5066V84.0254Z" fill="#FABFBA"/>
# MAGIC     <path d="M21.6914 129.215L35.5186 153.105L49.3457 129.215H21.6914Z" fill="#FF5F46"/>
# MAGIC     <path d="M21.6914 70.9293H49.2017L35.3745 47.0396L21.6914 70.9293Z" fill="#FF5F46"/>
# MAGIC     <path d="M61.7338 133.82L45.7461 161.451L73.4004 177.426L89.3881 149.794L61.7338 133.82Z" fill="#FABFBA"/>
# MAGIC     <path d="M73.4004 22.7178L45.7461 38.6923L61.7338 66.3238L89.3881 50.3493L73.4004 22.7178Z" fill="#FABFBA"/>
# MAGIC     <path d="M99.7571 72.3687L75.8477 86.1844V113.816L99.7571 127.632L123.667 113.816V86.1844L99.7571 72.3687Z" fill="#FF5F46"/>
# MAGIC     <path d="M85.9297 182.031H113.44L99.6128 158.142L85.9297 182.031Z" fill="#FF5F46"/>
# MAGIC     <path d="M85.9297 17.9688L99.7568 41.8585L113.584 17.9688H85.9297Z" fill="#FF5F46"/>
# MAGIC     <path d="M125.972 22.7178L109.984 50.3493L137.639 66.3238L153.626 38.6923L125.972 22.7178Z" fill="#FABFBA"/>
# MAGIC     <path d="M137.639 133.82L109.984 149.794L125.972 177.426L153.626 161.451L137.639 133.82Z" fill="#FABFBA"/>
# MAGIC     <path d="M179.839 84.0254H147.863V115.974H179.839V84.0254Z" fill="#FABFBA"/>
# MAGIC     <path d="M150.168 70.9293H177.678L163.851 47.0396L150.168 70.9293Z" fill="#FF5F46"/>
# MAGIC     <path d="M150.168 129.215L163.995 153.105L177.822 129.215H150.168Z" fill="#FF5F46"/>
# MAGIC   </svg>
# MAGIC   <text x="86" y="436" font-size="17" font-weight="800" fill="#0b2026">Unity Catalog</text>
# MAGIC   <text x="86" y="455" font-size="14" fill="#5A6F77">governance layer</text>
# MAGIC
# MAGIC   <!-- Structured data (wider: w250) -->
# MAGIC   <rect x="44" y="478" width="250" height="152" rx="12" fill="#FFFFFF" stroke="#FF3621" stroke-width="2"/>
# MAGIC   <foreignObject x="44" y="478" width="250" height="152">
# MAGIC     <div xmlns="http://www.w3.org/1999/xhtml" style="box-sizing:border-box;width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:8px 12px;font-family:sans-serif;">
# MAGIC       <div style="display:inline-block;background:#FF5F46;color:#FFFFFF;font-size:10px;font-weight:700;padding:2px 9px;border-radius:9px;margin-bottom:3px;white-space:nowrap;">Chat + Agent mode</div>
# MAGIC       <div style="font-size:14px;font-weight:700;color:#0b2026;margin-bottom:6px;">Structured data</div>
# MAGIC       <div style="display:grid;grid-template-columns:auto auto;gap:6px 20px;">
# MAGIC         <div style="display:flex;align-items:center;gap:4px;">
# MAGIC           <span style="width:26px;height:24px;display:flex;align-items:center;justify-content:center;flex:none;">
# MAGIC             <svg viewBox="1 1 14 14" width="20" fill="#FF5F46"><path fill-rule="evenodd" clip-rule="evenodd" d="M1 1.75C1 1.33579 1.33579 1 1.75 1H14.25C14.6642 1 15 1.33579 15 1.75V14.25C15 14.6642 14.6642 15 14.25 15H1.75C1.33579 15 1 14.6642 1 14.25V1.75ZM2.5 2.5V5.5H13.5V2.5H2.5ZM2.5 13.5V7H5V13.5H2.5ZM6.5 13.5H9.5V7H6.5V13.5ZM11 7V13.5H13.5V7H11Z"/></svg>
# MAGIC           </span>
# MAGIC           <span style="font-size:11px;font-weight:600;color:#0b2026;line-height:1.1;">Tables</span>
# MAGIC         </div>
# MAGIC         <div style="display:flex;align-items:center;gap:4px;">
# MAGIC           <span style="width:26px;height:24px;display:flex;align-items:center;justify-content:center;flex:none;">
# MAGIC             <svg viewBox="1 1 15 14" width="21" fill="#FF5F46"><path fill-rule="evenodd" clip-rule="evenodd" d="M1.75 1C1.33579 1 1 1.33579 1 1.75V14.25C1 14.6642 1.33579 15 1.75 15H4V13.5H2.5V7H5V9H6.5V7H9.5V9H11V7H13.5V9H15V1.75C15 1.33579 14.6642 1 14.25 1H1.75ZM13.5 5.5V2.5H2.5V5.5H13.5Z"/><path fill-rule="evenodd" clip-rule="evenodd" d="M11.75 10C11.5511 10 11.3603 10.079 11.2197 10.2197C11.1398 10.2995 11.0798 10.3955 11.0429 10.5H9.95735C9.85439 10.2087 9.57658 10 9.25003 10L5.75 10C5.55108 10 5.36031 10.079 5.21966 10.2197C5.07901 10.3603 4.99999 10.5511 5 10.75L5.00003 12.5C5.00003 13.8807 6.11932 15 7.50003 15C8.88074 15 10 13.8807 10 12.5V12H11L11 12.5C11 13.8807 12.1193 15 13.5 15C14.8807 15 16 13.8807 16 12.5V10.7506C16 10.3364 15.6642 10 15.25 10L11.75 10ZM12.5 12.5L12.5 11.5L14.5 11.5V12.5C14.5 13.0523 14.0523 13.5 13.5 13.5C12.9477 13.5 12.5 13.0523 12.5 12.5ZM6.50002 11.5L6.50003 12.5C6.50003 13.0523 6.94774 13.5 7.50003 13.5C8.05231 13.5 8.50003 13.0523 8.50003 12.5V11.5001L6.50002 11.5Z"/></svg>
# MAGIC           </span>
# MAGIC           <span style="font-size:11px;font-weight:600;color:#0b2026;line-height:1.1;">Views</span>
# MAGIC         </div>
# MAGIC         <div style="display:flex;align-items:center;gap:4px;">
# MAGIC           <span style="width:26px;height:24px;display:flex;align-items:center;justify-content:center;flex:none;">
# MAGIC             <svg viewBox="8 6 30 28" width="21" fill="#FF5F46"><rect x="8" y="6" width="28" height="3"/><rect x="8" y="6" width="3" height="28"/><rect x="33" y="6" width="3" height="16"/><rect x="8" y="15" width="28" height="3"/><rect x="16" y="18" width="3" height="4"/><rect x="25" y="18" width="3" height="4"/><rect x="16" y="26" width="3" height="5"/><rect x="21" y="28" width="3" height="3"/><rect x="26" y="26" width="2" height="5"/><rect x="30" y="28" width="3" height="3"/><rect x="35" y="26" width="3" height="5"/><rect x="8" y="31" width="6" height="3"/><rect x="16" y="31" width="22" height="3"/></svg>
# MAGIC           </span>
# MAGIC           <span style="font-size:11px;font-weight:600;color:#0b2026;line-height:1.1;">Metric views</span>
# MAGIC         </div>
# MAGIC         <div style="display:flex;align-items:center;gap:4px;">
# MAGIC           <span style="width:26px;height:24px;display:flex;align-items:center;justify-content:center;flex:none;">
# MAGIC             <svg viewBox="14 6 34 36" width="22" fill="#FF5F46"><rect x="14" y="6" width="34" height="3"/><rect x="14" y="6" width="3" height="33"/><rect x="45" y="6" width="3" height="19"/><rect x="14" y="17" width="34" height="3"/><rect x="24" y="20" width="3" height="17"/><rect x="14" y="37" width="17" height="3"/><rect x="32.015" y="26.436" width="10.5" height="3.5" transform="rotate(131 37.265 28.186)"/><rect x="32.5" y="31" width="15" height="3"/><rect x="37.485" y="35.064" width="10.5" height="3.5" transform="rotate(131 42.735 36.814)"/></svg>
# MAGIC           </span>
# MAGIC           <span style="font-size:11px;font-weight:600;color:#0b2026;line-height:1.1;">Materialized views</span>
# MAGIC         </div>
# MAGIC       </div>
# MAGIC     </div>
# MAGIC   </foreignObject>
# MAGIC
# MAGIC   <!-- Unstructured data (shifted right: x306, w320) -->
# MAGIC   <rect x="306" y="478" width="320" height="152" rx="12" fill="#FFFFFF" stroke="#FF3621" stroke-width="2"/>
# MAGIC   <svg x="324" y="529" width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#FF5F46" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
# MAGIC     <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/>
# MAGIC   </svg>
# MAGIC   <foreignObject x="388" y="478" width="222" height="152">
# MAGIC     <div xmlns="http://www.w3.org/1999/xhtml" style="box-sizing:border-box;width:100%;height:100%;display:flex;flex-direction:column;align-items:flex-start;justify-content:flex-start;text-align:left;padding:8px 6px;font-family:sans-serif;">
# MAGIC       <div style="display:inline-block;background:#FF5F46;color:#FFFFFF;font-size:11px;font-weight:700;letter-spacing:0.2px;padding:2px 9px;border-radius:9px;margin-bottom:7px;white-space:nowrap;">Agent mode only</div>
# MAGIC       <div style="font-size:16px;font-weight:700;color:#0b2026;margin-bottom:5px;">Unstructured data</div>
# MAGIC       <div style="font-size:12.5px;line-height:1.3;color:#5A6F77;">PDFs, Word documents, PowerPoint files, images, and other supported files stored in Unity Catalog Volumes.</div>
# MAGIC     </div>
# MAGIC   </foreignObject>
# MAGIC
# MAGIC
# MAGIC
# MAGIC   <!-- Dotted connector: Unstructured data box → Schema panel -->
# MAGIC   <path d="M628 540 C 664 540, 688 300, 700 280" fill="none" stroke="#FF5F46" stroke-width="1.4" stroke-dasharray="6 4" stroke-linecap="round"/>
# MAGIC
# MAGIC   <!-- Schema panel: What Lakebase Creates -->
# MAGIC   <rect x="700" y="20" width="480" height="630" rx="14" fill="#FFFFFF" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <foreignObject x="700" y="20" width="480" height="630">
# MAGIC     <div xmlns="http://www.w3.org/1999/xhtml" style="width:480px;height:630px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;box-sizing:border-box;overflow:hidden;border-radius:14px;display:flex;flex-direction:column;">
# MAGIC       <!-- Header -->
# MAGIC       <div style="background:transparent;border-bottom:1.5px solid #E2D9D5;padding:11px 18px;flex-shrink:0;">
# MAGIC         <span style="background:rgba(27,81,98,0.10);padding:2px 10px;border-radius:6px;font-size:11px;font-weight:700;color:#1B5162;letter-spacing:0.5px;">LAKEBASE</span>
# MAGIC         <span style="font-size:14px;font-weight:700;color:#0b2026;margin-left:8px;">Index Table Schema</span>
# MAGIC         <span style="background:#FFF3E0;border:1px solid #FFB74D;color:#E65100;font-size:10px;font-weight:600;padding:2px 10px;border-radius:10px;margin-left:10px;white-space:nowrap;">Need permission to access</span>
# MAGIC       </div>
# MAGIC       <div style="padding:10px 16px;font-size:12px;overflow:hidden;flex:1;">
# MAGIC         <!-- Document metadata -->
# MAGIC         <div style="font-size:10px;font-weight:700;color:#5A6F77;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:4px;">Document Metadata</div>
# MAGIC         <div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.65;color:#0b2026;font-size:11.5px;">
# MAGIC           <div>filename <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC           <div>modificationTime <span style="color:#D32F2F;">TIMESTAMPTZ</span></div>
# MAGIC           <div>doc_uri <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC           <div>doc_type <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC         </div>
# MAGIC         <!-- Parsing config -->
# MAGIC         <div style="font-size:10px;font-weight:700;color:#5A6F77;letter-spacing:0.8px;text-transform:uppercase;margin:6px 0 4px;">Parsing Config</div>
# MAGIC         <div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.65;color:#0b2026;font-size:11.5px;">
# MAGIC           <div>parser <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC           <div>splitter <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC           <div>chunk_size <span style="color:#D32F2F;">INTEGER</span></div>
# MAGIC           <div>chunk_overlap <span style="color:#D32F2F;">INTEGER</span></div>
# MAGIC         </div>
# MAGIC         <!-- Chunks -->
# MAGIC         <div style="font-size:10px;font-weight:700;color:#5A6F77;letter-spacing:0.8px;text-transform:uppercase;margin:6px 0 4px;">Chunks</div>
# MAGIC         <div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.65;color:#0b2026;font-size:11.5px;">
# MAGIC           <div>chunk_id <span style="color:#D32F2F;">TEXT</span> <span style="color:#1976d2;font-weight:700;">PRIMARY KEY</span></div>
# MAGIC           <div>chunk_pos_id <span style="color:#D32F2F;">INTEGER</span></div>
# MAGIC           <div>chunk_text <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC           <div>metadata <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC           <div>parsed_doc_summary <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC         </div>
# MAGIC         <!-- Search columns -->
# MAGIC         <div style="font-size:10px;font-weight:700;color:#FF5F46;letter-spacing:0.8px;text-transform:uppercase;margin:8px 0 4px;">Search Columns</div>
# MAGIC         <div style="background:#FFF3F0;border:1px solid #FFCDD2;border-radius:6px;padding:6px 8px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.65;color:#0b2026;font-size:11px;">
# MAGIC           <div>embedding_model <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC           <div>__db_chunk_text_vector <span style="color:#D32F2F;">JSONB</span></div>
# MAGIC           <div>__db_chunk_text_pgvector <span style="color:#D32F2F;">VECTOR(1024)</span></div>
# MAGIC           <div style="color:#618794;font-size:10px;padding-left:12px;">GENERATED ALWAYS AS ... STORED</div>
# MAGIC           <div>__db_chunk_text_tsvector <span style="color:#D32F2F;">TSVECTOR</span></div>
# MAGIC           <div style="color:#618794;font-size:10px;padding-left:12px;">GENERATED ALWAYS AS (to_tsvector(...))</div>
# MAGIC         </div>
# MAGIC         <!-- Indexes -->
# MAGIC         <div style="font-size:10px;font-weight:700;color:#00A972;letter-spacing:0.8px;text-transform:uppercase;margin:8px 0 4px;">Indexes</div>
# MAGIC         <div style="background:#E6F5EE;border:1px solid #A5D6C0;border-radius:6px;padding:6px 8px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.7;color:#0b2026;font-size:11px;">
# MAGIC           <div><span style="color:#00A972;font-weight:700;">LAKEBASE_ANN</span> (__db_chunk_text_pgvector)</div>
# MAGIC           <div><span style="color:#00A972;font-weight:700;">LAKEBASE_BM25</span> (__db_chunk_text_tsvector)</div>
# MAGIC           <div><span style="color:#00A972;font-weight:700;">BTREE</span> (chunk_id)</div>
# MAGIC         </div>
# MAGIC       </div>
# MAGIC     </div>
# MAGIC   </foreignObject>
# MAGIC </svg>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### C2. Knowledge Source Types
# MAGIC
# MAGIC Agents can draw from multiple governed UC knowledge sources:
# MAGIC
# MAGIC | Source Type                      | Description                                                                                                                                           | Best For |
# MAGIC |---|---|---|
# MAGIC | **Files in UC Volume**          | Point to a Unity Catalog Volume containing files like `txt`, `pdf`, `md`, `ppt/pptx`, or `doc/docx` files. KA parses, chunks, embeds, and indexes them for you. Files >50 MB are automatically skipped. | Document collections that change over time (policies, manuals, specs) |
# MAGIC | **AI Search Index**         | Attach a pre-built AI Search index (using embedding models like `databricks-gte-large-en`). You own the parsing/chunking/indexing pipeline. | Custom RAG pipelines where you already manage the index and schema |
# MAGIC | **Files in UC Table / File Table** | Use a UC table (e.g. **streaming table** or have **Change Data Feed enabled**) with a `content` column (BINARY/STRING) storing file contents plus a `_metadata`/`metadata` struct with file path/name/size/mod time.| Documents ingested via connectors (SharePoint, Google Drive, Jira, Confluence) that land as file tables |

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Declarative vs. Code-First
# MAGIC
# MAGIC Databricks offers two approaches to building retrieval agents:
# MAGIC
# MAGIC |                       | Declarative (Genie Agent)             | Code-First (Custom Agents)                           |
# MAGIC |---|---|---|
# MAGIC | **Approach**          | Declare what you want; the system learns and optimizes how | Write retrieval logic, prompts, tools, and orchestration yourself |
# MAGIC | **Setup time**        | Minutes                                                     | Hours to days                                                     |
# MAGIC | **Customizability**   | Configuration- and feedback-driven                          | Essentially unlimited                                             |
# MAGIC | **Optimization**      | Automatic via feedback loops and platform research upgrades | Manual tuning and experimentation                                 |
# MAGIC | **Best for**          | High-quality Q&A over docs, fast prototyping               | Novel architectures, complex multi-step / tool-heavy workflows   |
# MAGIC
# MAGIC This course focuses on the **declarative** path. We’ll explore the components that understand the retrieval building blocks (document parsing, chunking, and AI Search), then bring them together using Genie Agents as the recommended managed approach.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07);">
# MAGIC
# MAGIC   <div style="height: 5px; border-radius: 5px; margin: -4px 0 12px; background: #FF5F46;"></div>
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/building-rag-agents-with-agent-bricks-en_us-2.0.0/images/20261014T152401Z/Building RAG Agents with Agent Bricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to know more about Databricks features? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/building-rag-agents-with-agent-bricks-en_us-2.0.0/images/20261014T152401Z/Building RAG Agents with Agent Bricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. 
# MAGIC     <br></br>
# MAGIC     For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #F9F7F4; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-query" style="flex: 1;">How can Genie Agents be used with Databricks Apps?</span>
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
# MAGIC ## D. Introduction to Lakebase
# MAGIC
# MAGIC <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 160 1600 740" width="100%" style="max-width:1600px;height:auto;display:block;margin:0 auto">
# MAGIC   <defs>
# MAGIC     <style>
# MAGIC       @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&amp;display=swap');
# MAGIC       text { font-family: 'DM Sans','Segoe UI',Arial,sans-serif; }
# MAGIC       .hb{fill:#1B3139;font-weight:800}.h{fill:#1B3139;font-weight:700}
# MAGIC       .body{fill:#1B3139;font-weight:400}.lbl{fill:#1B3139;font-weight:700}.meta{fill:#5A6A70;font-weight:500}
# MAGIC     </style>
# MAGIC     <filter id="tileShadow" x="-45%" y="-45%" width="190%" height="200%"><feDropShadow dx="0" dy="2.5" stdDeviation="4" flood-color="#1B3139" flood-opacity="0.13"/></filter>
# MAGIC     <filter id="softShadow" x="-8%" y="-8%" width="116%" height="130%"><feDropShadow dx="0" dy="4" stdDeviation="10" flood-color="#1B3139" flood-opacity="0.10"/></filter>
# MAGIC     <symbol id="dbxMark" viewBox="0 0 105 113"><path d="M98.9 46.5996L52.3 72.8996L2.4 44.7996L0 46.0996V66.4996L52.3 95.8996L98.9 69.6996V80.4996L52.3 106.8L2.4 78.6996L0 79.9996V83.4996L52.3 112.9L104.5 83.4996V63.0996L102.1 61.7996L52.3 89.7996L5.6 63.5996V52.7996L52.3 78.9996L104.5 49.5996V29.4996L101.9 27.9996L52.3 55.8996L8 31.0996L52.3 6.19961L88.7 26.6996L91.9 24.8996V22.3996L52.3 0.0996094L0 29.4996V32.6996L52.3 62.0996L98.9 35.7996V46.5996Z" fill="#FF3621"/></symbol>
# MAGIC     <clipPath id="lakeClip"><rect x="580" y="402" width="446" height="164" rx="16"/></clipPath>
# MAGIC   </defs>
# MAGIC   <rect x="0" y="0" width="1600" height="900" fill="#FFFFFF"/>
# MAGIC   <rect x="196" y="196" width="1216" height="402" rx="18" fill="#FFFFFF" stroke="#C6DCEA" stroke-width="1.5" filter="url(#softShadow)"/>
# MAGIC   <text x="236" y="252" class="body" font-size="15">Separated storage</text>
# MAGIC   <text x="236" y="272" class="body" font-size="15">and compute, enabling</text>
# MAGIC   <text x="236" y="292" class="body" font-size="15">independent scale for</text>
# MAGIC   <text x="236" y="312" class="body" font-size="15">each resource</text>
# MAGIC   <text x="1156" y="258" class="body" font-size="15">Native services for agent</text>
# MAGIC   <text x="1156" y="278" class="body" font-size="15">storage, memory, time</text>
# MAGIC   <text x="1156" y="298" class="body" font-size="15">travel, and recovery</text>
# MAGIC   <rect x="458" y="222" width="206" height="124" rx="14" fill="#FFFFFF" filter="url(#tileShadow)"/>
# MAGIC   <svg x="486" y="234" width="30" height="30" viewBox="192 59 512 512"><path d="M192 196.45C192 120.538 253.538 59.0001 329.45 59.0001H566.55C642.462 59.0001 704 120.538 704 196.45V433.55C704 509.462 642.462 571 566.55 571H329.45C253.538 571 192 509.462 192 433.55V196.45Z" fill="white"/><path d="M464.872 376.037H592.092V462.101H445.293L393.629 376.037H303.304V289.973L413.206 289.975L464.872 376.037Z" fill="#FF5F46"/><path d="M501.768 259.377L592.092 259.377V345.443L482.191 345.441L430.525 259.38L303.304 259.377V173.313L450.103 173.315L501.768 259.377Z" fill="#FABFBA"/></svg>
# MAGIC   <text x="524" y="256" class="hb" font-size="21">Lakeflow</text>
# MAGIC   <rect x="472.0" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="476.5" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="475.5" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="475.5" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="502.2" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="506.8" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="505.8" y="286.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="505.8" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="532.5" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="537.0" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="536.0" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="536.0" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="562.8" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="567.2" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="566.2" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="566.2" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="593.0" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="597.5" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="596.5" y="286.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="596.5" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="623.2" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="627.8" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="626.8" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="626.8" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="472.0" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="476.5" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="475.5" y="318.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="475.5" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="502.2" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="506.8" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="505.8" y="318.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="505.8" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="532.5" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="537.0" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="536.0" y="318.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="536.0" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="562.8" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="567.2" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="566.2" y="318.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="566.2" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="593.0" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="597.5" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="596.5" y="318.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="596.5" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="623.2" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="627.8" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="626.8" y="318.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="626.8" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="690" y="222" width="206" height="124" rx="14" fill="#FFFFFF" filter="url(#tileShadow)"/>
# MAGIC   <svg x="718" y="234" width="30" height="30" viewBox="191 59 512 512"><path d="M191.359 196.33C191.359 120.418 252.898 58.88 328.809 58.88H565.91C641.821 58.88 703.359 120.418 703.359 196.33V433.43C703.359 509.342 641.821 570.88 565.91 570.88H328.809C252.898 570.88 191.359 509.342 191.359 433.43V196.33Z" fill="white"/><path d="M591.208 281.742V362.52H591.214L591.208 362.523V362.557H591.133C569.727 372.414 545.232 378.047 519.084 378.047C492.937 378.047 468.416 372.414 447.037 362.557H447.051C425.67 352.676 401.139 347.031 374.98 347.031C348.821 347.031 324.317 352.676 302.909 362.557H302.906V282.843H302.922V281.742L446.411 169.092L591.208 281.742Z" fill="#FABFBA"/><path d="M375.031 378.05C401.179 378.05 425.701 383.683 447.081 393.541H447.064C468.445 403.421 492.976 409.067 519.135 409.067C544.477 409.067 568.265 403.768 589.191 394.455L591.206 393.541V463.688H591.212C591.21 463.689 591.208 463.69 591.206 463.691V463.723H591.134C569.727 473.581 545.23 479.214 519.082 479.214C492.934 479.214 468.411 473.581 447.031 463.723H447.049C425.668 453.843 401.137 448.197 374.978 448.197C349.636 448.197 325.847 453.495 304.922 462.809L302.907 463.723H302.904L302.905 393.576V393.541H302.98C324.387 383.683 348.883 378.05 375.031 378.05Z" fill="#FF5F46"/></svg>
# MAGIC   <text x="756" y="256" class="hb" font-size="21">Lakehouse</text>
# MAGIC   <rect x="704.0" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="708.5" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="707.5" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="707.5" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="734.2" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="738.8" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="737.8" y="286.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="737.8" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="764.5" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="769.0" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="768.0" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="768.0" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="794.8" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="799.2" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="798.2" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="798.2" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="825.0" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="829.5" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="828.5" y="286.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="828.5" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="855.2" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="859.8" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="858.8" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="858.8" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="704.0" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="708.5" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="707.5" y="318.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="707.5" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="734.2" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="738.8" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="737.8" y="318.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="737.8" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="764.5" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="769.0" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="768.0" y="318.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="768.0" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="794.8" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="799.2" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="798.2" y="318.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="798.2" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="825.0" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="829.5" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="828.5" y="318.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="828.5" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="855.2" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="859.8" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="858.8" y="318.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="858.8" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="922" y="222" width="206" height="124" rx="14" fill="#FFFFFF" filter="url(#tileShadow)"/>
# MAGIC   <svg x="950" y="234" width="30" height="30" viewBox="192 59 512 512"><path d="M192 196.45C192 120.538 253.538 59 329.45 59H566.55C642.462 59 704 120.538 704 196.45V433.55C704 509.462 642.462 571 566.55 571H329.45C253.538 571 192 509.462 192 433.55V196.45Z" fill="white"/><path d="M592.094 355.275H592.091C570.646 362.487 546.1 366.607 519.896 366.607C493.693 366.607 469.12 362.487 447.702 355.275H303.305V278.516H592.094V355.275ZM375.502 343.466C349.297 343.466 324.75 347.76 303.305 355.274H447.699C426.281 347.759 401.707 343.466 375.502 343.466Z" fill="#FABFBA"/><path d="M375.841 371.695C401.108 371.695 424.851 375.365 445.691 381.815L449.707 383.081C470.548 389.532 494.291 393.201 519.558 393.201C544.825 393.201 568.543 389.532 589.406 383.081L591.416 382.448H592.094V462.102H303.305V382.448H303.982C325.327 375.605 349.759 371.695 375.841 371.695Z" fill="#FF5F46"/><rect x="303.304" y="173.312" width="288.79" height="79.6536" fill="#FABFBA"/></svg>
# MAGIC   <text x="988" y="256" class="hb" font-size="21">Lakebase</text>
# MAGIC   <rect x="936.0" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="940.5" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="939.5" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="939.5" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="966.2" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="970.8" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="969.8" y="286.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="969.8" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="996.5" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="1001.0" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="1000.0" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="1000.0" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="1026.8" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="1031.2" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="1030.2" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="1030.2" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="1057.0" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="1061.5" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="1060.5" y="286.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="1060.5" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="1087.2" y="272.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="1091.8" cy="276.6" r="1.2" fill="#FFFFFF"/><rect x="1090.8" y="286.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="1090.8" y="291.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="936.0" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="940.5" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="939.5" y="318.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="939.5" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="966.2" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="970.8" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="969.8" y="318.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="969.8" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="996.5" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="1001.0" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="1000.0" y="318.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="1000.0" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="1026.8" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="1031.2" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="1030.2" y="318.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="1030.2" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="1057.0" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="1061.5" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="1060.5" y="318.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/><rect x="1060.5" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <rect x="1087.2" y="304.0" width="26.8" height="27" rx="3" fill="#F3D2CC"/><circle cx="1091.8" cy="308.6" r="1.2" fill="#FFFFFF"/><rect x="1090.8" y="318.0" width="19.8" height="2.4" rx="1.2" fill="#EE5540"/><rect x="1090.8" y="323.4" width="19.8" height="1.6" rx="0.8" fill="#FFFFFF"/>
# MAGIC   <svg x="260" y="360" width="27" height="27" viewBox="48 46 414 420"><path d="M181.185 46C204.241 46 223.38 63.8165 225.026 86.8145L238 268H126L137.316 87.208C138.766 64.0446 157.976 46.0001 181.185 46ZM405.266 114C428.468 114 447.914 131.542 450.297 154.622L462 268H350L360.179 155.202C362.284 131.871 381.84 114 405.266 114Z" fill="#FF5F46"/><path d="M338.996 424.79C337.367 448.002 318.062 466 294.793 466H294.792C271.695 465.999 252.471 448.258 250.623 425.234L238 268H350L338.996 424.79ZM128 268L120.132 380.564C118.973 397.143 105.185 410 88.5664 410C72.0694 410 58.3413 397.324 57.0264 380.88L48 268H128Z" fill="#FABFBA"/></svg>
# MAGIC   <text x="296" y="370" class="body" font-size="15">Real-time</text>
# MAGIC   <text x="296" y="388" class="body" font-size="15">streaming</text>
# MAGIC   <svg x="388" y="364" width="23" height="23" viewBox="64 64 385 385"><path d="M64 449.034L64.0015 266.742L146.336 266.742L146.336 365.752L244.527 365.752L244.527 449.034L64 449.034Z" fill="#FABFBA"/><path d="M447.992 64.0015L447.991 246.293L365.656 246.293L365.656 147.284L267.466 147.284L267.466 64.0015L447.992 64.0015Z" fill="#FABFBA"/><rect width="137.404" height="137.409" transform="matrix(1 0 5.39465e-06 1 64 64.001)" fill="#FF5F46"/><rect width="137.405" height="137.409" transform="matrix(1 0 5.39465e-06 1 311.744 311.665)" fill="#FF5F46"/></svg>
# MAGIC   <text x="418" y="382" class="body" font-size="15">Zerobus</text>
# MAGIC   <rect x="322.0" y="410.0" width="72.0" height="56.0" fill="#9DA3A6"/><ellipse cx="358.0" cy="466.0" rx="36.0" ry="10.8" fill="#9DA3A6"/><path d="M322.0,438.0 A36.0,10.8 0 0 0 394.0,438.0" fill="none" stroke="#FFFFFF" stroke-width="1.4" opacity="0.85"/><ellipse cx="358.0" cy="410.0" rx="36.0" ry="10.8" fill="#B7BBBD" stroke="#FFFFFF" stroke-width="1.2"/>
# MAGIC   <rect x="400.0" y="416.0" width="64.0" height="52.0" fill="#9DA3A6"/><ellipse cx="432.0" cy="468.0" rx="32.0" ry="9.6" fill="#9DA3A6"/><path d="M400.0,442.0 A32.0,9.6 0 0 0 464.0,442.0" fill="none" stroke="#FFFFFF" stroke-width="1.4" opacity="0.85"/><ellipse cx="432.0" cy="416.0" rx="32.0" ry="9.6" fill="#B7BBBD" stroke="#FFFFFF" stroke-width="1.2"/>
# MAGIC   <rect x="272.0" y="440.0" width="60.0" height="48.0" fill="#CFD2D3"/><ellipse cx="302.0" cy="488.0" rx="30.0" ry="9.0" fill="#CFD2D3"/><path d="M272.0,464.0 A30.0,9.0 0 0 0 332.0,464.0" fill="none" stroke="#FFFFFF" stroke-width="1.4" opacity="0.85"/><ellipse cx="302.0" cy="440.0" rx="30.0" ry="9.0" fill="#E7E8E9" stroke="#FFFFFF" stroke-width="1.2"/>
# MAGIC   <rect x="342.0" y="444.0" width="68.0" height="50.0" fill="#CFD2D3"/><ellipse cx="376.0" cy="494.0" rx="34.0" ry="10.2" fill="#CFD2D3"/><path d="M342.0,469.0 A34.0,10.2 0 0 0 410.0,469.0" fill="none" stroke="#FFFFFF" stroke-width="1.4" opacity="0.85"/><ellipse cx="376.0" cy="444.0" rx="34.0" ry="10.2" fill="#E7E8E9" stroke="#FFFFFF" stroke-width="1.2"/>
# MAGIC   <rect x="427.0" y="452.0" width="42.0" height="38.0" fill="#CFD2D3"/><ellipse cx="448.0" cy="490.0" rx="21.0" ry="6.3" fill="#CFD2D3"/><path d="M427.0,471.0 A21.0,6.3 0 0 0 469.0,471.0" fill="none" stroke="#FFFFFF" stroke-width="1.4" opacity="0.85"/><ellipse cx="448.0" cy="452.0" rx="21.0" ry="6.3" fill="#E7E8E9" stroke="#FFFFFF" stroke-width="1.2"/>
# MAGIC   <text x="376" y="516" text-anchor="middle" class="body" font-size="15">100+ supported</text>
# MAGIC   <text x="376" y="535" text-anchor="middle" class="body" font-size="15">data/app connectors</text>
# MAGIC   <path d="M470,432 L490,432 L490,352" fill="none" stroke="#52707E" stroke-width="2"/>
# MAGIC   <path d="M486.5,357 L490,350 L493.5,357" fill="#52707E"/>
# MAGIC   <line x1="636" y1="348" x2="636" y2="396" stroke="#52707E" stroke-width="2"/>  <path d="M632,396 L636,402 L640,396" fill="#52707E"/>
# MAGIC   <line x1="738" y1="354" x2="738" y2="396" stroke="#20A15B" stroke-width="2"/>  <path d="M734,354 L738,348 L742,354" fill="#20A15B"/>  <path d="M734,396 L738,402 L742,396" fill="#20A15B"/>
# MAGIC   <line x1="996" y1="354" x2="996" y2="396" stroke="#FF3621" stroke-width="2"/>  <path d="M992,354 L996,348 L1000,354" fill="#FF3621"/>  <path d="M992,396 L996,402 L1000,396" fill="#FF3621"/>
# MAGIC   <text x="505" y="372" font-size="16" font-weight="700" fill="#52707E">Ingest, ETL</text>
# MAGIC   <text x="505" y="394" font-size="16" font-weight="700" fill="#52707E">&amp; streaming</text>
# MAGIC   <text x="758" y="372" font-size="16" font-weight="700" fill="#20A15B">Real-time +</text>
# MAGIC   <text x="758" y="394" font-size="16" font-weight="700" fill="#20A15B">batch analytics</text>
# MAGIC   <text x="1016" y="384" font-size="16" font-weight="700" fill="#FF3621">Transactions</text>
# MAGIC   <rect x="580" y="402" width="446" height="164" rx="16" fill="#FFFFFF" filter="url(#tileShadow)"/>
# MAGIC   <rect x="600.0" y="421" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="609.0" cy="430.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="668.8" y="421" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="677.8" cy="430.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="737.7" y="421" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="746.7" cy="430.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="806.5" y="421" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="815.5" cy="430.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="875.3" y="421" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="884.3" cy="430.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="944.2" y="421" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="953.2" cy="430.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="600.0" y="447" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="609.0" cy="456.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="668.8" y="447" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="677.8" cy="456.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="737.7" y="447" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="746.7" cy="456.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="806.5" y="447" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="815.5" cy="456.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="875.3" y="447" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="884.3" cy="456.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="944.2" y="447" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="953.2" cy="456.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="600.0" y="473" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="609.0" cy="482.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="668.8" y="473" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="677.8" cy="482.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="737.7" y="473" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="746.7" cy="482.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="806.5" y="473" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="815.5" cy="482.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="875.3" y="473" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="884.3" cy="482.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <rect x="944.2" y="473" width="61.8" height="18" rx="3" fill="#8FA3AD"/>
# MAGIC   <circle cx="953.2" cy="482.0" r="2.6" fill="#FFFFFF"/>
# MAGIC   <path d="M580,509.0 L583,507.6 L586,506.3 L589,505.1 L592,504.1 L595,503.3 L598,502.8 L601,502.5 L604,502.5 L607,502.9 L610,503.4 L613,504.3 L616,505.3 L619,506.5 L622,507.8 L625,509.2 L628,510.5 L631,511.8 L634,513.0 L637,514.0 L640,514.7 L643,515.3 L646,515.5 L649,515.4 L652,515.1 L655,514.5 L658,513.6 L661,512.5 L664,511.3 L667,510.0 L670,508.6 L673,507.3 L676,506.0 L679,504.9 L682,503.9 L685,503.2 L688,502.7 L691,502.5 L694,502.6 L697,503.0 L700,503.6 L703,504.5 L706,505.6 L709,506.8 L712,508.2 L715,509.5 L718,510.9 L721,512.2 L724,513.3 L727,514.2 L730,514.9 L733,515.3 L736,515.5 L739,515.4 L742,514.9 L745,514.3 L748,513.3 L751,512.2 L754,511.0 L757,509.6 L760,508.3 L763,506.9 L766,505.7 L769,504.6 L772,503.7 L775,503.0 L778,502.6 L781,502.5 L784,502.7 L787,503.1 L790,503.8 L793,504.8 L796,505.9 L799,507.2 L802,508.5 L805,509.9 L808,511.2 L811,512.5 L814,513.5 L817,514.4 L820,515.1 L823,515.4 L826,515.5 L829,515.3 L832,514.8 L835,514.0 L838,513.1 L841,511.9 L844,510.6 L847,509.3 L850,507.9 L853,506.6 L856,505.4 L859,504.3 L862,503.5 L865,502.9 L868,502.6 L871,502.5 L874,502.8 L877,503.3 L880,504.1 L883,505.1 L886,506.3 L889,507.5 L892,508.9 L895,510.3 L898,511.6 L901,512.8 L904,513.8 L907,514.6 L910,515.2 L913,515.5 L916,515.5 L919,515.2 L922,514.6 L925,513.8 L928,512.8 L931,511.6 L934,510.3 L937,508.9 L940,507.5 L943,506.3 L946,505.1 L949,504.1 L952,503.3 L955,502.8 L958,502.5 L961,502.6 L964,502.9 L967,503.5 L970,504.3 L973,505.4 L976,506.6 L979,507.9 L982,509.3 L985,510.6 L988,511.9 L991,513.1 L994,514.0 L997,514.8 L1000,515.3 L1003,515.5 L1006,515.4 L1009,515.1 L1012,514.4 L1015,513.5 L1018,512.5 L1021,511.2 L1024,509.9 L1026,566 L580,566 Z" fill="#16495B" clip-path="url(#lakeClip)"/>
# MAGIC   <svg x="712" y="534" width="77" height="16" viewBox="6.5 31.4 176.3 36.4"><path d="M43.4295 61.3529L26.3179 31.8768C25.9325 31.2299 24.9818 31.2299 24.5964 31.8768L10.0541 56.9276C9.79712 57.3676 10.2596 57.8851 10.7221 57.6522C10.9019 57.5746 11.1332 57.4711 11.3644 57.3934C11.3644 57.3934 11.7498 57.2382 12.4435 57.0311C13.1886 56.7982 13.9851 56.5912 14.7302 56.41C16.7857 55.9442 19.4835 55.556 21.8986 55.9442C24.1339 56.3065 25.7012 56.9276 26.8317 57.5228C27.3456 57.8075 27.8338 57.1088 27.4227 56.6947C25.804 55.1161 23.2861 53.1752 19.4835 52.3729C19.3293 52.347 19.2265 52.14 19.3293 52.0106L25.0332 42.1508C25.213 41.8143 25.6755 41.8143 25.8811 42.1508L36.8264 60.9906C37.0576 61.3788 36.8521 61.8964 36.441 62.0516C35.5931 62.3363 34.5654 62.5433 33.4092 62.5433C32.6898 62.5433 31.9447 62.4657 31.1996 62.2845C31.1996 62.2845 29.0927 61.9481 26.2408 60.2401C23.4145 58.5321 17.9419 56.3583 11.8269 59.8778C9.20618 61.5599 7.45904 63.889 6.53408 65.3383C6.37992 65.5712 6.63686 65.83 6.8681 65.7006C8.97494 64.5619 14.6531 61.8964 19.7661 62.8539C22.7979 63.4232 26.2922 65.7264 29.8892 67.0721C35.8757 69.2977 41.7338 65.5712 41.7338 65.5453C42.3504 65.1312 42.7101 64.8207 42.7872 64.7689L42.8643 64.6913C43.8406 63.8632 44.0719 62.4657 43.4295 61.3529Z" fill="#FFFFFF"/><path d="M49.1079 56.876V42.2803C49.1079 42.1509 49.1593 42.0215 49.2621 41.918C49.3648 41.8145 49.4933 41.7627 49.6218 41.7627H55.4028C57.6124 41.7627 59.4366 42.4097 60.8497 43.7295C61.6205 44.4282 62.1858 45.2822 62.5969 46.2656C62.9823 47.249 63.1878 48.3618 63.1878 49.5781C63.1878 50.7944 62.9823 51.9072 62.5969 52.8906C62.2115 53.874 61.6205 54.728 60.8497 55.4268C60.156 56.0737 59.3338 56.5395 58.4346 56.876C57.5353 57.2124 56.5076 57.3677 55.4284 57.3677H49.6475C49.519 57.3677 49.3905 57.3159 49.2878 57.2124C49.1593 57.1348 49.1079 57.0054 49.1079 56.876ZM53.8098 53.46H55.4028C55.7882 53.46 56.1992 53.4082 56.636 53.3047C57.0471 53.2012 57.4325 53.02 57.7665 52.7354C58.1005 52.4507 58.3832 52.0625 58.6144 51.5708C58.82 51.0532 58.9484 50.4063 58.9484 49.604C58.9484 48.8018 58.8456 48.1289 58.6144 47.6372C58.3832 47.1196 58.1005 46.7314 57.7665 46.4727C57.4325 46.188 57.0471 46.0068 56.636 45.9033C56.2249 45.7998 55.8138 45.748 55.4284 45.748H53.8098C53.6813 45.748 53.5528 45.7998 53.4501 45.9033C53.3473 46.0068 53.2959 46.1362 53.2959 46.2656V52.9683C53.2959 53.1235 53.3473 53.227 53.4501 53.3306C53.5271 53.4082 53.6556 53.46 53.8098 53.46Z" fill="#FFFFFF"/><path d="M69.2256 46.2139V47.1196C69.2256 47.249 69.277 47.3784 69.3797 47.4819C69.4825 47.5854 69.611 47.6372 69.7394 47.6372H75.469C75.5975 47.6372 75.726 47.689 75.8287 47.7925C75.9315 47.896 75.9829 48.0254 75.9829 48.1548V51.105C75.9829 51.2344 75.9315 51.3638 75.8287 51.4673C75.726 51.5708 75.5975 51.6226 75.469 51.6226H69.7394C69.611 51.6226 69.4825 51.6743 69.3797 51.7778C69.277 51.8813 69.2256 52.0107 69.2256 52.1401V52.9424C69.2256 53.0977 69.277 53.2012 69.3797 53.3047C69.4825 53.4082 69.611 53.46 69.7394 53.46H76.3169C76.4454 53.46 76.5738 53.5117 76.6766 53.6152C76.7794 53.7187 76.8308 53.8481 76.8308 53.9775V56.9277C76.8308 57.083 76.7794 57.1865 76.6766 57.29C76.5738 57.3936 76.4454 57.4453 76.3169 57.4453H65.5515C65.423 57.4453 65.2945 57.3936 65.1918 57.29C65.089 57.1865 65.0376 57.0571 65.0376 56.9277V42.2803C65.0376 42.1509 65.089 42.0215 65.1918 41.918C65.2945 41.8145 65.423 41.7627 65.5515 41.7627H76.3169C76.4454 41.7627 76.5738 41.8145 76.6766 41.918C76.7794 42.0215 76.8308 42.1509 76.8308 42.2803V45.1787C76.8308 45.334 76.7794 45.4375 76.6766 45.541C76.5738 45.6445 76.4454 45.6963 76.3169 45.6963H69.7394C69.611 45.6963 69.4825 45.748 69.3797 45.8516C69.2513 45.9551 69.2256 46.0586 69.2256 46.2139Z" fill="#FFFFFF"/><path d="M91.0649 53.9775V56.876C91.0649 57.0312 91.0135 57.1348 90.9107 57.2383C90.8079 57.3418 90.6795 57.3936 90.551 57.3936H79.9911C79.8369 57.3936 79.7085 57.3418 79.6314 57.2383C79.5286 57.1348 79.5029 57.0054 79.5029 56.876V42.2803C79.5029 42.1509 79.5543 42.0215 79.6314 41.918C79.7342 41.8145 79.8369 41.7627 79.9911 41.7627H83.1771C83.3055 41.7627 83.434 41.8145 83.5368 41.918C83.6395 42.0215 83.6909 42.1509 83.6909 42.2803V52.9424C83.6909 53.0977 83.7423 53.2012 83.8194 53.3047C83.9222 53.4082 84.0249 53.46 84.1791 53.46H90.551C90.6795 53.46 90.8079 53.5117 90.9107 53.6152C91.0135 53.7187 91.0649 53.8223 91.0649 53.9775Z" fill="#FFFFFF"/><path d="M100.905 45.6963H97.9507C97.8222 45.6963 97.6938 45.748 97.591 45.8516C97.4882 45.9551 97.4369 46.0845 97.4369 46.2139V56.876C97.4369 57.0312 97.3855 57.1348 97.3084 57.2383C97.2056 57.3418 97.1028 57.3936 96.9487 57.3936H93.7627C93.6343 57.3936 93.5058 57.3418 93.403 57.2383C93.3003 57.1348 93.2489 57.0054 93.2489 56.876V46.2139C93.2489 46.0845 93.1975 45.9551 93.1204 45.8516C93.0176 45.748 92.9149 45.6963 92.7607 45.6963H89.7546C89.6004 45.6963 89.4977 45.6445 89.3949 45.541C89.2921 45.4375 89.2407 45.3081 89.2407 45.1787V42.2803C89.2407 42.1509 89.2921 42.0215 89.3949 41.918C89.4977 41.8145 89.6004 41.7627 89.7546 41.7627H100.931C101.085 41.7627 101.214 41.8145 101.291 41.918C101.394 42.0215 101.419 42.1509 101.419 42.2803V45.1787C101.419 45.334 101.368 45.4375 101.291 45.541C101.188 45.6445 101.06 45.6963 100.905 45.6963Z" fill="#FFFFFF"/><path d="M105.967 42.2539C106.018 42.1245 106.121 42.021 106.25 41.9175C106.378 41.814 106.532 41.7881 106.687 41.7881H109.641C109.77 41.7881 109.924 41.8398 110.052 41.9175C110.181 42.021 110.284 42.1245 110.335 42.2539L116.527 56.7461C116.578 56.8755 116.578 57.0049 116.501 57.1602C116.424 57.2896 116.27 57.3672 116.039 57.3672H112.416C112.262 57.3672 112.108 57.3154 111.979 57.2378C111.851 57.1343 111.748 57.0308 111.671 56.9014L111.106 55.6333C111.029 55.5039 110.926 55.4004 110.797 55.2969C110.669 55.1934 110.515 55.1675 110.386 55.1675H105.916C105.762 55.1675 105.607 55.2192 105.479 55.2969C105.35 55.4004 105.248 55.5039 105.196 55.6333L104.605 56.9014C104.554 57.0308 104.451 57.1343 104.323 57.2378C104.194 57.3413 104.04 57.3672 103.912 57.3672H100.263C100.186 57.3672 100.109 57.3413 100.032 57.3154C99.9549 57.2896 99.9035 57.2378 99.8521 57.186C99.8007 57.1343 99.775 57.0566 99.7493 56.979C99.7236 56.9014 99.7493 56.7979 99.8007 56.6685L105.967 42.2539ZM107.226 51.5444H109.102C109.23 51.5444 109.333 51.4927 109.41 51.415C109.461 51.3115 109.461 51.208 109.41 51.0786C109.282 50.7163 109.102 50.3022 108.896 49.8364C108.691 49.3706 108.511 48.9048 108.357 48.4907C108.305 48.3613 108.228 48.3096 108.177 48.3096C108.1 48.3096 108.048 48.3613 107.997 48.4907L106.943 51.0786C106.892 51.208 106.892 51.3115 106.943 51.415C106.969 51.4927 107.072 51.5444 107.226 51.5444Z" fill="#FFFFFF"/><path d="M136.825 53.9775V56.876C136.825 57.0312 136.773 57.1348 136.671 57.2383C136.568 57.3418 136.439 57.3936 136.311 57.3936H125.751C125.597 57.3936 125.468 57.3418 125.391 57.2383C125.288 57.1348 125.263 57.0054 125.263 56.876V42.2803C125.263 42.1509 125.314 42.0215 125.391 41.918C125.494 41.8145 125.597 41.7627 125.751 41.7627H128.937C129.065 41.7627 129.194 41.8145 129.297 41.918C129.399 42.0215 129.451 42.1509 129.451 42.2803V52.9424C129.451 53.0977 129.502 53.2012 129.579 53.3047C129.682 53.4082 129.785 53.46 129.939 53.46H136.311C136.439 53.46 136.568 53.5117 136.671 53.6152C136.773 53.7187 136.825 53.8223 136.825 53.9775Z" fill="#FFFFFF"/><path d="M143.351 42.2539C143.402 42.1245 143.505 42.021 143.633 41.9175C143.762 41.814 143.916 41.7881 144.07 41.7881H147.025C147.153 41.7881 147.308 41.8398 147.436 41.9175C147.565 42.021 147.667 42.1245 147.719 42.2539L153.911 56.7461C153.962 56.8755 153.962 57.0049 153.885 57.1602C153.808 57.2896 153.654 57.3672 153.423 57.3672H149.8C149.646 57.3672 149.492 57.3154 149.363 57.2378C149.235 57.1343 149.132 57.0308 149.055 56.9014L148.49 55.6333C148.412 55.5039 148.31 55.4004 148.181 55.2969C148.053 55.1934 147.899 55.1675 147.77 55.1675H143.299C143.145 55.1675 142.991 55.2192 142.863 55.2969C142.734 55.4004 142.631 55.5039 142.58 55.6333L141.989 56.9014C141.938 57.0308 141.835 57.1343 141.707 57.2378C141.578 57.3413 141.424 57.3672 141.295 57.3672H137.647C137.57 57.3672 137.493 57.3413 137.416 57.3154C137.339 57.2896 137.287 57.2378 137.236 57.186C137.185 57.1343 137.159 57.0566 137.133 56.979C137.107 56.9014 137.133 56.7979 137.185 56.6685L143.351 42.2539ZM144.584 51.5444H146.46C146.588 51.5444 146.691 51.4927 146.768 51.415C146.819 51.3115 146.819 51.208 146.768 51.0786C146.64 50.7163 146.46 50.3022 146.254 49.8364C146.049 49.3706 145.869 48.9048 145.715 48.4907C145.663 48.3613 145.586 48.3096 145.535 48.3096C145.483 48.3096 145.406 48.3613 145.355 48.4907L144.302 51.0786C144.25 51.208 144.25 51.3115 144.302 51.415C144.327 51.4927 144.43 51.5444 144.584 51.5444Z" fill="#FFFFFF"/><path d="M164.085 57.0054L159.974 52.373C159.871 52.2695 159.794 52.2178 159.743 52.2437C159.692 52.2695 159.64 52.3472 159.64 52.5024V56.876C159.64 57.0312 159.589 57.1348 159.486 57.2383C159.383 57.3418 159.255 57.3936 159.126 57.3936H155.94C155.786 57.3936 155.658 57.3418 155.581 57.2383C155.478 57.1348 155.452 57.0054 155.452 56.876V42.2803C155.452 42.1509 155.504 42.0215 155.581 41.918C155.683 41.8145 155.786 41.7627 155.94 41.7627H159.126C159.255 41.7627 159.383 41.8145 159.486 41.918C159.589 42.0215 159.64 42.1509 159.64 42.2803V47.1973C159.64 47.3525 159.666 47.4302 159.717 47.4302C159.769 47.4561 159.846 47.4043 159.923 47.3008L163.725 42.2026C163.802 42.0991 163.931 41.9956 164.085 41.918C164.239 41.8403 164.393 41.7886 164.548 41.7886H168.581C168.736 41.7886 168.838 41.8662 168.89 41.9956C168.941 42.125 168.915 42.2544 168.838 42.3579L163.289 49.397C163.211 49.5005 163.16 49.6299 163.16 49.7852C163.16 49.9404 163.212 50.0698 163.314 50.1475C163.802 50.665 164.291 51.2085 164.83 51.7778C165.37 52.3472 165.909 52.9165 166.423 53.4858C166.963 54.0552 167.477 54.6245 167.99 55.1938C168.504 55.7632 168.992 56.2808 169.455 56.7725C169.558 56.876 169.583 57.0054 169.558 57.1606C169.532 57.29 169.429 57.3677 169.275 57.3677H164.882C164.727 57.3677 164.573 57.3418 164.393 57.2642C164.291 57.2124 164.188 57.1089 164.085 57.0054Z" fill="#FFFFFF"/><path d="M175.133 46.2139V47.1196C175.133 47.249 175.184 47.3784 175.287 47.4819C175.39 47.5854 175.493 47.6372 175.647 47.6372H181.376C181.505 47.6372 181.633 47.689 181.736 47.7925C181.839 47.896 181.89 48.0254 181.89 48.1548V51.105C181.89 51.2344 181.839 51.3638 181.736 51.4673C181.633 51.5708 181.505 51.6226 181.376 51.6226H175.647C175.493 51.6226 175.39 51.6743 175.287 51.7778C175.184 51.8813 175.133 52.0107 175.133 52.1401V52.9424C175.133 53.0977 175.184 53.2012 175.287 53.3047C175.39 53.4082 175.493 53.46 175.647 53.46H182.224C182.353 53.46 182.481 53.5117 182.584 53.6152C182.687 53.7187 182.738 53.8481 182.738 53.9775V56.9277C182.738 57.083 182.687 57.1865 182.584 57.29C182.481 57.3936 182.353 57.4453 182.224 57.4453H171.459C171.33 57.4453 171.202 57.3936 171.099 57.29C170.996 57.1865 170.945 57.0571 170.945 56.9277V42.2803C170.945 42.1509 170.996 42.0215 171.099 41.918C171.202 41.8145 171.33 41.7627 171.459 41.7627H182.224C182.353 41.7627 182.481 41.8145 182.584 41.918C182.687 42.0215 182.738 42.1509 182.738 42.2803V45.1787C182.738 45.334 182.687 45.4375 182.584 45.541C182.481 45.6445 182.353 45.6963 182.224 45.6963H175.647C175.493 45.6963 175.39 45.748 175.287 45.8516C175.184 45.9551 175.133 46.0586 175.133 46.2139Z" fill="#FFFFFF"/></svg>
# MAGIC   <svg x="842" y="535" width="49" height="15" viewBox="0 0 168 51.3"><path d="M5.53331 39.1625H0V12.6328H5.53331V39.1625Z" fill="#FFFFFF"/><path d="M10.2703 25.8961C10.2703 17.7871 16.1914 12.1384 24.1021 12.1384C29.1054 12.1384 33.1585 14.3736 35.4322 17.4674L30.6538 20.5908C29.2505 18.6575 26.9028 17.4467 24.0961 17.4467C19.3592 17.4467 16.0227 20.8957 16.0227 25.8991C16.0227 30.9025 19.3207 34.3515 24.0577 34.3515C26.9383 34.3515 29.4755 33.0637 30.9173 30.6745L35.5417 33.9311C33.117 37.5697 28.8331 39.6539 24.0577 39.6539C16.147 39.6539 10.2703 34.0081 10.2703 25.8961Z" fill="#FFFFFF"/><path d="M60.1411 34.0437V39.1595H40.2816V12.6328H59.8362V17.7486H45.8149V23.1665H59.2293V28.3238H45.8149V34.0466L60.1411 34.0437Z" fill="#FFFFFF"/><path d="M87.2007 31.6574C87.2007 35.9384 84.0181 39.1625 78.8637 39.1625H64.875V12.6328H77.9874C82.9907 12.6328 86.0224 15.5519 86.0224 19.7559C86.0224 22.5211 84.6191 24.6053 82.1559 25.4787C85.3414 26.0471 87.2007 28.3238 87.2007 31.6574ZM70.4083 17.3312V23.3589H77.0015C79.1627 23.3589 80.2581 22.1747 80.2581 20.1763C80.2581 18.4 79.1598 17.3342 77.0015 17.3342L70.4083 17.3312ZM81.4364 31.163C81.4364 29.1173 80.2522 27.9804 77.9489 27.9804H70.4083V34.4611H77.9489C80.2611 34.4611 81.4364 33.2502 81.4364 31.166V31.163Z" fill="#FFFFFF"/><path d="M111.791 34.0437V39.1595H91.9376V12.6328H111.477V17.7486H97.465V23.1665H110.885V28.3238H97.465V34.0466L111.791 34.0437Z" fill="#FFFFFF"/><path d="M125.623 29.4962H122.07V39.1595H116.528V12.6328H128.391C134.114 12.6328 137.602 16.0818 137.602 21.0467C137.602 25.0257 135.366 28.0574 131.5 29.1173L138.398 39.1595H131.766L125.623 29.4962ZM122.07 24.3418H128.249C130.635 24.3418 131.888 23.205 131.888 21.0852C131.888 18.9654 130.638 17.7871 128.249 17.7871H122.059L122.07 24.3418Z" fill="#FFFFFF"/><path d="M154.613 39.6539C146.694 39.6539 140.781 34.0081 140.781 25.8606C140.781 17.7131 146.806 12.1413 154.729 12.1413C158.873 12.1413 162.648 13.6571 165.387 16.7272L161.094 20.3273C160.312 19.3995 159.331 18.659 158.225 18.1605C157.119 17.6619 155.915 17.418 154.702 17.4467C149.814 17.4467 146.554 20.9342 146.554 25.8991C146.554 30.864 149.775 34.3515 154.74 34.3515C158.379 34.3515 160.958 32.4183 161.828 28.9692H154.702V24.3803H167.853C167.949 25.02 167.987 25.6671 167.965 26.3136C168.024 34.1976 162.42 39.6539 154.613 39.6539Z" fill="#FFFFFF"/></svg>
# MAGIC   <text x="1044" y="432" class="body" font-size="15">Unified, open-format data lake</text>
# MAGIC   <text x="1044" y="452" class="body" font-size="15">for ALL business data</text>
# MAGIC   <text x="1044" y="484" class="body" font-size="15">Single copy of data,</text>
# MAGIC   <text x="1044" y="504" class="body" font-size="15">no copying or sync</text>
# MAGIC   <path d="M804,598 L804,652" fill="none" stroke="#A9C4D6" stroke-width="1.5"/>
# MAGIC   <rect x="96" y="652" width="1408" height="164" rx="20" fill="#FFFFFF" filter="url(#softShadow)"/>
# MAGIC   <rect x="110" y="664" width="1380" height="76" rx="14" fill="#EBF1F6" stroke="#C6DCEA" stroke-width="1.2"/>
# MAGIC   <text x="132" y="710" class="hb" font-size="23">Agentic Data</text>
# MAGIC   <rect x="436" y="678" width="48" height="48" rx="13" fill="#FFFFFF" filter="url(#tileShadow)"/>
# MAGIC   <svg x="436" y="678" width="48" height="48" viewBox="192 59 512 512"><path d="M192 196.45C192 120.538 253.538 59.0001 329.45 59.0001H566.55C642.462 59.0001 704 120.538 704 196.45V433.55C704 509.462 642.462 571 566.55 571H329.45C253.538 571 192 509.462 192 433.55V196.45Z" fill="white"/><path d="M464.872 376.037H592.092V462.101H445.293L393.629 376.037H303.304V289.973L413.206 289.975L464.872 376.037Z" fill="#FF5F46"/><path d="M501.768 259.377L592.092 259.377V345.443L482.191 345.441L430.525 259.38L303.304 259.377V173.313L450.103 173.315L501.768 259.377Z" fill="#FABFBA"/></svg>
# MAGIC   <text x="500" y="710" class="hb" font-size="23">Lakeflow</text>
# MAGIC   <rect x="772" y="678" width="48" height="48" rx="13" fill="#FFFFFF" filter="url(#tileShadow)"/>
# MAGIC   <svg x="772" y="678" width="48" height="48" viewBox="191 59 512 512"><path d="M191.359 196.33C191.359 120.418 252.898 58.88 328.809 58.88H565.91C641.821 58.88 703.359 120.418 703.359 196.33V433.43C703.359 509.342 641.821 570.88 565.91 570.88H328.809C252.898 570.88 191.359 509.342 191.359 433.43V196.33Z" fill="white"/><path d="M591.208 281.742V362.52H591.214L591.208 362.523V362.557H591.133C569.727 372.414 545.232 378.047 519.084 378.047C492.937 378.047 468.416 372.414 447.037 362.557H447.051C425.67 352.676 401.139 347.031 374.98 347.031C348.821 347.031 324.317 352.676 302.909 362.557H302.906V282.843H302.922V281.742L446.411 169.092L591.208 281.742Z" fill="#FABFBA"/><path d="M375.031 378.05C401.179 378.05 425.701 383.683 447.081 393.541H447.064C468.445 403.421 492.976 409.067 519.135 409.067C544.477 409.067 568.265 403.768 589.191 394.455L591.206 393.541V463.688H591.212C591.21 463.689 591.208 463.69 591.206 463.691V463.723H591.134C569.727 473.581 545.23 479.214 519.082 479.214C492.934 479.214 468.411 473.581 447.031 463.723H447.049C425.668 453.843 401.137 448.197 374.978 448.197C349.636 448.197 325.847 453.495 304.922 462.809L302.907 463.723H302.904L302.905 393.576V393.541H302.98C324.387 383.683 348.883 378.05 375.031 378.05Z" fill="#FF5F46"/></svg>
# MAGIC   <text x="836" y="710" class="hb" font-size="23">Lakehouse</text>
# MAGIC   <rect x="1088" y="678" width="48" height="48" rx="13" fill="#FFFFFF" filter="url(#tileShadow)"/>
# MAGIC   <svg x="1088" y="678" width="48" height="48" viewBox="192 59 512 512"><path d="M192 196.45C192 120.538 253.538 59 329.45 59H566.55C642.462 59 704 120.538 704 196.45V433.55C704 509.462 642.462 571 566.55 571H329.45C253.538 571 192 509.462 192 433.55V196.45Z" fill="white"/><path d="M592.094 355.275H592.091C570.646 362.487 546.1 366.607 519.896 366.607C493.693 366.607 469.12 362.487 447.702 355.275H303.305V278.516H592.094V355.275ZM375.502 343.466C349.297 343.466 324.75 347.76 303.305 355.274H447.699C426.281 347.759 401.707 343.466 375.502 343.466Z" fill="#FABFBA"/><path d="M375.841 371.695C401.108 371.695 424.851 375.365 445.691 381.815L449.707 383.081C470.548 389.532 494.291 393.201 519.558 393.201C544.825 393.201 568.543 389.532 589.406 383.081L591.416 382.448H592.094V462.102H303.305V382.448H303.982C325.327 375.605 349.759 371.695 375.841 371.695Z" fill="#FF5F46"/><rect x="303.304" y="173.312" width="288.79" height="79.6536" fill="#FABFBA"/></svg>
# MAGIC   <text x="1152" y="710" class="hb" font-size="23">Lakebase</text>
# MAGIC   <rect x="110" y="752" width="1380" height="50" rx="12" fill="#EBF1F6" stroke="#C6DCEA" stroke-width="1.2"/>
# MAGIC   <text x="132" y="782" class="hb" font-size="19">Open Infrastructure</text>
# MAGIC   <text x="486" y="781" class="lbl" font-size="14">Open Format Data Lake</text>
# MAGIC   <svg x="676" y="761" width="97" height="20" viewBox="6.5 31.4 176.3 36.4"><path d="M43.4295 61.3529L26.3179 31.8768C25.9325 31.2299 24.9818 31.2299 24.5964 31.8768L10.0541 56.9276C9.79712 57.3676 10.2596 57.8851 10.7221 57.6522C10.9019 57.5746 11.1332 57.4711 11.3644 57.3934C11.3644 57.3934 11.7498 57.2382 12.4435 57.0311C13.1886 56.7982 13.9851 56.5912 14.7302 56.41C16.7857 55.9442 19.4835 55.556 21.8986 55.9442C24.1339 56.3065 25.7012 56.9276 26.8317 57.5228C27.3456 57.8075 27.8338 57.1088 27.4227 56.6947C25.804 55.1161 23.2861 53.1752 19.4835 52.3729C19.3293 52.347 19.2265 52.14 19.3293 52.0106L25.0332 42.1508C25.213 41.8143 25.6755 41.8143 25.8811 42.1508L36.8264 60.9906C37.0576 61.3788 36.8521 61.8964 36.441 62.0516C35.5931 62.3363 34.5654 62.5433 33.4092 62.5433C32.6898 62.5433 31.9447 62.4657 31.1996 62.2845C31.1996 62.2845 29.0927 61.9481 26.2408 60.2401C23.4145 58.5321 17.9419 56.3583 11.8269 59.8778C9.20618 61.5599 7.45904 63.889 6.53408 65.3383C6.37992 65.5712 6.63686 65.83 6.8681 65.7006C8.97494 64.5619 14.6531 61.8964 19.7661 62.8539C22.7979 63.4232 26.2922 65.7264 29.8892 67.0721C35.8757 69.2977 41.7338 65.5712 41.7338 65.5453C42.3504 65.1312 42.7101 64.8207 42.7872 64.7689L42.8643 64.6913C43.8406 63.8632 44.0719 62.4657 43.4295 61.3529Z" fill="#00ADD4"/><path d="M49.1079 56.876V42.2803C49.1079 42.1509 49.1593 42.0215 49.2621 41.918C49.3648 41.8145 49.4933 41.7627 49.6218 41.7627H55.4028C57.6124 41.7627 59.4366 42.4097 60.8497 43.7295C61.6205 44.4282 62.1858 45.2822 62.5969 46.2656C62.9823 47.249 63.1878 48.3618 63.1878 49.5781C63.1878 50.7944 62.9823 51.9072 62.5969 52.8906C62.2115 53.874 61.6205 54.728 60.8497 55.4268C60.156 56.0737 59.3338 56.5395 58.4346 56.876C57.5353 57.2124 56.5076 57.3677 55.4284 57.3677H49.6475C49.519 57.3677 49.3905 57.3159 49.2878 57.2124C49.1593 57.1348 49.1079 57.0054 49.1079 56.876ZM53.8098 53.46H55.4028C55.7882 53.46 56.1992 53.4082 56.636 53.3047C57.0471 53.2012 57.4325 53.02 57.7665 52.7354C58.1005 52.4507 58.3832 52.0625 58.6144 51.5708C58.82 51.0532 58.9484 50.4063 58.9484 49.604C58.9484 48.8018 58.8456 48.1289 58.6144 47.6372C58.3832 47.1196 58.1005 46.7314 57.7665 46.4727C57.4325 46.188 57.0471 46.0068 56.636 45.9033C56.2249 45.7998 55.8138 45.748 55.4284 45.748H53.8098C53.6813 45.748 53.5528 45.7998 53.4501 45.9033C53.3473 46.0068 53.2959 46.1362 53.2959 46.2656V52.9683C53.2959 53.1235 53.3473 53.227 53.4501 53.3306C53.5271 53.4082 53.6556 53.46 53.8098 53.46Z" fill="#302D2D"/><path d="M69.2256 46.2139V47.1196C69.2256 47.249 69.277 47.3784 69.3797 47.4819C69.4825 47.5854 69.611 47.6372 69.7394 47.6372H75.469C75.5975 47.6372 75.726 47.689 75.8287 47.7925C75.9315 47.896 75.9829 48.0254 75.9829 48.1548V51.105C75.9829 51.2344 75.9315 51.3638 75.8287 51.4673C75.726 51.5708 75.5975 51.6226 75.469 51.6226H69.7394C69.611 51.6226 69.4825 51.6743 69.3797 51.7778C69.277 51.8813 69.2256 52.0107 69.2256 52.1401V52.9424C69.2256 53.0977 69.277 53.2012 69.3797 53.3047C69.4825 53.4082 69.611 53.46 69.7394 53.46H76.3169C76.4454 53.46 76.5738 53.5117 76.6766 53.6152C76.7794 53.7187 76.8308 53.8481 76.8308 53.9775V56.9277C76.8308 57.083 76.7794 57.1865 76.6766 57.29C76.5738 57.3936 76.4454 57.4453 76.3169 57.4453H65.5515C65.423 57.4453 65.2945 57.3936 65.1918 57.29C65.089 57.1865 65.0376 57.0571 65.0376 56.9277V42.2803C65.0376 42.1509 65.089 42.0215 65.1918 41.918C65.2945 41.8145 65.423 41.7627 65.5515 41.7627H76.3169C76.4454 41.7627 76.5738 41.8145 76.6766 41.918C76.7794 42.0215 76.8308 42.1509 76.8308 42.2803V45.1787C76.8308 45.334 76.7794 45.4375 76.6766 45.541C76.5738 45.6445 76.4454 45.6963 76.3169 45.6963H69.7394C69.611 45.6963 69.4825 45.748 69.3797 45.8516C69.2513 45.9551 69.2256 46.0586 69.2256 46.2139Z" fill="#302D2D"/><path d="M91.0649 53.9775V56.876C91.0649 57.0312 91.0135 57.1348 90.9107 57.2383C90.8079 57.3418 90.6795 57.3936 90.551 57.3936H79.9911C79.8369 57.3936 79.7085 57.3418 79.6314 57.2383C79.5286 57.1348 79.5029 57.0054 79.5029 56.876V42.2803C79.5029 42.1509 79.5543 42.0215 79.6314 41.918C79.7342 41.8145 79.8369 41.7627 79.9911 41.7627H83.1771C83.3055 41.7627 83.434 41.8145 83.5368 41.918C83.6395 42.0215 83.6909 42.1509 83.6909 42.2803V52.9424C83.6909 53.0977 83.7423 53.2012 83.8194 53.3047C83.9222 53.4082 84.0249 53.46 84.1791 53.46H90.551C90.6795 53.46 90.8079 53.5117 90.9107 53.6152C91.0135 53.7187 91.0649 53.8223 91.0649 53.9775Z" fill="#302D2D"/><path d="M100.905 45.6963H97.9507C97.8222 45.6963 97.6938 45.748 97.591 45.8516C97.4882 45.9551 97.4369 46.0845 97.4369 46.2139V56.876C97.4369 57.0312 97.3855 57.1348 97.3084 57.2383C97.2056 57.3418 97.1028 57.3936 96.9487 57.3936H93.7627C93.6343 57.3936 93.5058 57.3418 93.403 57.2383C93.3003 57.1348 93.2489 57.0054 93.2489 56.876V46.2139C93.2489 46.0845 93.1975 45.9551 93.1204 45.8516C93.0176 45.748 92.9149 45.6963 92.7607 45.6963H89.7546C89.6004 45.6963 89.4977 45.6445 89.3949 45.541C89.2921 45.4375 89.2407 45.3081 89.2407 45.1787V42.2803C89.2407 42.1509 89.2921 42.0215 89.3949 41.918C89.4977 41.8145 89.6004 41.7627 89.7546 41.7627H100.931C101.085 41.7627 101.214 41.8145 101.291 41.918C101.394 42.0215 101.419 42.1509 101.419 42.2803V45.1787C101.419 45.334 101.368 45.4375 101.291 45.541C101.188 45.6445 101.06 45.6963 100.905 45.6963Z" fill="#302D2D"/><path d="M105.967 42.2539C106.018 42.1245 106.121 42.021 106.25 41.9175C106.378 41.814 106.532 41.7881 106.687 41.7881H109.641C109.77 41.7881 109.924 41.8398 110.052 41.9175C110.181 42.021 110.284 42.1245 110.335 42.2539L116.527 56.7461C116.578 56.8755 116.578 57.0049 116.501 57.1602C116.424 57.2896 116.27 57.3672 116.039 57.3672H112.416C112.262 57.3672 112.108 57.3154 111.979 57.2378C111.851 57.1343 111.748 57.0308 111.671 56.9014L111.106 55.6333C111.029 55.5039 110.926 55.4004 110.797 55.2969C110.669 55.1934 110.515 55.1675 110.386 55.1675H105.916C105.762 55.1675 105.607 55.2192 105.479 55.2969C105.35 55.4004 105.248 55.5039 105.196 55.6333L104.605 56.9014C104.554 57.0308 104.451 57.1343 104.323 57.2378C104.194 57.3413 104.04 57.3672 103.912 57.3672H100.263C100.186 57.3672 100.109 57.3413 100.032 57.3154C99.9549 57.2896 99.9035 57.2378 99.8521 57.186C99.8007 57.1343 99.775 57.0566 99.7493 56.979C99.7236 56.9014 99.7493 56.7979 99.8007 56.6685L105.967 42.2539ZM107.226 51.5444H109.102C109.23 51.5444 109.333 51.4927 109.41 51.415C109.461 51.3115 109.461 51.208 109.41 51.0786C109.282 50.7163 109.102 50.3022 108.896 49.8364C108.691 49.3706 108.511 48.9048 108.357 48.4907C108.305 48.3613 108.228 48.3096 108.177 48.3096C108.1 48.3096 108.048 48.3613 107.997 48.4907L106.943 51.0786C106.892 51.208 106.892 51.3115 106.943 51.415C106.969 51.4927 107.072 51.5444 107.226 51.5444Z" fill="#302D2D"/><path d="M136.825 53.9775V56.876C136.825 57.0312 136.773 57.1348 136.671 57.2383C136.568 57.3418 136.439 57.3936 136.311 57.3936H125.751C125.597 57.3936 125.468 57.3418 125.391 57.2383C125.288 57.1348 125.263 57.0054 125.263 56.876V42.2803C125.263 42.1509 125.314 42.0215 125.391 41.918C125.494 41.8145 125.597 41.7627 125.751 41.7627H128.937C129.065 41.7627 129.194 41.8145 129.297 41.918C129.399 42.0215 129.451 42.1509 129.451 42.2803V52.9424C129.451 53.0977 129.502 53.2012 129.579 53.3047C129.682 53.4082 129.785 53.46 129.939 53.46H136.311C136.439 53.46 136.568 53.5117 136.671 53.6152C136.773 53.7187 136.825 53.8223 136.825 53.9775Z" fill="#302D2D"/><path d="M143.351 42.2539C143.402 42.1245 143.505 42.021 143.633 41.9175C143.762 41.814 143.916 41.7881 144.07 41.7881H147.025C147.153 41.7881 147.308 41.8398 147.436 41.9175C147.565 42.021 147.667 42.1245 147.719 42.2539L153.911 56.7461C153.962 56.8755 153.962 57.0049 153.885 57.1602C153.808 57.2896 153.654 57.3672 153.423 57.3672H149.8C149.646 57.3672 149.492 57.3154 149.363 57.2378C149.235 57.1343 149.132 57.0308 149.055 56.9014L148.49 55.6333C148.412 55.5039 148.31 55.4004 148.181 55.2969C148.053 55.1934 147.899 55.1675 147.77 55.1675H143.299C143.145 55.1675 142.991 55.2192 142.863 55.2969C142.734 55.4004 142.631 55.5039 142.58 55.6333L141.989 56.9014C141.938 57.0308 141.835 57.1343 141.707 57.2378C141.578 57.3413 141.424 57.3672 141.295 57.3672H137.647C137.57 57.3672 137.493 57.3413 137.416 57.3154C137.339 57.2896 137.287 57.2378 137.236 57.186C137.185 57.1343 137.159 57.0566 137.133 56.979C137.107 56.9014 137.133 56.7979 137.185 56.6685L143.351 42.2539ZM144.584 51.5444H146.46C146.588 51.5444 146.691 51.4927 146.768 51.415C146.819 51.3115 146.819 51.208 146.768 51.0786C146.64 50.7163 146.46 50.3022 146.254 49.8364C146.049 49.3706 145.869 48.9048 145.715 48.4907C145.663 48.3613 145.586 48.3096 145.535 48.3096C145.483 48.3096 145.406 48.3613 145.355 48.4907L144.302 51.0786C144.25 51.208 144.25 51.3115 144.302 51.415C144.327 51.4927 144.43 51.5444 144.584 51.5444Z" fill="#302D2D"/><path d="M164.085 57.0054L159.974 52.373C159.871 52.2695 159.794 52.2178 159.743 52.2437C159.692 52.2695 159.64 52.3472 159.64 52.5024V56.876C159.64 57.0312 159.589 57.1348 159.486 57.2383C159.383 57.3418 159.255 57.3936 159.126 57.3936H155.94C155.786 57.3936 155.658 57.3418 155.581 57.2383C155.478 57.1348 155.452 57.0054 155.452 56.876V42.2803C155.452 42.1509 155.504 42.0215 155.581 41.918C155.683 41.8145 155.786 41.7627 155.94 41.7627H159.126C159.255 41.7627 159.383 41.8145 159.486 41.918C159.589 42.0215 159.64 42.1509 159.64 42.2803V47.1973C159.64 47.3525 159.666 47.4302 159.717 47.4302C159.769 47.4561 159.846 47.4043 159.923 47.3008L163.725 42.2026C163.802 42.0991 163.931 41.9956 164.085 41.918C164.239 41.8403 164.393 41.7886 164.548 41.7886H168.581C168.736 41.7886 168.838 41.8662 168.89 41.9956C168.941 42.125 168.915 42.2544 168.838 42.3579L163.289 49.397C163.211 49.5005 163.16 49.6299 163.16 49.7852C163.16 49.9404 163.212 50.0698 163.314 50.1475C163.802 50.665 164.291 51.2085 164.83 51.7778C165.37 52.3472 165.909 52.9165 166.423 53.4858C166.963 54.0552 167.477 54.6245 167.99 55.1938C168.504 55.7632 168.992 56.2808 169.455 56.7725C169.558 56.876 169.583 57.0054 169.558 57.1606C169.532 57.29 169.429 57.3677 169.275 57.3677H164.882C164.727 57.3677 164.573 57.3418 164.393 57.2642C164.291 57.2124 164.188 57.1089 164.085 57.0054Z" fill="#302D2D"/><path d="M175.133 46.2139V47.1196C175.133 47.249 175.184 47.3784 175.287 47.4819C175.39 47.5854 175.493 47.6372 175.647 47.6372H181.376C181.505 47.6372 181.633 47.689 181.736 47.7925C181.839 47.896 181.89 48.0254 181.89 48.1548V51.105C181.89 51.2344 181.839 51.3638 181.736 51.4673C181.633 51.5708 181.505 51.6226 181.376 51.6226H175.647C175.493 51.6226 175.39 51.6743 175.287 51.7778C175.184 51.8813 175.133 52.0107 175.133 52.1401V52.9424C175.133 53.0977 175.184 53.2012 175.287 53.3047C175.39 53.4082 175.493 53.46 175.647 53.46H182.224C182.353 53.46 182.481 53.5117 182.584 53.6152C182.687 53.7187 182.738 53.8481 182.738 53.9775V56.9277C182.738 57.083 182.687 57.1865 182.584 57.29C182.481 57.3936 182.353 57.4453 182.224 57.4453H171.459C171.33 57.4453 171.202 57.3936 171.099 57.29C170.996 57.1865 170.945 57.0571 170.945 56.9277V42.2803C170.945 42.1509 170.996 42.0215 171.099 41.918C171.202 41.8145 171.33 41.7627 171.459 41.7627H182.224C182.353 41.7627 182.481 41.8145 182.584 41.918C182.687 42.0215 182.738 42.1509 182.738 42.2803V45.1787C182.738 45.334 182.687 45.4375 182.584 45.541C182.481 45.6445 182.353 45.6963 182.224 45.6963H175.647C175.493 45.6963 175.39 45.748 175.287 45.8516C175.184 45.9551 175.133 46.0586 175.133 46.2139Z" fill="#302D2D"/></svg>
# MAGIC   <svg x="803" y="764" width="74" height="17" viewBox="0 0 224 51.3"><path d="M219.858 12.0673C221.827 14.267 222.7 17.0055 223.325 19.7648C226.537 33.9756 217.996 47.6386 203.764 50.6969C188.21 54.0394 174.997 43.177 173.013 29.5702C172.134 23.9752 173.141 18.2467 175.876 13.2871C176.128 12.8222 176.293 12.2657 176.877 12.0584L184.222 12.1798L181.717 17.4763C181.717 17.6579 181.717 17.8394 181.717 18.021C183.526 23.2701 186.898 27.5837 190.098 31.9979C190.442 33.8719 190.187 35.8466 191.022 37.6378C193.355 40.2727 195.687 42.9096 198.018 45.5484C198.19 45.7438 198.397 45.8682 198.61 45.5781C199.676 44.9267 199.987 43.7307 200.576 42.7566C201.914 40.554 203.368 38.4105 204.425 36.045L204.75 35.5299C206.059 35.0206 206.885 33.8483 208.043 33.1348C208.43 32.9221 208.746 32.6001 208.952 32.209C209.158 31.8179 209.244 31.3748 209.2 30.935C209.132 28.7235 209.176 26.4942 209.173 24.2915C209.698 23.7726 210.192 23.224 210.654 22.6484C211.782 21.0645 213.558 19.9839 214.221 18.0477C214.338 18.0629 214.457 18.0429 214.563 17.9901C214.668 17.9374 214.755 17.8543 214.813 17.7516C215.281 17.0411 214.778 16.446 214.574 15.8213C214.183 14.6193 213.434 13.5357 213.357 12.2272L219.858 12.0673Z" fill="#287ABE"/><path d="M184.225 12.1798L176.88 12.0584C177.848 9.53897 179.763 7.73598 181.741 6.07806C185.145 3.22111 189.05 1.24936 193.459 0.452969C201.127 -0.911856 208.102 0.749027 214.275 5.52739C216.575 7.30374 218.662 9.31693 219.864 12.0584L213.351 12.2005L208.022 12.1354V10.3383L211.841 10.3176L211.965 10.2495L211.844 10.0482C211.201 9.41788 210.433 8.92919 209.589 8.61357C208.746 8.29794 207.846 8.16238 206.947 8.2156C201.695 8.33994 196.44 8.26593 191.188 8.25408C189.346 8.25408 187.813 9.0416 186.336 10.0156C184.974 10.0748 184.95 11.2383 184.598 12.1265L184.225 12.1798Z" fill="#B8E1F2"/><path d="M181.717 18.0151C181.717 17.8335 181.717 17.6519 181.717 17.4704C184.787 16.5585 187.606 15.0427 190.495 13.7104C191.41 13.2871 192.671 13.3226 193.16 12.1502C194.918 12.2242 196.712 11.8541 198.438 12.3811C198.452 16.5259 198.466 20.6806 198.48 24.8451H198.208C194.835 23.4033 191.487 21.9053 188.085 20.5405C185.974 19.7026 184.041 18.3644 181.717 18.0151Z" fill="#B6E0F2"/><path d="M181.717 18.0151C184.038 18.3615 185.971 19.6967 188.094 20.5494C191.496 21.9142 194.844 23.4122 198.216 24.854C197.947 26.1182 196.695 26.4202 195.907 27.1633C194.11 28.8626 191.961 30.1564 190.247 31.9594C190.2 31.9806 190.15 31.9927 190.099 31.9949C186.901 27.5807 183.529 23.2701 181.717 18.0151Z" fill="#90D4F0"/><path d="M209.176 24.2797C209.176 26.4942 209.135 28.7205 209.203 30.9232C209.247 31.363 209.161 31.806 208.955 32.1971C208.749 32.5883 208.433 32.9102 208.046 33.1229C206.888 33.8364 206.062 35.0088 204.753 35.518C203.356 33.7417 202.486 31.6219 201.233 29.7567C200.381 28.4896 200.028 26.8554 198.569 26.0264C198.569 25.7461 198.574 25.4659 198.584 25.1856L199.149 24.8895C199.404 25.2034 199.762 25.1619 200.093 25.1323C203.115 24.856 206.143 24.5718 209.176 24.2797Z" fill="#4D8AC8"/><path d="M191.022 37.6407C190.187 35.8466 190.43 33.8749 190.101 31.9979C190.153 31.9956 190.203 31.9836 190.249 31.9624C192.914 32.0749 195.578 32.143 198.225 32.5545C198.225 34.6269 198.22 36.6993 198.21 38.7717C197.676 38.751 197.144 38.6936 196.618 38.6C194.747 38.3098 192.941 37.6467 191.022 37.6407Z" fill="#B6E0F2"/><path d="M198.208 38.7687C198.208 36.6963 198.212 34.6239 198.222 32.5515C198.835 30.4021 198.317 28.1936 198.557 26.0383C200.019 26.8672 200.375 28.5044 201.221 29.7686C202.477 31.6367 203.344 33.7446 204.742 35.5299L204.416 36.045L198.462 38.7835C198.377 38.787 198.292 38.7821 198.208 38.7687Z" fill="#58AFDD"/><path d="M208.007 12.1443L213.336 12.2094C213.413 13.518 214.162 14.6016 214.553 15.8036C214.757 16.4312 215.26 17.0233 214.793 17.7338C214.676 17.7245 214.56 17.7471 214.455 17.7993C214.351 17.8515 214.263 17.9312 214.2 18.0299C213.09 18.2371 212.128 18.8263 211.124 19.2763C210.121 19.7263 209.23 20.4783 208.042 20.4457C208.042 17.6983 208.038 14.9519 208.028 12.2065L208.007 12.1443Z" fill="#4D8AC8"/><path d="M208.048 20.4487C209.233 20.4813 210.121 19.7263 211.13 19.2793C212.14 18.8322 213.096 18.2401 214.206 18.0329C213.54 19.972 211.767 21.0497 210.639 22.6336C210.177 23.2092 209.683 23.7578 209.159 24.2767C206.129 24.5629 203.099 24.8481 200.07 25.1323C199.738 25.1619 199.38 25.2034 199.125 24.8896C201.218 23.9214 203.326 22.9859 205.396 21.9704C206.313 21.5263 207.376 21.3161 208.048 20.4487Z" fill="#266FA8"/><path d="M198.208 38.7687C198.295 38.7778 198.383 38.7778 198.471 38.7687C198.518 41.0346 198.565 43.3043 198.61 45.578C198.397 45.8741 198.193 45.7409 198.018 45.5484C195.683 42.9175 193.351 40.2845 191.022 37.6496C192.938 37.6496 194.744 38.3187 196.615 38.6089C197.141 38.6985 197.674 38.7519 198.208 38.7687Z" fill="#90D4F0"/><path d="M193.172 12.1561C192.671 13.3404 191.41 13.293 190.507 13.7163C187.618 15.0486 184.799 16.5644 181.729 17.4763L184.237 12.1798L184.604 12.1324L193.172 12.1561Z" fill="#90D4F0"/><path d="M198.607 45.5692C198.562 43.3033 198.515 41.0336 198.468 38.7598L204.422 36.0213C203.365 38.3898 201.911 40.5303 200.573 42.7329C199.999 43.7188 199.673 44.9149 198.607 45.5692Z" fill="#266FA8"/><path d="M214.209 18.0388C214.272 17.9401 214.36 17.8604 214.464 17.8082C214.569 17.756 214.685 17.7334 214.802 17.7427C214.744 17.8454 214.656 17.9285 214.551 17.9812C214.445 18.034 214.326 18.054 214.209 18.0388Z" fill="#266FA8"/><path d="M186.327 10.0215C187.807 9.04752 189.338 8.2452 191.179 8.26001C196.434 8.26001 201.689 8.34586 206.938 8.22152C207.838 8.16785 208.74 8.30347 209.584 8.61963C210.429 8.93579 211.197 9.42547 211.841 10.0571C211.841 10.1479 211.841 10.2387 211.841 10.3294L207.992 10.3502H188.186C187.547 10.3502 186.901 10.3887 186.327 10.0215Z" fill="#F4FAFD"/><path d="M186.327 10.0215C186.901 10.3887 187.547 10.3472 188.186 10.3472H207.992V12.1443L208.016 12.2094L198.45 12.39C196.709 11.866 194.93 12.2331 193.172 12.1591L184.586 12.1384C184.944 11.2443 184.965 10.0808 186.327 10.0215Z" fill="#E9F6FD"/><path d="M211.841 10.3176C211.841 10.2268 211.841 10.136 211.841 10.0452L211.962 10.2465L211.841 10.3176Z" fill="#E9F6FD"/><path d="M198.45 12.39L208.034 12.2153C208.034 14.9627 208.039 17.7092 208.048 20.4546C207.376 21.3221 206.316 21.5323 205.408 21.9793C203.335 22.9948 201.23 23.9303 199.137 24.8984L198.572 25.1945L198.492 24.8599L198.45 12.39Z" fill="#68C1EA"/><path d="M5.53331 39.1625H0V12.6328H5.53331V39.1625Z" fill="#2C7CBF"/><path d="M10.2703 25.8961C10.2703 17.7871 16.1914 12.1384 24.1021 12.1384C29.1054 12.1384 33.1585 14.3736 35.4322 17.4674L30.6538 20.5908C29.2505 18.6575 26.9028 17.4467 24.0961 17.4467C19.3592 17.4467 16.0227 20.8957 16.0227 25.8991C16.0227 30.9025 19.3207 34.3515 24.0577 34.3515C26.9383 34.3515 29.4755 33.0637 30.9173 30.6745L35.5417 33.9311C33.117 37.5697 28.8331 39.6539 24.0577 39.6539C16.147 39.6539 10.2703 34.0081 10.2703 25.8961Z" fill="#2C7CBF"/><path d="M60.1411 34.0437V39.1595H40.2816V12.6328H59.8362V17.7486H45.8149V23.1665H59.2293V28.3238H45.8149V34.0466L60.1411 34.0437Z" fill="#2C7CBF"/><path d="M87.2007 31.6574C87.2007 35.9384 84.0181 39.1625 78.8637 39.1625H64.875V12.6328H77.9874C82.9907 12.6328 86.0224 15.5519 86.0224 19.7559C86.0224 22.5211 84.6191 24.6053 82.1559 25.4787C85.3414 26.0471 87.2007 28.3238 87.2007 31.6574ZM70.4083 17.3312V23.3589H77.0015C79.1627 23.3589 80.2581 22.1747 80.2581 20.1763C80.2581 18.4 79.1598 17.3342 77.0015 17.3342L70.4083 17.3312ZM81.4364 31.163C81.4364 29.1173 80.2522 27.9804 77.9489 27.9804H70.4083V34.4611H77.9489C80.2611 34.4611 81.4364 33.2502 81.4364 31.166V31.163Z" fill="#2C7CBF"/><path d="M111.791 34.0437V39.1595H91.9376V12.6328H111.477V17.7486H97.465V23.1665H110.885V28.3238H97.465V34.0466L111.791 34.0437Z" fill="#2C7CBF"/><path d="M125.623 29.4962H122.07V39.1595H116.528V12.6328H128.391C134.114 12.6328 137.602 16.0818 137.602 21.0467C137.602 25.0257 135.366 28.0574 131.5 29.1173L138.398 39.1595H131.766L125.623 29.4962ZM122.07 24.3418H128.249C130.635 24.3418 131.888 23.205 131.888 21.0852C131.888 18.9654 130.638 17.7871 128.249 17.7871H122.059L122.07 24.3418Z" fill="#2C7CBF"/><path d="M154.613 39.6539C146.694 39.6539 140.781 34.0081 140.781 25.8606C140.781 17.7131 146.806 12.1413 154.729 12.1413C158.873 12.1413 162.648 13.6571 165.387 16.7272L161.094 20.3273C160.312 19.3995 159.331 18.659 158.225 18.1605C157.119 17.6619 155.915 17.418 154.702 17.4467C149.814 17.4467 146.554 20.9342 146.554 25.8991C146.554 30.864 149.775 34.3515 154.74 34.3515C158.379 34.3515 160.958 32.4183 161.828 28.9692H154.702V24.3803H167.853C167.949 25.02 167.987 25.6671 167.965 26.3136C168.024 34.1976 162.42 39.6539 154.613 39.6539Z" fill="#2C7CBF"/></svg>
# MAGIC   <text x="1058" y="781" class="meta" font-size="14">Any Cloud</text>
# MAGIC   <svg x="1132" y="766" width="15" height="15" viewBox="0 0 16 16"><path d="M3.39373 5.58639C3.91293 3.52534 5.77786 2 8 2C10.5504 2 12.6314 4.01005 12.7451 6.5324C14.1591 6.7189 15.3247 7.69323 15.7866 9H14.1211C13.7175 8.39701 13.0301 8 12.25 8H12C11.5858 8 11.25 7.66421 11.25 7.25V6.75C11.25 4.95507 9.79493 3.5 8 3.5C6.34131 3.5 4.97186 4.74324 4.7745 6.34833C4.73041 6.70685 4.43704 6.98301 4.07651 7.00536C2.63892 7.09448 1.5 8.28952 1.5 9.75C1.5 11.1845 2.59873 12.3629 4 12.4888V14C3.93845 14 3.87864 13.9926 3.8214 13.9786C1.67511 13.7633 0 11.9526 0 9.75C0 7.69604 1.45669 5.98279 3.39373 5.58639Z" fill="#2D6FC7"/><path fill-rule="evenodd" clip-rule="evenodd" d="M8 7C9.24264 7 10.25 8.00736 10.25 9.25C10.25 9.59736 10.1713 9.92634 10.0307 10.2201L10.5299 10.7193C10.8237 10.5787 11.1526 10.5 11.5 10.5C12.7426 10.5 13.75 11.5074 13.75 12.75C13.75 13.9926 12.7426 15 11.5 15C10.2574 15 9.25 13.9926 9.25 12.75C9.25 12.4026 9.32871 12.0737 9.46928 11.7799L8.97006 11.2807C8.67634 11.4213 8.34736 11.5 8 11.5C6.75736 11.5 5.75 10.4926 5.75 9.25C5.75 8.00736 6.75736 7 8 7ZM8.75 9.25C8.75 8.83579 8.41421 8.5 8 8.5C7.58579 8.5 7.25 8.83579 7.25 9.25C7.25 9.66421 7.58579 10 8 10C8.41421 10 8.75 9.66421 8.75 9.25ZM12.25 12.75C12.25 12.3358 11.9142 12 11.5 12C11.0858 12 10.75 12.3358 10.75 12.75C10.75 13.1642 11.0858 13.5 11.5 13.5C11.9142 13.5 12.25 13.1642 12.25 12.75Z" fill="#2D6FC7"/></svg>
# MAGIC   <text x="1174" y="781" class="meta" font-size="14">Any Model</text>
# MAGIC   <svg x="1252" y="766" width="15" height="15" viewBox="0 0 100 100"><g stroke="#2D6FC7" stroke-width="9" stroke-linecap="round"><line x1="30" y1="36" x2="68" y2="22"/><line x1="30" y1="36" x2="73" y2="58"/><line x1="30" y1="36" x2="43" y2="74"/><line x1="68" y1="22" x2="73" y2="58"/><line x1="73" y1="58" x2="43" y2="74"/></g><circle cx="30" cy="36" r="13" fill="#2D6FC7"/><circle cx="68" cy="22" r="13" fill="#2D6FC7"/><circle cx="73" cy="58" r="13" fill="#2D6FC7"/><circle cx="43" cy="74" r="13" fill="#2D6FC7"/><circle cx="30" cy="36" r="6" fill="#EBF1F6"/><circle cx="68" cy="22" r="6" fill="#EBF1F6"/><circle cx="73" cy="58" r="6" fill="#EBF1F6"/><circle cx="43" cy="74" r="6" fill="#EBF1F6"/></svg>
# MAGIC   <text x="1290" y="781" class="meta" font-size="14">Any Data</text>
# MAGIC   <svg x="1360" y="766" width="15" height="15" viewBox="0 0 16 16"><path fill-rule="evenodd" clip-rule="evenodd" d="M8.64576 0.368453C8.51082 0.140099 8.2653 0 8.00006 0C7.73482 0 7.4893 0.140099 7.35437 0.368453L4.10437 5.86845C3.96736 6.1003 3.96518 6.38781 4.09865 6.62171C4.23213 6.85561 4.48076 7 4.75006 7H11.2501C11.5194 7 11.768 6.85561 11.9015 6.62171C12.0349 6.38781 12.0328 6.1003 11.8958 5.86845L8.64576 0.368453ZM8.00006 2.22426L9.93572 5.5H6.0644L8.00006 2.22426ZM8.5 9.25C8.5 8.83579 8.83579 8.5 9.25 8.5H14.25C14.6642 8.5 15 8.83579 15 9.25V14.25C15 14.6642 14.6642 15 14.25 15H9.25C8.83579 15 8.5 14.6642 8.5 14.25V9.25ZM10 10V13.5H13.5V10H10ZM4.25 10C3.2835 10 2.5 10.7835 2.5 11.75C2.5 12.7165 3.2835 13.5 4.25 13.5C5.2165 13.5 6 12.7165 6 11.75C6 10.7835 5.2165 10 4.25 10ZM1 11.75C1 9.95507 2.45507 8.5 4.25 8.5C6.04493 8.5 7.5 9.95507 7.5 11.75C7.5 13.5449 6.04493 15 4.25 15C2.45507 15 1 13.5449 1 11.75Z" fill="#2D6FC7"/></svg>
# MAGIC </svg>
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC What is Lakebase and what are its key capabilities?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p><strong>Lakebase</strong> is Databricks' fully managed, serverless Postgres database — built directly into the Databricks platform. It separates stateless Postgres compute from a durable cloud storage layer, enabling <strong>autoscaling</strong>, <strong>scale-to-zero</strong>, <strong>instant branching</strong>, and <strong>fast failover</strong>.</p>
# MAGIC
# MAGIC <ul style="margin: 12px 0; padding-left: 24px;">
# MAGIC <li><strong>Transactional workloads</strong> — Build low-latency OLTP applications alongside your lakehouse data, connected via Databricks Apps or any standard Postgres driver.</li>
# MAGIC <li><strong>Lakehouse sync</strong> — Sync Unity Catalog Delta tables into Lakebase so applications can query them at sub-millisecond latency.</li>
# MAGIC <li><strong>AI &amp; agent state</strong> — Use Lakebase as an online feature store for ML models or as a state and memory store for AI agents.</li>
# MAGIC <li><strong>Native Unity Catalog integration</strong> — Governance, permissions, and identity flow seamlessly across the platform.</li>
# MAGIC </ul>
# MAGIC
# MAGIC <p>With that foundation in place, the next question is: how do we search this data efficiently? That's where <strong>Lakebase Search</strong> comes in.</p>
# MAGIC
# MAGIC </div>
# MAGIC </details>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## E. Lakebase Search for Retrieval
# MAGIC
# MAGIC Managed retrieval products like Genie Agents handle document-based RAG end to end. But when your data already lives in an operational database — customer reviews, product records, support tickets — you may want search **inside** that database so a single query can retrieve, JOIN, and enforce permissions in one transaction.
# MAGIC
# MAGIC **Lakebase Search** (GA) adds hybrid vector and keyword search directly to Lakebase Postgres projects through two managed extensions:
# MAGIC
# MAGIC <div style="display: flex; align-items: stretch; gap: 28px; flex-wrap: wrap; justify-content: center; margin: 20px 0;">
# MAGIC
# MAGIC   <!-- lakebase_vector -->
# MAGIC   <div style="border: 1.5px solid #2272B4; border-radius: 12px; padding: 20px 24px; width: 280px; background: #EBF3FA; box-shadow: 0 4px 16px rgba(34,114,180,0.12);">
# MAGIC     <div style="font-size: 12px; font-weight: 700; color: #2272B4; letter-spacing: 0.08em; text-transform: uppercase;">Extension</div>
# MAGIC     <div style="font-size: 19px; font-weight: 700; color: #0b2026; margin-top: 4px;">lakebase_vector</div>
# MAGIC     <div style="font-size: 15px; color: #618794; margin-top: 8px; line-height: 1.5;">Approximate nearest-neighbor (ANN) search via the <code>lakebase_ann</code> index type. pgvector-compatible syntax. Uses IVF + RaBitQ quantization — supports 1B+ vectors per index, builds 50–100× faster than HNSW, and indexes survive scale-to-zero.</div>
# MAGIC   </div>
# MAGIC
# MAGIC   <!-- + sign -->
# MAGIC   <div style="display: flex; align-items: center; font-size: 28px; font-weight: 700; color: #618794;">+</div>
# MAGIC
# MAGIC   <!-- lakebase_text -->
# MAGIC   <div style="border: 1.5px solid #00A972; border-radius: 12px; padding: 20px 24px; width: 280px; background: #E6F5EE; box-shadow: 0 4px 16px rgba(0,169,114,0.12);">
# MAGIC     <div style="font-size: 12px; font-weight: 700; color: #00A972; letter-spacing: 0.08em; text-transform: uppercase;">Extension</div>
# MAGIC     <div style="font-size: 19px; font-weight: 700; color: #0b2026; margin-top: 4px;">lakebase_text</div>
# MAGIC     <div style="font-size: 15px; color: #618794; margin-top: 8px; line-height: 1.5;">BM25 full-text keyword search via the <code>lakebase_bm25</code> index type. Compatible with PostgreSQL's <code>tsvector</code> types and operators. Top-K pushdown (Block-Max WAND) retrieves only the K most relevant results instead of scoring every match.</div>
# MAGIC   </div>
# MAGIC
# MAGIC   <!-- = sign -->
# MAGIC   <div style="display: flex; align-items: center; font-size: 28px; font-weight: 700; color: #618794;">=</div>
# MAGIC
# MAGIC   <!-- Hybrid -->
# MAGIC   <div style="border: 2px solid #7B61FF; border-radius: 12px; padding: 20px 24px; width: 280px; background: #F3F0FF; box-shadow: 0 6px 24px rgba(123,97,255,0.15);">
# MAGIC     <div style="font-size: 12px; font-weight: 700; color: #7B61FF; letter-spacing: 0.08em; text-transform: uppercase;">Combined</div>
# MAGIC     <div style="font-size: 19px; font-weight: 700; color: #7B61FF; margin-top: 4px;">Hybrid Search</div>
# MAGIC     <div style="font-size: 15px; color: #618794; margin-top: 8px; line-height: 1.5;">Run both searches and fuse results with reciprocal rank fusion (RRF). Reviews ranked well by <em>either</em> signal surface, and those ranked well by <em>both</em> rise to the top — the best of exact-term and semantic retrieval.</div>
# MAGIC   </div>
# MAGIC
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Learn More</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC See the Databricks documentation on Lakebase Search (<a href="https://docs.databricks.com/aws/en/oltp/projects/lakebase-search/" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/oltp/projects/lakebase-search" style="color: #1976d2; text-decoration: underline;">Azure</a>) for setup instructions and query examples.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E1. Lakebase Search Modes: Vector, Keyword, and Hybrid
# MAGIC <div class="lbs">
# MAGIC   <style>
# MAGIC     .lbs { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; color: #0b2026; width: 100%; max-width: 100%; margin: 0; padding: 4px 0; box-sizing: border-box; }
# MAGIC     .lbs .diagram { width: 75%; max-width: 100%; margin: 0 auto; box-sizing: border-box; }
# MAGIC     .lbs .grid { display: flex; flex-wrap: nowrap; gap: 18px; margin-top: 18px; width: 100%; }
# MAGIC     .lbs .card { flex: 1 1 0; min-width: 0; background: #fff; border-radius: 14px; box-shadow: 0 2px 8px rgba(27,49,57,0.07); position: relative; padding: 16px 18px 18px; overflow: hidden; box-sizing: border-box; text-align: left; font: inherit; color: inherit; cursor: pointer; transition: box-shadow .15s ease, opacity .15s ease; }
# MAGIC     .lbs .bar { display: block; width: 100%; height: 6px; border-radius: 5px; margin: -4px 0 12px; }
# MAGIC     .lbs .c1 { border: 1.5px solid #C6DCEA; }
# MAGIC     .lbs .c2 { border: 1.5px solid #C6DCEA; }
# MAGIC     .lbs .c3 { border: 1.5px solid #C6DCEA; }
# MAGIC     .lbs .card h3 { margin: 6px 0 4px 0; font-size: 17px; font-weight: 600; }
# MAGIC     .lbs .sub { font-size: 14px; color: #1B5162; margin: 0 0 12px 0; }
# MAGIC     .lbs ul { margin: 0; padding-left: 18px; }
# MAGIC     .lbs li { font-size: 14px; margin-bottom: 11px; line-height: 1.45; }
# MAGIC     .lbs code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 14px; background: #EEEDE9; border-radius: 4px; padding: 1px 5px; }
# MAGIC     .lbs svg { display: block; width: 100%; height: auto; margin: 0 auto; }
# MAGIC     .lbs .hint { font-size: 14px; color: #618794; text-align: center; margin: 10px 0 0 0; }
# MAGIC     /* --- focus states: only active once a card is picked --- */
# MAGIC     .lbs .card.is-dim { opacity: .55; }
# MAGIC     .lbs .card.is-on { box-shadow: 0 0 0 3px rgba(34,114,180,.35), 0 2px 8px rgba(27,49,57,.10); }
# MAGIC     .lbs g[data-part] { transition: opacity .18s ease; }
# MAGIC     .lbs[data-focus] g[data-part] { opacity: .12; }
# MAGIC     .lbs[data-focus="vector"] g[data-part~="vector"] { opacity: 1; }
# MAGIC     .lbs[data-focus="keyword"] g[data-part~="keyword"] { opacity: 1; }
# MAGIC     .lbs[data-focus="hybrid"] g[data-part~="hybrid"] { opacity: 1; }
# MAGIC     @media (max-width: 900px) { .lbs .diagram { width: 100%; } .lbs .grid { flex-wrap: wrap; } .lbs .card { flex: 1 1 280px; } }
# MAGIC   </style>
# MAGIC   <div class="diagram">
# MAGIC     <svg viewBox="0 0 1000 680" preserveAspectRatio="xMidYMid meet" role="img" aria-label="One shared query, shown beneath the Query box, splits into vector search in blue (linked to an embedding-space example at top right) and keyword search in orange (linked to an exact-term example at bottom left), then both merge into a purple hybrid step and an amber ranked list. A hybrid offshoot at bottom right shows the top vector hit and top keyword hit feeding a rank fusion sub-container that holds one ranked list with both kinds of match.">
# MAGIC       <defs>
# MAGIC         <marker id="lbsArrowV" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M 0 0 L 10 5 L 0 10 z" fill="#4299E0"></path>
# MAGIC         </marker>
# MAGIC         <marker id="lbsArrowK" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M 0 0 L 10 5 L 0 10 z" fill="#FF5F46"></path>
# MAGIC         </marker>
# MAGIC         <marker id="lbsArrowG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M 0 0 L 10 5 L 0 10 z" fill="#7B61FF"></path>
# MAGIC         </marker>
# MAGIC         <marker id="lbsArrowA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M 0 0 L 10 5 L 0 10 z" fill="#FFAB00"></path>
# MAGIC         </marker>
# MAGIC       </defs>
# MAGIC       <!-- ===== VECTOR EXAMPLE (top right) ===== -->
# MAGIC       <g data-part="vector hybrid">
# MAGIC         <rect x="372" y="14" width="620" height="170" rx="8" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="2"></rect>
# MAGIC         <text x="392" y="40" font-size="15" font-weight="600" fill="#2272B4">Vector: matches by MEANING</text>
# MAGIC         <rect x="392" y="52" width="196" height="116" rx="6" fill="#fff" stroke="#4299E0" stroke-width="1"></rect>
# MAGIC         <text x="400" y="68" font-size="14" fill="#90A5B1">embedding space</text>
# MAGIC         <circle cx="476" cy="118" r="42" fill="rgba(66,153,224,0.14)" stroke="#4299E0" stroke-width="1.5" stroke-dasharray="4 3"></circle>
# MAGIC         <circle cx="476" cy="118" r="7" fill="#2272B4"></circle>
# MAGIC         <text x="476" y="101" text-anchor="middle" font-size="14" fill="#2272B4">query</text>
# MAGIC         <circle cx="448" cy="140" r="5" fill="#4299E0"></circle>
# MAGIC         <circle cx="506" cy="104" r="5" fill="#4299E0"></circle>
# MAGIC         <circle cx="502" cy="142" r="5" fill="#4299E0"></circle>
# MAGIC         <circle cx="556" cy="70" r="5" fill="#C2C2C2"></circle>
# MAGIC         <circle cx="414" cy="78" r="5" fill="#C2C2C2"></circle>
# MAGIC         <circle cx="562" cy="152" r="5" fill="#C2C2C2"></circle>
# MAGIC         <text x="608" y="70" font-size="14" fill="#5A6F77">nearest rows returned:</text>
# MAGIC         <rect x="608" y="80" width="266" height="26" rx="5" fill="#fff" stroke="#4299E0" stroke-width="1"></rect>
# MAGIC         <text x="618" y="98" font-size="14" fill="#0b2026">&#8220;machine is dead, no lights&#8221;</text>
# MAGIC         <rect x="608" y="112" width="266" height="26" rx="5" fill="#fff" stroke="#4299E0" stroke-width="1"></rect>
# MAGIC         <text x="618" y="130" font-size="14" fill="#0b2026">&#8220;notebook stopped booting&#8221;</text>
# MAGIC         <text x="608" y="160" font-size="14" font-style="italic" fill="#2272B4">&#8593; zero shared words &#8212; still matches</text>
# MAGIC         <!-- dashed leader: Vector box -> vector panel -->
# MAGIC         <path d="M 396 220 L 396 202 L 682 202 L 682 184" fill="none" stroke="#4299E0" stroke-width="2" stroke-dasharray="6 5"></path>
# MAGIC         <circle cx="396" cy="220" r="4.5" fill="#4299E0"></circle>
# MAGIC         <circle cx="682" cy="184" r="4.5" fill="#4299E0"></circle>
# MAGIC       </g>
# MAGIC       <!-- ===== QUERY (shared by every path) ===== -->
# MAGIC       <g data-part="vector keyword hybrid">
# MAGIC         <rect x="8" y="302" width="160" height="76" rx="8" fill="#F9F7F4" stroke="#1B5162" stroke-width="2"></rect>
# MAGIC         <text x="88" y="333" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Query</text>
# MAGIC         <text x="88" y="356" text-anchor="middle" font-size="14" fill="#1B5162">text + embedding</text>
# MAGIC         <rect x="8" y="392" width="160" height="48" rx="6" fill="#fff" stroke="#1B5162" stroke-width="1.5" stroke-dasharray="6 4"></rect>
# MAGIC         <text x="88" y="413" text-anchor="middle" font-size="14" font-weight="600" font-family="ui-monospace, Menlo, Consolas, monospace" fill="#0b2026">&#8220;laptop won&#8217;t</text>
# MAGIC         <text x="88" y="431" text-anchor="middle" font-size="14" font-weight="600" font-family="ui-monospace, Menlo, Consolas, monospace" fill="#0b2026">power on&#8221;</text>
# MAGIC       </g>
# MAGIC       <!-- ===== VECTOR BRANCH ===== -->
# MAGIC       <g data-part="vector hybrid">
# MAGIC         <path d="M 168 340 C 216 340, 220 256, 268 256" fill="none" stroke="#4299E0" stroke-width="2" marker-end="url(#lbsArrowV)"></path>
# MAGIC         <rect x="272" y="220" width="248" height="72" rx="8" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="2" stroke-dasharray="8 5"></rect>
# MAGIC         <text x="292" y="248" font-size="16" font-weight="600" fill="#0b2026">Vector (semantic)</text>
# MAGIC         <text x="292" y="272" font-size="14" fill="#2272B4">nearest vectors by distance</text>
# MAGIC       </g>
# MAGIC       <!-- ===== KEYWORD BRANCH ===== -->
# MAGIC       <g data-part="keyword hybrid">
# MAGIC         <path d="M 168 340 C 216 340, 220 424, 268 424" fill="none" stroke="#FF5F46" stroke-width="2" marker-end="url(#lbsArrowK)"></path>
# MAGIC         <rect x="272" y="388" width="248" height="72" rx="8" fill="rgba(255,95,70,0.10)" stroke="#FF5F46" stroke-width="2" stroke-dasharray="8 5"></rect>
# MAGIC         <text x="292" y="416" font-size="16" font-weight="600" fill="#0b2026">Keyword (full-text)</text>
# MAGIC         <text x="292" y="440" font-size="14" fill="#FF5F46">BM25 term relevance</text>
# MAGIC       </g>
# MAGIC       <!-- ===== MERGE + RANKED LIST (hybrid only) ===== -->
# MAGIC       <g data-part="hybrid">
# MAGIC         <path d="M 520 256 C 570 256, 562 340, 612 340" fill="none" stroke="#7B61FF" stroke-width="2" marker-end="url(#lbsArrowG)"></path>
# MAGIC         <path d="M 520 424 C 570 424, 562 340, 612 340" fill="none" stroke="#7B61FF" stroke-width="2" marker-end="url(#lbsArrowG)"></path>
# MAGIC         <rect x="616" y="302" width="160" height="76" rx="8" fill="rgba(123,97,255,0.10)" stroke="#7B61FF" stroke-width="2" stroke-dasharray="8 5"></rect>
# MAGIC         <text x="696" y="333" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Hybrid</text>
# MAGIC         <text x="696" y="356" text-anchor="middle" font-size="14" fill="#7B61FF">combine with RRF</text>
# MAGIC         <path d="M 776 340 L 824 340" fill="none" stroke="#FFAB00" stroke-width="2" marker-end="url(#lbsArrowA)"></path>
# MAGIC         <rect x="828" y="288" width="164" height="104" rx="8" fill="rgba(255,171,0,0.12)" stroke="#FFAB00" stroke-width="2"></rect>
# MAGIC         <text x="910" y="314" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">One ranked list</text>
# MAGIC         <rect x="848" y="326" width="124" height="14" rx="4" fill="#FFAB00"></rect>
# MAGIC         <rect x="848" y="346" width="102" height="14" rx="4" fill="#FFCC66"></rect>
# MAGIC         <rect x="848" y="366" width="84" height="14" rx="4" fill="#FFCC66"></rect>
# MAGIC       </g>
# MAGIC       <!-- ===== KEYWORD EXAMPLE (bottom left) ===== -->
# MAGIC       <g data-part="keyword hybrid">
# MAGIC         <rect x="8" y="492" width="620" height="170" rx="8" fill="rgba(255,95,70,0.10)" stroke="#FF5F46" stroke-width="2"></rect>
# MAGIC         <text x="28" y="518" font-size="15" font-weight="600" fill="#FF5F46">Keyword: matches by EXACT TERMS</text>
# MAGIC         <text x="28" y="546" font-size="14" fill="#5A6F77">terms searched:</text>
# MAGIC         <rect x="28" y="554" width="72" height="26" rx="5" fill="#FF9E94" stroke="#FF5F46" stroke-width="1"></rect>
# MAGIC         <text x="64" y="572" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">laptop</text>
# MAGIC         <rect x="108" y="554" width="66" height="26" rx="5" fill="#FF9E94" stroke="#FF5F46" stroke-width="1"></rect>
# MAGIC         <text x="141" y="572" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">power</text>
# MAGIC         <rect x="28" y="598" width="146" height="26" rx="5" fill="#EEEDE9" stroke="#C2C2C2" stroke-width="1"></rect>
# MAGIC         <text x="38" y="616" font-size="14" fill="#5A6F77">&#8220;screen is blank&#8221;</text>
# MAGIC         <text x="244" y="546" font-size="14" fill="#5A6F77">rows containing those terms:</text>
# MAGIC         <rect x="244" y="554" width="266" height="26" rx="5" fill="#fff" stroke="#FF5F46" stroke-width="1"></rect>
# MAGIC         <text x="254" y="572" font-size="14" fill="#0b2026">&#8220;<tspan font-weight="700" fill="#FF5F46">laptop</tspan> has no <tspan font-weight="700" fill="#FF5F46">power</tspan>&#8221;</text>
# MAGIC         <rect x="244" y="586" width="266" height="26" rx="5" fill="#fff" stroke="#FF5F46" stroke-width="1"></rect>
# MAGIC         <text x="254" y="604" font-size="14" fill="#0b2026">&#8220;<tspan font-weight="700" fill="#FF5F46">power</tspan> supply for <tspan font-weight="700" fill="#FF5F46">laptop</tspan>&#8221;</text>
# MAGIC         <text x="244" y="634" font-size="14" font-style="italic" fill="#FF5F46">&#8593; no shared term &#8212; not returned</text>
# MAGIC         <!-- dashed leader: Keyword box -> keyword panel -->
# MAGIC         <path d="M 396 460 L 396 476 L 318 476 L 318 492" fill="none" stroke="#FF5F46" stroke-width="2" stroke-dasharray="6 5"></path>
# MAGIC         <circle cx="396" cy="460" r="4.5" fill="#FF5F46"></circle>
# MAGIC         <circle cx="318" cy="492" r="4.5" fill="#FF5F46"></circle>
# MAGIC       </g>
# MAGIC       <!-- ===== HYBRID OFFSHOOT (bottom right) ===== -->
# MAGIC       <g data-part="hybrid">
# MAGIC         <path d="M 696 378 L 696 404" fill="none" stroke="#7B61FF" stroke-width="2" stroke-dasharray="6 5"></path>
# MAGIC         <circle cx="696" cy="378" r="4.5" fill="#7B61FF"></circle>
# MAGIC         <circle cx="696" cy="404" r="4.5" fill="#7B61FF"></circle>
# MAGIC         <rect x="644" y="404" width="348" height="258" rx="8" fill="rgba(123,97,255,0.08)" stroke="#7B61FF" stroke-width="2"></rect>
# MAGIC         <text x="662" y="440" font-size="15" font-weight="600" fill="#7B61FF">Hybrid: MERGES both result sets</text>
# MAGIC         <rect x="662" y="464" width="150" height="34" rx="6" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="1.5"></rect>
# MAGIC         <text x="737" y="486" text-anchor="middle" font-size="14" fill="#2272B4">top hit &#8212; Vector</text>
# MAGIC         <rect x="824" y="464" width="150" height="34" rx="6" fill="rgba(255,95,70,0.10)" stroke="#FF5F46" stroke-width="1.5"></rect>
# MAGIC         <text x="899" y="486" text-anchor="middle" font-size="14" fill="#FF5F46">top hit &#8212; Keyword</text>
# MAGIC         <path d="M 737 498 L 737 516 L 818 516 L 818 532" fill="none" stroke="#7B61FF" stroke-width="2" stroke-dasharray="6 5"></path>
# MAGIC         <path d="M 899 498 L 899 516 L 818 516 L 818 532" fill="none" stroke="#7B61FF" stroke-width="2" stroke-dasharray="6 5"></path>
# MAGIC         <rect x="662" y="532" width="312" height="116" rx="8" fill="none" stroke="#7B61FF" stroke-width="2"></rect>
# MAGIC         <text x="818" y="561" text-anchor="middle" font-size="18" font-weight="600" fill="#7B61FF"><tspan font-size="26">&#8853;</tspan> rank fusion (RRF)</text>
# MAGIC         <rect x="676" y="574" width="284" height="28" rx="5" fill="#fff" stroke="#7B61FF" stroke-width="1"></rect>
# MAGIC         <text x="688" y="593" font-size="14" fill="#0b2026">&#8220;machine is dead, no lights&#8221;</text>
# MAGIC         <text x="922" y="593" font-size="14" fill="#2272B4">vec</text>
# MAGIC         <rect x="676" y="608" width="284" height="28" rx="5" fill="#fff" stroke="#7B61FF" stroke-width="1"></rect>
# MAGIC         <text x="688" y="627" font-size="14" fill="#0b2026">&#8220;laptop has no power&#8221;</text>
# MAGIC         <text x="916" y="627" font-size="14" fill="#FF5F46">term</text>
# MAGIC       </g>
# MAGIC     </svg>
# MAGIC   </div>
# MAGIC   <div class="grid">
# MAGIC     <div class="card c1" data-focus-btn="vector" aria-pressed="false">
# MAGIC       <div style="height:5px; border-radius:5px; margin:-4px 0 12px; background:#4299E0;"></div>
# MAGIC       <h3>Vector (semantic) search</h3>
# MAGIC       <p class="sub">Powered by <code>lakebase_vector</code></p>
# MAGIC       <ul>
# MAGIC         <li>Finds rows whose <strong>meaning</strong> is closest &mdash; even with no shared words.</li>
# MAGIC         <li>Query with an embedding; index returns nearest vectors by distance.</li>
# MAGIC         <li>Best for natural-language questions, recommendations, and RAG.</li>
# MAGIC       </ul>
# MAGIC     </div>
# MAGIC     <div class="card c2" data-focus-btn="keyword" aria-pressed="false">
# MAGIC       <div style="height:5px; border-radius:5px; margin:-4px 0 12px; background:#FF5F46;"></div>
# MAGIC       <h3>Keyword (full-text) search</h3>
# MAGIC       <p class="sub">Powered by <code>lakebase_text</code></p>
# MAGIC       <ul>
# MAGIC         <li>Ranks rows by how well they match your <strong>exact terms</strong>.</li>
# MAGIC         <li>Uses BM25 relevance scoring.</li>
# MAGIC         <li>Best for names, codes, and exact-term lookups where wording matters.</li>
# MAGIC       </ul>
# MAGIC     </div>
# MAGIC     <div class="card c3" data-focus-btn="hybrid" aria-pressed="false">
# MAGIC       <div style="height:5px; border-radius:5px; margin:-4px 0 12px; background:#7B61FF;"></div>
# MAGIC       <h3>Hybrid search</h3>
# MAGIC       <p class="sub">Runs both, merges the results</p>
# MAGIC       <ul>
# MAGIC         <li>Combines both result sets into <strong>one ranked list</strong>.</li>
# MAGIC         <li>Returns semantically similar <em>and</em> exact-term matches together.</li>
# MAGIC         <li>Best when queries mix intent with specific terms.</li>
# MAGIC       </ul>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <p class="hint">Click a card to spotlight that path in the diagram &mdash; click it again to show everything.</p>
# MAGIC   <script>
# MAGIC   (function () {
# MAGIC     var root = document.currentScript.closest('.lbs');
# MAGIC     if (!root) { return; }
# MAGIC     var btns = root.querySelectorAll('[data-focus-btn]');
# MAGIC     function apply(mode) {
# MAGIC       if (mode) { root.setAttribute('data-focus', mode); } else { root.removeAttribute('data-focus'); }
# MAGIC       btns.forEach(function (b) {
# MAGIC         var on = mode && b.getAttribute('data-focus-btn') === mode;
# MAGIC         b.setAttribute('aria-pressed', on ? 'true' : 'false');
# MAGIC         b.classList.toggle('is-on', !!on);
# MAGIC         b.classList.toggle('is-dim', !!mode && !on);
# MAGIC       });
# MAGIC     }
# MAGIC     btns.forEach(function (b) {
# MAGIC       b.addEventListener('click', function () {
# MAGIC         var mode = b.getAttribute('data-focus-btn');
# MAGIC         apply(root.getAttribute('data-focus') === mode ? null : mode);
# MAGIC       });
# MAGIC     });
# MAGIC   })();
# MAGIC   </script>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC