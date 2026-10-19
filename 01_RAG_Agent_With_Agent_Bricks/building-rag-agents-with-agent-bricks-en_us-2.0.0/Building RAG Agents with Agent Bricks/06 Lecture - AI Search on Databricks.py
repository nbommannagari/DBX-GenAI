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
# MAGIC # Lecture - AI Search on Databricks
# MAGIC #### Duration: 20 minutes
# MAGIC ## Overview
# MAGIC This lecture covers the theory behind AI Search before you build one in the next demo. We'll explore how embeddings and similarity search power retrieval, how AI Search is organized into endpoints and indexes and the search modes it supports, how agents use AI Search under the hood, and when to choose managed RAG vs. building your own. We'll also cover how Unity Catalog governs the entire pipeline.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lecture, you will be able to:
# MAGIC 1. Explain how embeddings transform text into searchable vectors
# MAGIC 2. Describe how cosine similarity drives retrieval ranking
# MAGIC 3. Describe the roles of AI Search endpoints and indexes
# MAGIC 4. Compare the index types for providing embeddings (managed, self-managed, direct access)
# MAGIC 5. Distinguish semantic, keyword, and hybrid search
# MAGIC 6. Compare managed RAG (Knowledge Assistants) and custom RAG approaches
# MAGIC 7. Explain how Unity Catalog governs AI Search indexes

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Embeddings and AI Search Fundamentals

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. What Are Embeddings?
# MAGIC
# MAGIC An **embedding** is a calculation (represented as a vector) that is based on the context that a word appears in. Texts with similar meanings produce vectors that are close together in mathematical space.
# MAGIC
# MAGIC For example, "cozy apartment near the park" and "comfortable flat close to green space" would have similar embeddings, even though they use completely different words.
# MAGIC
# MAGIC For example, when creating an AI Search index with `embedding_model_endpoint_name="databricks-gte-large-en"`, Databricks calls this model on every chunk in your table to produce these vectors. Now let's understand how those vectors are used to find relevant results.
# MAGIC
# MAGIC **Key considerations when working with embeddings:**
# MAGIC - **Use the same model** for indexing documents and processing queries, as this ensures they exist in the same vector space. With Genie Agents, this is managed for you, but Databricks has fully support for custom workloads.
# MAGIC - **Respect context windows** - embedding models have token limits (e.g., 512 tokens). Text beyond the limit is silently truncated
# MAGIC - **Dimension trade-offs** - higher dimensions capture more nuance but cost more to store and search. The number of dimensions represents the amount of context the model considers when calculating the embeddings. 
# MAGIC
# MAGIC <div class="emb" style="width:100%;margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#0b2026;">
# MAGIC   <style>
# MAGIC     .emb *{box-sizing:border-box;}
# MAGIC     .emb .mono{font-family:ui-monospace,Menlo,Consolas,monospace;}
# MAGIC     .emb .row{display:flex;gap:24px;flex-wrap:wrap;align-items:stretch;width:100%;}
# MAGIC     .emb .card{flex:1 1 420px;min-width:0;background:#fff;border:1.5px solid #C6DCEA;border-radius:14px;padding:16px 18px 18px;box-shadow:0 2px 8px rgba(27,49,57,.07);display:flex;flex-direction:column;overflow:hidden;}
# MAGIC     .emb .accent{height:5px;border-radius:5px;margin:-4px 0 12px;background:#FF5F46;}
# MAGIC     .emb .card.b .accent{background:#4299E0;}
# MAGIC     .emb .inner{padding:0 6px 0;display:flex;flex-direction:column;flex:1 1 auto;}
# MAGIC     .emb .eyebrow{font-size:14px;font-weight:800;letter-spacing:.4px;color:#FF5F46;padding-bottom:12px;margin-bottom:18px;border-bottom:1px solid #EEEDE9;}
# MAGIC     .emb .card.b .eyebrow{color:#2272B4;}
# MAGIC     .emb .body{flex:1 1 auto;display:flex;flex-direction:column;justify-content:center;}
# MAGIC     .emb .model{background:#FFF6F4;border:1px solid #FF5F46;border-radius:6px;text-align:center;padding:12px 10px;margin-bottom:16px;}
# MAGIC     .emb .model .t{font-size:16px;font-weight:800;color:#0b2026;}
# MAGIC     .emb .model .s{font-size:14px;color:#5A6F77;margin-top:4px;}
# MAGIC     .emb .mlabel{font-size:14px;font-weight:700;letter-spacing:.3px;color:#618794;margin-bottom:10px;}
# MAGIC     .emb .models{margin-bottom:20px;}
# MAGIC     .emb .mrow{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
# MAGIC     .emb .mrow:last-child{margin-bottom:0;}
# MAGIC     .emb .dot{flex:none;width:7px;height:7px;border-radius:50%;background:#FF5F46;}
# MAGIC     .emb .dot.ext{background:#618794;}
# MAGIC     .emb .mname{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px;color:#0b2026;}
# MAGIC     .emb .mext{font-size:14px;font-style:italic;color:#5A6F77;}
# MAGIC     .emb .lane{display:flex;align-items:center;gap:12px;margin-bottom:12px;}
# MAGIC     .emb .lane:last-child{margin-bottom:0;}
# MAGIC     .emb .word{flex:none;width:72px;text-align:center;font-size:14px;font-weight:700;background:transparent;border:1px solid #DCE0E2;border-radius:6px;padding:8px 0;color:#0b2026;}
# MAGIC     .emb .arw{flex:none;color:#618794;font-size:16px;font-weight:800;}
# MAGIC     .emb .vec{flex:1 1 0;min-width:0;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px;background:transparent;border:1px solid #DCE0E2;border-radius:6px;padding:9px 12px;color:#0b2026;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
# MAGIC     .emb .plot{width:100%;max-width:520px;margin:0 auto;}
# MAGIC     .emb svg{display:block;width:100%;height:auto;}
# MAGIC     .emb .cap{font-size:14px;color:#5A6F77;line-height:1.5;margin:18px 6px 0;padding-top:14px;border-top:1px solid #EEEDE9;}
# MAGIC     .emb .cap b{color:#0b2026;}
# MAGIC   </style>
# MAGIC   <div class="row">
# MAGIC     <div class="card a">
# MAGIC       <div class="accent"></div>
# MAGIC       <div class="inner">
# MAGIC         <div class="eyebrow">SAMPLE EMBEDDINGS</div>
# MAGIC         <div class="body" style="justify-content:flex-start;">
# MAGIC           <div class="model">
# MAGIC             <div class="t">Embedding model</div>
# MAGIC             <div class="s">a neural network trained on large amounts of text</div>
# MAGIC           </div>
# MAGIC           <div class="mlabel">Examples on Databricks</div>
# MAGIC           <div class="models">
# MAGIC             <div class="mrow"><span class="dot"></span><span class="mname">databricks-gte-large-en</span></div>
# MAGIC             <div class="mrow"><span class="dot"></span><span class="mname">databricks-bge-large-en</span></div>
# MAGIC             <div class="mrow"><span class="dot"></span><span class="mname">databricks-qwen3-embedding-0-6b</span></div>
# MAGIC             <div class="mrow"><span class="dot ext"></span><span class="mext">Bring your own external embedding model</span></div>
# MAGIC           </div>
# MAGIC           <div class="lane">
# MAGIC             <div class="word">cat</div><div class="arw">&rarr;</div>
# MAGIC             <div class="vec">[ 0.12, &minus;0.83, 0.05, 0.44, 0.671, &minus;0.29, 0.38, &minus;0.07, 0.52, 0.16, &minus;0.61, 0.24, 0.09, &minus;0.45, 0.73, 0.31, &hellip; ]</div>
# MAGIC           </div>
# MAGIC           <div class="lane">
# MAGIC             <div class="word">dog</div><div class="arw">&rarr;</div>
# MAGIC             <div class="vec">[ 0.15, &minus;0.79, 0.11, 0.40, 0.63, &minus;0.33, 0.35, &minus;0.04, 0.49, 0.21, &minus;0.58, 0.27, 0.13, &minus;0.41, 0.70, 0.28, &hellip; ]</div>
# MAGIC           </div>
# MAGIC           <div class="lane">
# MAGIC             <div class="word">car</div><div class="arw">&rarr;</div>
# MAGIC             <div class="vec">[ &minus;0.66, 0.28, 0.91, &minus;0.10, &minus;0.47, 0.82, &minus;0.19, 0.55, &minus;0.72, 0.34, 0.08, &minus;0.63, 0.41, 0.77, &minus;0.22, 0.60, &hellip; ]</div>
# MAGIC           </div>
# MAGIC         </div>
# MAGIC       </div>
# MAGIC       <div class="cap">Every item maps to a <b>fixed-length</b> list of numbers (often 1024), regardless of input size. Related inputs receive similar numbers: <span class="mono">cat</span> and <span class="mono">dog</span> are close; <span class="mono">car</span> is not.</div>
# MAGIC     </div>
# MAGIC     <div class="card b">
# MAGIC       <div class="accent"></div>
# MAGIC       <div class="inner">
# MAGIC         <div class="eyebrow">SIMILAR MEANING LANDS NEARBY</div>
# MAGIC         <div class="body">
# MAGIC           <div class="plot">
# MAGIC             <svg viewBox="0 0 400 290" role="img" aria-label="Scatter plot of word embeddings grouped into three clusters: animals, royalty, and vehicles.">
# MAGIC               <defs>
# MAGIC                 <marker id="emb-ax" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#618794"/></marker>
# MAGIC               </defs>
# MAGIC               <line x1="16" y1="275" x2="392" y2="275" stroke="#618794" stroke-width="1.5" marker-end="url(#emb-ax)"/>
# MAGIC               <line x1="16" y1="275" x2="16" y2="14" stroke="#618794" stroke-width="1.5" marker-end="url(#emb-ax)"/>
# MAGIC               <ellipse cx="118" cy="116" rx="62" ry="52" fill="none" stroke="#FF9E94" stroke-width="1.3" stroke-dasharray="4 4"/>
# MAGIC               <ellipse cx="306" cy="100" rx="56" ry="42" fill="none" stroke="#C4CCD6" stroke-width="1.3" stroke-dasharray="4 4"/>
# MAGIC               <ellipse cx="306" cy="212" rx="58" ry="42" fill="none" stroke="#C4CCD6" stroke-width="1.3" stroke-dasharray="4 4"/>
# MAGIC               <text x="118" y="50" text-anchor="middle" font-size="14" font-weight="800" letter-spacing="0.6" fill="#FF5F46">ANIMALS</text>
# MAGIC               <text x="306" y="44" text-anchor="middle" font-size="14" font-weight="800" letter-spacing="0.6" fill="#1B5162">ROYALTY</text>
# MAGIC               <text x="306" y="160" text-anchor="middle" font-size="14" font-weight="800" letter-spacing="0.6" fill="#2272B4">VEHICLES</text>
# MAGIC               <circle cx="102" cy="118" r="5" fill="#FF5F46"/><text x="92" y="123" text-anchor="end" font-size="14" font-weight="600" fill="#0b2026">cat</text>
# MAGIC               <circle cx="142" cy="92" r="5" fill="#FF5F46"/><text x="152" y="97" font-size="14" font-weight="600" fill="#0b2026">dog</text>
# MAGIC               <circle cx="108" cy="146" r="5" fill="#FF5F46"/><text x="118" y="151" font-size="14" font-weight="600" fill="#0b2026">puppy</text>
# MAGIC               <circle cx="292" cy="94" r="5" fill="#1B5162"/><text x="282" y="99" text-anchor="end" font-size="14" font-weight="600" fill="#0b2026">king</text>
# MAGIC               <circle cx="318" cy="116" r="5" fill="#1B5162"/><text x="328" y="121" font-size="14" font-weight="600" fill="#0b2026">queen</text>
# MAGIC               <circle cx="292" cy="206" r="5" fill="#2272B4"/><text x="282" y="211" text-anchor="end" font-size="14" font-weight="600" fill="#0b2026">car</text>
# MAGIC               <circle cx="322" cy="226" r="5" fill="#2272B4"/><text x="332" y="231" font-size="14" font-weight="600" fill="#0b2026">truck</text>
# MAGIC             </svg>
# MAGIC           </div>
# MAGIC         </div>
# MAGIC       </div>
# MAGIC       <div class="cap">Because meaning is stored as <b>position</b>, related words fall into the same region. Real embeddings span <b>hundreds or thousands of dimensions</b> &mdash; shown here in 2D for intuition.</div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A2. The Similarity Search Pipeline
# MAGIC
# MAGIC When you search a vector index, the system:
# MAGIC 1. Converts your query text into an embedding vector
# MAGIC 2. Finds the closest vectors in the index (using cosine similarity or other distance metrics)
# MAGIC 3. Returns the corresponding text chunks, ranked by relevance
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 680 200" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Similarity Search Flow</title>
# MAGIC   <defs>
# MAGIC     <marker id="ss-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC
# MAGIC   <!-- Query -->
# MAGIC   <rect x="10" y="60" width="130" height="60" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="75" y="84" text-anchor="middle" font-size="17" font-weight="600" fill="#0b2026">Query Text</text>
# MAGIC   <text x="75" y="102" text-anchor="middle" font-size="15" fill="#618794">"pet-friendly?"</text>
# MAGIC
# MAGIC   <path d="M140 90 L180 90" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#ss-arrow)"/>
# MAGIC
# MAGIC   <!-- Embed -->
# MAGIC   <rect x="185" y="60" width="130" height="60" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="250" y="84" text-anchor="middle" font-size="17" font-weight="600" fill="#0b2026">Embed</text>
# MAGIC   <text x="250" y="102" text-anchor="middle" font-size="15" fill="#618794">[0.23, -0.41, ...]</text>
# MAGIC
# MAGIC   <path d="M315 90 L355 90" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#ss-arrow)"/>
# MAGIC
# MAGIC   <!-- Search -->
# MAGIC   <rect x="360" y="60" width="130" height="60" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="425" y="84" text-anchor="middle" font-size="17" font-weight="600" fill="#0b2026">Find Nearest</text>
# MAGIC   <text x="425" y="102" text-anchor="middle" font-size="15" fill="#618794">Cosine similarity</text>
# MAGIC
# MAGIC   <path d="M490 90 L530 90" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#ss-arrow)"/>
# MAGIC
# MAGIC   <!-- Results -->
# MAGIC   <rect x="535" y="40" width="130" height="100" rx="8" fill="#F9F7F4" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="600" y="64" text-anchor="middle" font-size="17" font-weight="600" fill="#0b2026">Top Results</text>
# MAGIC   <rect x="547" y="76" width="106" height="20" rx="4" fill="#2272B4" fill-opacity="0.30"/>
# MAGIC   <text x="600" y="90" text-anchor="middle" font-size="14" fill="#0b2026">Chunk 1 (0.94)</text>
# MAGIC   <rect x="547" y="100" width="106" height="20" rx="4" fill="#2272B4" fill-opacity="0.18"/>
# MAGIC   <text x="600" y="114" text-anchor="middle" font-size="14" fill="#0b2026">Chunk 2 (0.87)</text>
# MAGIC   <rect x="547" y="124" width="106" height="10" rx="4" fill="#2272B4" fill-opacity="0.10"/>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC For fast retrieval at scale, Databricks uses **Approximate Nearest Neighbors (ANN)** algorithms, trading a small amount of precision for dramatic speed gains.
# MAGIC
# MAGIC When you use `similarity_search` on an AI Search index, your query text is embedded, compared against all vectors in the index, and the top matches are returned with their similarity scores. You'll see this in action in the next demo.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Learn More</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC See the full AI Search documentation (<a href="https://docs.databricks.com/aws/en/ai-search/ai-search" style="color: #1976d2;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/ai-search/ai-search" style="color: #1976d2;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/ai-search/ai-search" style="color: #1976d2;">GCP</a>) for details on endpoints, indexes, and APIs.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A3. What Is AI Search?
# MAGIC
# MAGIC **AI Search** is a retrieval solution built into the Databricks Data Intelligence Platform and integrated with Unity Catalog governance. Given a query, its core job is to find the stored embeddings that are **most semantically similar** (using vector similarity). It is the retrieval engine behind RAG systems, recommender systems, and image or video search.
# MAGIC
# MAGIC To stay fast at scale, AI Search uses the **Hierarchical Navigable Small World (HNSW)** algorithm for approximate nearest neighbor (ANN) search. This means lookups remain quick even across hundreds of millions of vectors.
# MAGIC
# MAGIC Every deployment is built from two components: **endpoints** and **indexes**. We look at those next, then at how embeddings get into an index and the ways you can search it.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A4. Providing Embeddings: Index Types
# MAGIC
# MAGIC How do vectors get into an index? There are three approaches. The first two keep the index automatically in sync with a Delta source; the third gives you manual control. In each diagram, the left side shows how the index is **created** and the right side shows the **query endpoint** at serving time — the numbered steps trace a single query.
# MAGIC
# MAGIC **Databricks-managed embeddings (Delta Sync)** — you point at a text column and choose an embedding model; Databricks computes the embeddings, optionally saves them to a table, and keeps the index synced as the source changes. The simplest path, and the one used in this course. At query time, (1) the query is embedded with the same model, (2) compared against the vector index, and (3) the nearest results are returned to the app.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 860 300" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Databricks-managed embeddings: source Delta table to embedding model to vector index, queried at serving time</title>
# MAGIC   <defs>
# MAGIC     <marker id="a5m-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC       <path d="M1 1L9 5L1 9Z" fill="#1B3139"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <text x="20" y="26" font-size="14" font-weight="700" fill="#618794">Create vector index</text>
# MAGIC   <text x="20" y="44" font-size="14" font-weight="700" fill="#618794">(Databricks calculates and</text>
# MAGIC   <text x="20" y="62" font-size="14" font-weight="700" fill="#618794">optionally saves embeddings)</text>
# MAGIC   <text x="560" y="26" font-size="14" font-weight="700" fill="#618794">Query endpoint</text>
# MAGIC   <line x1="490" y1="18" x2="490" y2="292" stroke="#DCE0E2" stroke-width="1.5"/>
# MAGIC   <!-- Source Delta table -->
# MAGIC   <rect x="15" y="125" width="115" height="56" rx="2" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.2"/>
# MAGIC   <text x="72" y="149" text-anchor="middle" font-size="13" fill="#0b2026">Source Delta</text>
# MAGIC   <text x="72" y="166" text-anchor="middle" font-size="13" fill="#0b2026">table</text>
# MAGIC   <path d="M130 153 L210 153" fill="none" stroke="#1B3139" stroke-width="1.6" marker-end="url(#a5m-arrow)"/>
# MAGIC   <!-- Embedding model -->
# MAGIC   <rect x="215" y="125" width="140" height="56" rx="2" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="285" y="158" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Embedding model</text>
# MAGIC   <!-- Embeddings table (optional, dashed) -->
# MAGIC   <path d="M258 181 L122 246" fill="none" stroke="#1B3139" stroke-width="1.6" marker-end="url(#a5m-arrow)"/>
# MAGIC   <rect x="35" y="250" width="130" height="46" rx="2" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.2" stroke-dasharray="5 4"/>
# MAGIC   <text x="100" y="272" text-anchor="middle" font-size="13" fill="#0b2026">Embeddings</text>
# MAGIC   <text x="100" y="289" text-anchor="middle" font-size="13" fill="#0b2026">table</text>
# MAGIC   <!-- Vector index cylinder (drawn before the incoming arrows so the arrowheads sit on top) -->
# MAGIC   <path d="M236 234 L236 286 A64 10 0 0 1 364 286 L364 234 Z" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <ellipse cx="300" cy="234" rx="64" ry="10" fill="#2272B4" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="300" y="268" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Vector index</text>
# MAGIC   <!-- Embedding model to vector index: solid + dashed(step 2) -->
# MAGIC   <path d="M278 181 L278 230" fill="none" stroke="#1B3139" stroke-width="1.6" marker-end="url(#a5m-arrow)"/>
# MAGIC   <path d="M322 181 L322 230" fill="none" stroke="#1B3139" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#a5m-arrow)"/>
# MAGIC   <!-- Query -->
# MAGIC   <ellipse cx="600" cy="153" rx="72" ry="35" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.8"/>
# MAGIC   <text x="600" y="158" text-anchor="middle" font-size="15" font-weight="700" fill="#0b2026">Query</text>
# MAGIC   <path d="M528 153 L360 153" fill="none" stroke="#1B3139" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#a5m-arrow)"/>
# MAGIC   <!-- LLM application -->
# MAGIC   <rect x="730" y="125" width="110" height="56" rx="2" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="785" y="149" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">LLM</text>
# MAGIC   <text x="785" y="166" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">application</text>
# MAGIC   <path d="M672 153 L726 153" fill="none" stroke="#1B3139" stroke-width="1.6" marker-end="url(#a5m-arrow)"/>
# MAGIC   <!-- step 3: vector index back to query -->
# MAGIC   <path d="M366 266 C 480 300, 556 246, 598 190" fill="none" stroke="#1B3139" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#a5m-arrow)"/>
# MAGIC   <!-- badges -->
# MAGIC   <circle cx="505" cy="139" r="12" fill="#1B3139" stroke="#fff" stroke-width="2"/>
# MAGIC   <text x="505" y="144" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">1</text>
# MAGIC   <circle cx="345" cy="205" r="12" fill="#1B3139" stroke="#fff" stroke-width="2"/>
# MAGIC   <text x="345" y="210" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">2</text>
# MAGIC   <circle cx="522" cy="255" r="12" fill="#1B3139" stroke="#fff" stroke-width="2"/>
# MAGIC   <text x="522" y="260" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">3</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC **Self-managed embeddings (Delta Sync)** — you precompute embeddings and store them in the source table; the index still auto-syncs on changes, but there is no embedding model on the create side. You **cannot later convert** a self-managed index to managed — you would recreate it.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 860 240" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Self-managed precalculated embeddings: source Delta table syncs to a vector index that is queried at serving time</title>
# MAGIC   <defs>
# MAGIC     <marker id="a5s-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC       <path d="M1 1L9 5L1 9Z" fill="#1B3139"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <text x="20" y="26" font-size="14" font-weight="700" fill="#618794">Create vector index</text>
# MAGIC   <text x="20" y="44" font-size="14" font-weight="700" fill="#618794">(precalculated embeddings)</text>
# MAGIC   <text x="560" y="26" font-size="14" font-weight="700" fill="#618794">Query endpoint</text>
# MAGIC   <line x1="490" y1="18" x2="490" y2="232" stroke="#DCE0E2" stroke-width="1.5"/>
# MAGIC   <!-- Source Delta table -->
# MAGIC   <rect x="15" y="100" width="115" height="56" rx="2" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.2"/>
# MAGIC   <text x="72" y="124" text-anchor="middle" font-size="13" fill="#0b2026">Source Delta</text>
# MAGIC   <text x="72" y="141" text-anchor="middle" font-size="13" fill="#0b2026">table</text>
# MAGIC   <path d="M130 128 L212 128" fill="none" stroke="#1B3139" stroke-width="1.6" marker-end="url(#a5s-arrow)"/>
# MAGIC   <!-- Vector index cylinder cx=290 -->
# MAGIC   <path d="M226 108 L226 158 A64 10 0 0 1 354 158 L354 108 Z" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <ellipse cx="290" cy="108" rx="64" ry="10" fill="#2272B4" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="290" y="140" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Vector index</text>
# MAGIC   <!-- Query -->
# MAGIC   <ellipse cx="585" cy="128" rx="72" ry="35" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.8"/>
# MAGIC   <text x="585" y="133" text-anchor="middle" font-size="15" font-weight="700" fill="#0b2026">Query</text>
# MAGIC   <path d="M513 128 L358 128" fill="none" stroke="#1B3139" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#a5s-arrow)"/>
# MAGIC   <!-- LLM application -->
# MAGIC   <rect x="715" y="100" width="110" height="56" rx="2" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="770" y="124" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">LLM</text>
# MAGIC   <text x="770" y="141" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">application</text>
# MAGIC   <path d="M657 128 L711 128" fill="none" stroke="#1B3139" stroke-width="1.6" marker-end="url(#a5s-arrow)"/>
# MAGIC   <!-- step 2: vector index back to query -->
# MAGIC   <path d="M300 168 C 420 224, 520 206, 583 165" fill="none" stroke="#1B3139" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#a5s-arrow)"/>
# MAGIC   <!-- badges -->
# MAGIC   <circle cx="492" cy="114" r="12" fill="#1B3139" stroke="#fff" stroke-width="2"/>
# MAGIC   <text x="492" y="119" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">1</text>
# MAGIC   <circle cx="470" cy="205" r="12" fill="#1B3139" stroke="#fff" stroke-width="2"/>
# MAGIC   <text x="470" y="210" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">2</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC **Direct Access** — you write vectors directly through the REST API or SDK, with no source table and no automatic sync. Best when an external pipeline computes embeddings or you need real-time upserts.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 760 230" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Direct Access: a vector index with no source table and no automatic sync, queried at serving time</title>
# MAGIC   <defs>
# MAGIC     <marker id="a5d-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC       <path d="M1 1L9 5L1 9Z" fill="#1B3139"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <text x="20" y="26" font-size="14" font-weight="700" fill="#618794">Create vector index</text>
# MAGIC   <text x="20" y="44" font-size="14" font-weight="700" fill="#618794">(precalculated embeddings,</text>
# MAGIC   <text x="20" y="62" font-size="14" font-weight="700" fill="#618794">no automatic sync)</text>
# MAGIC   <text x="470" y="26" font-size="14" font-weight="700" fill="#618794">Query endpoint</text>
# MAGIC   <line x1="380" y1="18" x2="380" y2="222" stroke="#DCE0E2" stroke-width="1.5"/>
# MAGIC   <!-- step 2: vector index back to query (arcs over the top of the cylinder) -->
# MAGIC   <path d="M170 94 C 250 50, 390 50, 440 106" fill="none" stroke="#1B3139" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#a5d-arrow)"/>
# MAGIC   <!-- Vector index cylinder cx=170 -->
# MAGIC   <path d="M106 104 L106 154 A64 10 0 0 1 234 154 L234 104 Z" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <ellipse cx="170" cy="104" rx="64" ry="10" fill="#2272B4" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="170" y="136" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Vector index</text>
# MAGIC   <!-- Query -->
# MAGIC   <ellipse cx="500" cy="126" rx="72" ry="35" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.8"/>
# MAGIC   <text x="500" y="131" text-anchor="middle" font-size="15" font-weight="700" fill="#0b2026">Query</text>
# MAGIC   <path d="M428 126 L238 126" fill="none" stroke="#1B3139" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#a5d-arrow)"/>
# MAGIC   <!-- LLM application -->
# MAGIC   <rect x="630" y="98" width="110" height="56" rx="2" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="685" y="122" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">LLM</text>
# MAGIC   <text x="685" y="139" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">application</text>
# MAGIC   <path d="M572 126 L626 126" fill="none" stroke="#1B3139" stroke-width="1.6" marker-end="url(#a5d-arrow)"/>
# MAGIC   <!-- badges -->
# MAGIC   <circle cx="408" cy="112" r="12" fill="#1B3139" stroke="#fff" stroke-width="2"/>
# MAGIC   <text x="408" y="117" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">1</text>
# MAGIC   <circle cx="319" cy="58" r="12" fill="#1B3139" stroke="#fff" stroke-width="2"/>
# MAGIC   <text x="319" y="63" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">2</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Full-text search index (Beta)</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Available on <strong>storage-optimized</strong> endpoints only, a full-text search index needs <strong>no embedding column</strong>: it searches text directly with BM25 keyword matching and uses triggered sync. Useful for pure keyword retrieval when vectors are not needed.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A5. CDF for RAG Workloads
# MAGIC A RAG agent is only as good as the index behind it. Source documents changes constantly. For example, policies are revised, tickets arrive, specs evolved, etc. When it comes to RAG workloads, there is always one crucial question that needs answering from an engineering perspective: **which chunks actually changed since the last pipeline run?** Databricks' Change Data Feed (CDF) answers this question by addressing cost and sync modes. 
# MAGIC - **Embeddings can be expensive**: Embeddings require compute or API calls. Without change tracking, you either re-embed the whole corpus or accept a stale nightly batch. CDF turns the work into a function of change volume, not corpus size. 
# MAGIC - **Two sync modes**: A Delta Sync index reads CDF either on demand or as a continuous stream. 
# MAGIC
# MAGIC <div class="cdfai">
# MAGIC   <style>
# MAGIC     .cdfai { max-width: 100%; margin: 0 auto; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; color: #0b2026; }
# MAGIC     .cdfai h3 { font-size: 21px; margin: 0 0 10px 0; color: #0b2026; }
# MAGIC     .cdfai h4 { font-size: 17px; margin: 0 0 8px 0; color: #0b2026; }
# MAGIC     .cdfai p { font-size: 15px; line-height: 1.55; margin: 0 0 14px 0; }
# MAGIC     .cdfai .card { background: #fff; border: 1.5px solid #C6DCEA; border-radius: 14px; padding: 16px 18px 18px; box-shadow: 0 2px 8px rgba(27,49,57,.07); margin-bottom: 22px; }
# MAGIC     .cdfai .bar { height: 5px; border-radius: 5px; margin: -4px 0 14px; }
# MAGIC     .cdfai .bar-blue { background: #4299E0; }
# MAGIC     .cdfai .bar-red { background: #FF3621; }
# MAGIC     .cdfai .bar-coral { background: #FF5F46; }
# MAGIC     .cdfai .cap { font-size: 14px; color: #618794; margin: 8px 0 0 0; line-height: 1.5; }
# MAGIC     .cdfai ul { margin: 6px 0 22px 0; padding-left: 22px; }
# MAGIC     .cdfai li { font-size: 15px; line-height: 1.55; margin-bottom: 12px; }
# MAGIC     .cdfai code { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 14px; background: rgba(66,153,224,0.10); padding: 1px 5px; border-radius: 4px; }
# MAGIC   </style>
# MAGIC   <div class="card">
# MAGIC     <div class="bar bar-blue"></div>
# MAGIC     <h4>The incremental RAG pipeline</h4>
# MAGIC     <p>CDF emits a row-level change stream (<code>insert</code>, <code>update</code>, <code>delete</code>) from the source Delta table. Every downstream stage &mdash; chunking, embedding, index sync &mdash; consumes only those rows. The numbered steps trace one changed document all the way to the agent.</p>
# MAGIC     <svg width="100%" viewBox="0 0 1060 320" role="img" style="font-family: sans-serif;">
# MAGIC       <title>Change data feed flows from source Delta table through chunking and embedding into a Delta Sync vector index serving a RAG agent</title>
# MAGIC       <defs>
# MAGIC         <marker id="cdfai-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M1 1L9 5L1 9Z" fill="#0b2026"/>
# MAGIC         </marker>
# MAGIC         <marker id="cdfai-arrc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M1 1L9 5L1 9Z" fill="#FF5F46"/>
# MAGIC         </marker>
# MAGIC       </defs>
# MAGIC       <text x="148" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="#618794">Source of truth</text>
# MAGIC       <text x="578" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="#618794">Incremental processing (only changed rows)</text>
# MAGIC       <text x="960" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="#618794">Serving</text>
# MAGIC       <line x1="296" y1="34" x2="296" y2="300" stroke="#C6DCEA" stroke-width="2"/>
# MAGIC       <line x1="860" y1="34" x2="860" y2="300" stroke="#C6DCEA" stroke-width="2"/>
# MAGIC       <!-- source documents -->
# MAGIC       <rect x="18" y="60" width="150" height="54" rx="6" fill="#fff" stroke="#0b2026" stroke-width="1.3"/>
# MAGIC       <text x="93" y="83" text-anchor="middle" font-size="14" fill="#0b2026">Documents, tickets,</text>
# MAGIC       <text x="93" y="101" text-anchor="middle" font-size="14" fill="#0b2026">policies, specs</text>
# MAGIC       <path d="M93 116 L93 150" fill="none" stroke="#0b2026" stroke-width="1.6" marker-end="url(#cdfai-arr)"/>
# MAGIC       <!-- delta table -->
# MAGIC       <path d="M22 166 L22 232 A71 11 0 0 0 164 232 L164 166 Z" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="1.6"/>
# MAGIC       <ellipse cx="93" cy="166" rx="71" ry="11" fill="#4299E0" stroke="#4299E0" stroke-width="1.6"/>
# MAGIC       <text x="93" y="198" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Source Delta</text>
# MAGIC       <text x="93" y="216" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">table</text>
# MAGIC       <text x="93" y="262" text-anchor="middle" font-size="14" fill="#618794">delta.enableChangeData</text>
# MAGIC       <text x="93" y="280" text-anchor="middle" font-size="14" fill="#618794">Feed = true</text>
# MAGIC       <!-- CDF stream band -->
# MAGIC       <rect x="186" y="150" width="86" height="98" rx="6" fill="rgba(255,95,70,0.12)" stroke="#FF5F46" stroke-width="1.8"/>
# MAGIC       <text x="229" y="176" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">CDF</text>
# MAGIC       <text x="229" y="196" text-anchor="middle" font-size="14" fill="#0b2026">insert</text>
# MAGIC       <text x="229" y="214" text-anchor="middle" font-size="14" fill="#0b2026">update</text>
# MAGIC       <text x="229" y="232" text-anchor="middle" font-size="14" fill="#0b2026">delete</text>
# MAGIC       <path d="M164 199 L182 199" fill="none" stroke="#FF5F46" stroke-width="2" marker-end="url(#cdfai-arrc)"/>
# MAGIC       <path d="M272 199 L316 199" fill="none" stroke="#FF5F46" stroke-width="2" marker-end="url(#cdfai-arrc)"/>
# MAGIC       <!-- chunk -->
# MAGIC       <rect x="320" y="170" width="140" height="58" rx="6" fill="#fff" stroke="#0b2026" stroke-width="1.3"/>
# MAGIC       <text x="390" y="195" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Parse + chunk</text>
# MAGIC       <text x="390" y="215" text-anchor="middle" font-size="14" fill="#618794">changed docs only</text>
# MAGIC       <path d="M460 199 L502 199" fill="none" stroke="#0b2026" stroke-width="1.8" marker-end="url(#cdfai-arr)"/>
# MAGIC       <!-- embed -->
# MAGIC       <rect x="506" y="170" width="150" height="58" rx="6" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="1.8"/>
# MAGIC       <text x="581" y="195" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Embedding model</text>
# MAGIC       <text x="581" y="215" text-anchor="middle" font-size="14" fill="#618794">GPU / API $$$</text>
# MAGIC       <path d="M656 199 L700 199" fill="none" stroke="#0b2026" stroke-width="1.8" marker-end="url(#cdfai-arr)"/>
# MAGIC       <!-- vector index -->
# MAGIC       <path d="M704 172 L704 232 A62 10 0 0 0 828 232 L828 172 Z" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="1.6"/>
# MAGIC       <ellipse cx="766" cy="172" rx="62" ry="10" fill="#4299E0" stroke="#4299E0" stroke-width="1.6"/>
# MAGIC       <text x="766" y="202" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Delta Sync</text>
# MAGIC       <text x="766" y="220" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">vector index</text>
# MAGIC       <text x="766" y="266" text-anchor="middle" font-size="14" fill="#618794">checkpointed &mdash; no</text>
# MAGIC       <text x="766" y="284" text-anchor="middle" font-size="14" fill="#618794">reload of existing rows</text>
# MAGIC       <!-- agent -->
# MAGIC       <rect x="892" y="100" width="146" height="58" rx="6" fill="rgba(255,95,70,0.12)" stroke="#FF5F46" stroke-width="1.8"/>
# MAGIC       <text x="965" y="125" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Retrieval query</text>
# MAGIC       <text x="965" y="145" text-anchor="middle" font-size="14" fill="#618794">from the agent</text>
# MAGIC       <rect x="892" y="212" width="146" height="58" rx="6" fill="#fff" stroke="#0b2026" stroke-width="1.3"/>
# MAGIC       <text x="965" y="237" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">LLM / agent</text>
# MAGIC       <text x="965" y="257" text-anchor="middle" font-size="14" fill="#618794">grounded answer</text>
# MAGIC       <path d="M888 129 L832 170" fill="none" stroke="#0b2026" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#cdfai-arr)"/>
# MAGIC       <path d="M830 236 L888 241" fill="none" stroke="#0b2026" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#cdfai-arr)"/>
# MAGIC       <!-- badges -->
# MAGIC       <circle cx="229" cy="132" r="13" fill="#FF3621"/>
# MAGIC       <text x="229" y="137" text-anchor="middle" font-size="14" font-weight="700" fill="#fff">1</text>
# MAGIC       <circle cx="390" cy="152" r="13" fill="#FF3621"/>
# MAGIC       <text x="390" y="157" text-anchor="middle" font-size="14" font-weight="700" fill="#fff">2</text>
# MAGIC       <circle cx="581" cy="152" r="13" fill="#FF3621"/>
# MAGIC       <text x="581" y="157" text-anchor="middle" font-size="14" font-weight="700" fill="#fff">3</text>
# MAGIC       <circle cx="766" cy="140" r="13" fill="#FF3621"/>
# MAGIC       <text x="766" y="145" text-anchor="middle" font-size="14" font-weight="700" fill="#fff">4</text>
# MAGIC       <circle cx="965" cy="186" r="13" fill="#FF3621"/>
# MAGIC       <text x="965" y="191" text-anchor="middle" font-size="14" font-weight="700" fill="#fff">5</text>
# MAGIC     </svg>
# MAGIC
# MAGIC   </div>
# MAGIC   <div class="card">
# MAGIC     <div class="bar bar-red"></div>
# MAGIC     <h4>Deletes and updates prevent hallucination</h4>
# MAGIC     <p>Insert-only pipelines are the hidden failure mode. A retracted policy that lingers in the index will be retrieved and cited as fact.</p>
# MAGIC     <svg width="100%" viewBox="0 0 1060 210" role="img" style="font-family: sans-serif;">
# MAGIC       <title>Without CDF a deleted document remains in the index and is cited; with CDF the delete event removes the chunk</title>
# MAGIC       <defs>
# MAGIC         <marker id="cdfai-arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M1 1L9 5L1 9Z" fill="#0b2026"/>
# MAGIC         </marker>
# MAGIC       </defs>
# MAGIC       <text x="18" y="22" font-size="14" font-weight="700" fill="#FF3621">Insert-only sync</text>
# MAGIC       <rect x="18" y="34" width="250" height="50" rx="6" fill="#fff" stroke="#0b2026" stroke-width="1.3"/>
# MAGIC       <text x="143" y="56" text-anchor="middle" font-size="14" fill="#0b2026">Doc retracted</text>
# MAGIC       <text x="143" y="75" text-anchor="middle" font-size="14" fill="#618794">in source</text>
# MAGIC       <path d="M268 59 L322 59" fill="none" stroke="#0b2026" stroke-width="1.6" marker-end="url(#cdfai-arr2)"/>
# MAGIC       <rect x="326" y="34" width="280" height="50" rx="6" fill="rgba(255,54,33,0.14)" stroke="#FF3621" stroke-width="1.8"/>
# MAGIC       <text x="466" y="56" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Stale chunk stays</text>
# MAGIC       <text x="466" y="75" text-anchor="middle" font-size="14" fill="#0b2026">in the index</text>
# MAGIC       <path d="M606 59 L660 59" fill="none" stroke="#0b2026" stroke-width="1.6" marker-end="url(#cdfai-arr2)"/>
# MAGIC       <rect x="664" y="34" width="250" height="50" rx="6" fill="#fff" stroke="#FF3621" stroke-width="1.6"/>
# MAGIC       <text x="789" y="56" text-anchor="middle" font-size="14" font-weight="700" fill="#FF3621">Agent cites</text>
# MAGIC       <text x="789" y="75" text-anchor="middle" font-size="14" font-weight="700" fill="#FF3621">dead content</text>
# MAGIC       <line x1="18" y1="106" x2="1042" y2="106" stroke="#C6DCEA" stroke-width="2"/>
# MAGIC       <text x="18" y="134" font-size="14" font-weight="700" fill="#4299E0">CDF-driven sync</text>
# MAGIC       <rect x="18" y="146" width="250" height="50" rx="6" fill="#fff" stroke="#0b2026" stroke-width="1.3"/>
# MAGIC       <text x="143" y="168" text-anchor="middle" font-size="14" fill="#0b2026">Doc retracted</text>
# MAGIC       <text x="143" y="187" text-anchor="middle" font-size="14" fill="#618794">in source</text>
# MAGIC       <path d="M268 171 L322 171" fill="none" stroke="#0b2026" stroke-width="1.6" marker-end="url(#cdfai-arr2)"/>
# MAGIC       <rect x="326" y="146" width="280" height="50" rx="6" fill="rgba(255,95,70,0.12)" stroke="#FF5F46" stroke-width="1.8"/>
# MAGIC       <text x="466" y="168" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">delete /</text>
# MAGIC       <text x="466" y="187" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">update_postimage</text>
# MAGIC       <path d="M606 171 L660 171" fill="none" stroke="#0b2026" stroke-width="1.6" marker-end="url(#cdfai-arr2)"/>
# MAGIC       <rect x="664" y="146" width="250" height="50" rx="6" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="1.8"/>
# MAGIC       <text x="789" y="168" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Chunk removed</text>
# MAGIC       <text x="789" y="187" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">or replaced</text>
# MAGIC     </svg>
# MAGIC
# MAGIC   </div>
# MAGIC   <div class="card">
# MAGIC     <div class="bar bar-coral"></div>
# MAGIC     <h4>Lakebase CDF: operational data straight into the agent</h4>
# MAGIC     <p>Operational Postgres state &mdash; customer records, order history, live catalogs &mdash; streams into Delta with roughly 15-second latency, then feeds the same Delta Sync index. No bespoke CDC pipeline to build or babysit.</p>
# MAGIC     <svg width="100%" viewBox="0 0 1060 150" role="img" style="font-family: sans-serif;">
# MAGIC       <title>Lakebase Postgres streams change data into Delta with about fifteen second latency, feeding a vector index and RAG agent</title>
# MAGIC       <defs>
# MAGIC         <marker id="cdfai-arr4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M1 1L9 5L1 9Z" fill="#0b2026"/>
# MAGIC         </marker>
# MAGIC       </defs>
# MAGIC       <rect x="18" y="40" width="166" height="62" rx="6" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="1.8"/>
# MAGIC       <text x="101" y="66" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Lakebase Postgres</text>
# MAGIC       <text x="101" y="86" text-anchor="middle" font-size="14" fill="#618794">orders, customers</text>
# MAGIC       <path d="M184 71 L241 71" fill="none" stroke="#0b2026" stroke-width="1.8" marker-end="url(#cdfai-arr4)"/>
# MAGIC       <text x="213" y="58" text-anchor="middle" font-size="14" font-weight="700" fill="#FF5F46">~15s</text>
# MAGIC       <rect x="259" y="40" width="166" height="62" rx="6" fill="rgba(255,95,70,0.12)" stroke="#FF5F46" stroke-width="1.8"/>
# MAGIC       <text x="342" y="66" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Lakebase CDF</text>
# MAGIC       <text x="342" y="86" text-anchor="middle" font-size="14" fill="#618794">change stream</text>
# MAGIC       <path d="M425 71 L482 71" fill="none" stroke="#0b2026" stroke-width="1.8" marker-end="url(#cdfai-arr4)"/>
# MAGIC       <rect x="500" y="40" width="166" height="62" rx="6" fill="#fff" stroke="#0b2026" stroke-width="1.3"/>
# MAGIC       <text x="583" y="66" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Delta table</text>
# MAGIC       <text x="583" y="86" text-anchor="middle" font-size="14" fill="#618794">in Unity Catalog</text>
# MAGIC       <path d="M666 71 L723 71" fill="none" stroke="#0b2026" stroke-width="1.8" marker-end="url(#cdfai-arr4)"/>
# MAGIC       <rect x="741" y="40" width="130" height="62" rx="6" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="1.8"/>
# MAGIC       <text x="806" y="66" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Vector index</text>
# MAGIC       <text x="806" y="86" text-anchor="middle" font-size="14" fill="#618794">Delta Sync</text>
# MAGIC       <path d="M871 71 L928 71" fill="none" stroke="#0b2026" stroke-width="1.8" marker-end="url(#cdfai-arr4)"/>
# MAGIC       <rect x="946" y="40" width="96" height="62" rx="6" fill="rgba(255,95,70,0.12)" stroke="#FF5F46" stroke-width="1.8"/>
# MAGIC       <text x="994" y="66" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">Agent</text>
# MAGIC       <text x="994" y="86" text-anchor="middle" font-size="14" fill="#618794">answers</text>
# MAGIC     </svg>
# MAGIC
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A6. Search Modes: Semantic, Keyword, and Hybrid
# MAGIC
# MAGIC Once an index exists, you can query it three ways — **semantic** (vector), **keyword**, or **hybrid**. The diagram traces one query through all three paths; click a card below to spotlight a single path.
# MAGIC <div class="aism">
# MAGIC   <style>
# MAGIC     .aism { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; color: #0b2026; width: 100%; max-width: 100%; margin: 0; padding: 4px 0; box-sizing: border-box; }
# MAGIC     .aism .diagram { width: 90%; max-width: 100%; margin: 0 auto; box-sizing: border-box; }
# MAGIC     .aism .grid { display: flex; flex-wrap: nowrap; gap: 18px; margin-top: 18px; width: 100%; }
# MAGIC     .aism .card { flex: 1 1 0; min-width: 0; background: #fff; border-radius: 14px; box-shadow: 0 2px 8px rgba(27,49,57,0.07); position: relative; padding: 16px 18px 18px; overflow: hidden; box-sizing: border-box; text-align: left; font: inherit; color: inherit; cursor: pointer; transition: box-shadow .15s ease, opacity .15s ease; }
# MAGIC     .aism .c1 { border: 1.5px solid #C6DCEA; }
# MAGIC     .aism .c2 { border: 1.5px solid #C6DCEA; }
# MAGIC     .aism .c3 { border: 1.5px solid #C6DCEA; }
# MAGIC     .aism .card h3 { margin: 6px 0 4px 0; font-size: 17px; font-weight: 600; }
# MAGIC     .aism .sub { font-size: 14px; color: #1B5162; margin: 0 0 12px 0; }
# MAGIC     .aism ul { margin: 0; padding-left: 18px; }
# MAGIC     .aism li { font-size: 14px; margin-bottom: 11px; line-height: 1.45; }
# MAGIC     .aism code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 14px; background: #EEEDE9; border-radius: 4px; padding: 1px 5px; }
# MAGIC     .aism svg { display: block; width: 100%; height: auto; margin: 0 auto; }
# MAGIC     .aism .hint { font-size: 14px; color: #618794; text-align: center; margin: 10px 0 0 0; }
# MAGIC     /* --- focus states: only active once a card is picked --- */
# MAGIC     .aism .card.is-dim { opacity: .55; }
# MAGIC     .aism .card.is-on { box-shadow: 0 0 0 3px rgba(34,114,180,.35), 0 2px 8px rgba(27,49,57,.10); }
# MAGIC     .aism g[data-part] { transition: opacity .18s ease; }
# MAGIC     .aism[data-focus] g[data-part] { opacity: .12; }
# MAGIC     .aism[data-focus="vector"] g[data-part~="vector"] { opacity: 1; }
# MAGIC     .aism[data-focus="keyword"] g[data-part~="keyword"] { opacity: 1; }
# MAGIC     .aism[data-focus="hybrid"] g[data-part~="hybrid"] { opacity: 1; }
# MAGIC     @media (max-width: 900px) { .aism .diagram { width: 100%; } .aism .grid { flex-wrap: wrap; } .aism .card { flex: 1 1 280px; } }
# MAGIC   </style>
# MAGIC   <div class="diagram">
# MAGIC     <svg viewBox="0 208 1000 262" preserveAspectRatio="xMidYMid meet" role="img" aria-label="One shared query splits into vector search using HNSW approximate nearest neighbor and keyword search using Okapi BM25, then both merge into a hybrid step that fuses them with reciprocal rank fusion into one ranked list.">
# MAGIC       <defs>
# MAGIC         <marker id="aismV" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M 0 0 L 10 5 L 0 10 z" fill="#4299E0"></path>
# MAGIC         </marker>
# MAGIC         <marker id="aismK" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M 0 0 L 10 5 L 0 10 z" fill="#FF5F46"></path>
# MAGIC         </marker>
# MAGIC         <marker id="aismG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M 0 0 L 10 5 L 0 10 z" fill="#7B61FF"></path>
# MAGIC         </marker>
# MAGIC         <marker id="aismA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
# MAGIC           <path d="M 0 0 L 10 5 L 0 10 z" fill="#FFAB00"></path>
# MAGIC         </marker>
# MAGIC       </defs>
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
# MAGIC         <path d="M 168 340 C 216 340, 220 256, 268 256" fill="none" stroke="#4299E0" stroke-width="2" marker-end="url(#aismV)"></path>
# MAGIC         <rect x="272" y="220" width="248" height="72" rx="8" fill="rgba(66,153,224,0.10)" stroke="#4299E0" stroke-width="2" stroke-dasharray="8 5"></rect>
# MAGIC         <text x="292" y="248" font-size="16" font-weight="600" fill="#0b2026">Vector (semantic)</text>
# MAGIC         <text x="292" y="272" font-size="14" fill="#2272B4">HNSW ANN &#183; nearest by distance</text>
# MAGIC       </g>
# MAGIC       <!-- ===== KEYWORD BRANCH ===== -->
# MAGIC       <g data-part="keyword hybrid">
# MAGIC         <path d="M 168 340 C 216 340, 220 424, 268 424" fill="none" stroke="#FF5F46" stroke-width="2" marker-end="url(#aismK)"></path>
# MAGIC         <rect x="272" y="388" width="248" height="72" rx="8" fill="rgba(255,95,70,0.10)" stroke="#FF5F46" stroke-width="2" stroke-dasharray="8 5"></rect>
# MAGIC         <text x="292" y="416" font-size="16" font-weight="600" fill="#0b2026">Keyword (full-text)</text>
# MAGIC         <text x="292" y="440" font-size="14" fill="#FF5F46">Okapi BM25 term relevance</text>
# MAGIC       </g>
# MAGIC       <!-- ===== MERGE + RANKED LIST (hybrid only) ===== -->
# MAGIC       <g data-part="hybrid">
# MAGIC         <path d="M 520 256 C 570 256, 562 340, 612 340" fill="none" stroke="#7B61FF" stroke-width="2" marker-end="url(#aismG)"></path>
# MAGIC         <path d="M 520 424 C 570 424, 562 340, 612 340" fill="none" stroke="#7B61FF" stroke-width="2" marker-end="url(#aismG)"></path>
# MAGIC         <rect x="616" y="302" width="160" height="76" rx="8" fill="rgba(123,97,255,0.10)" stroke="#7B61FF" stroke-width="2" stroke-dasharray="8 5"></rect>
# MAGIC         <text x="696" y="333" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Hybrid</text>
# MAGIC         <text x="696" y="356" text-anchor="middle" font-size="14" fill="#7B61FF">combine with RRF</text>
# MAGIC         <path d="M 776 340 L 824 340" fill="none" stroke="#FFAB00" stroke-width="2" marker-end="url(#aismA)"></path>
# MAGIC         <rect x="828" y="288" width="164" height="104" rx="8" fill="rgba(255,171,0,0.12)" stroke="#FFAB00" stroke-width="2"></rect>
# MAGIC         <text x="910" y="314" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">One ranked list</text>
# MAGIC         <rect x="848" y="326" width="124" height="14" rx="4" fill="#FFAB00"></rect>
# MAGIC         <rect x="848" y="346" width="102" height="14" rx="4" fill="#FFCC66"></rect>
# MAGIC         <rect x="848" y="366" width="84" height="14" rx="4" fill="#FFCC66"></rect>
# MAGIC       </g>
# MAGIC     </svg>
# MAGIC   </div>
# MAGIC   <div class="grid">
# MAGIC     <div class="card c1" data-focus-btn="vector" aria-pressed="false">
# MAGIC       <div style="height:5px; border-radius:5px; margin:-4px 0 12px; background:#4299E0;"></div>
# MAGIC       <h3>Vector (semantic) search</h3>
# MAGIC       <p class="sub">Approximate nearest neighbor (HNSW)</p>
# MAGIC       <ul>
# MAGIC         <li>Finds rows whose <strong>meaning</strong> is closest &mdash; even with no shared words.</li>
# MAGIC         <li>The query is embedded; the index returns nearest vectors via <strong>HNSW ANN</strong>.</li>
# MAGIC         <li>Relevance from L2 distance, scored <code>1 / (1 + distance&#178;)</code> &mdash; rank-equivalent to cosine with normalized vectors.</li>
# MAGIC       </ul>
# MAGIC     </div>
# MAGIC     <div class="card c2" data-focus-btn="keyword" aria-pressed="false">
# MAGIC       <div style="height:5px; border-radius:5px; margin:-4px 0 12px; background:#FF5F46;"></div>
# MAGIC       <h3>Keyword (full-text) search</h3>
# MAGIC       <p class="sub">Okapi BM25 relevance</p>
# MAGIC       <ul>
# MAGIC         <li>Ranks rows by how well they match your <strong>exact terms</strong>.</li>
# MAGIC         <li>Uses <strong>Okapi BM25</strong> scoring across text columns.</li>
# MAGIC         <li>Best for names, codes, SKUs, and exact-term lookups where wording matters.</li>
# MAGIC       </ul>
# MAGIC     </div>
# MAGIC     <div class="card c3" data-focus-btn="hybrid" aria-pressed="false">
# MAGIC       <div style="height:5px; border-radius:5px; margin:-4px 0 12px; background:#7B61FF;"></div>
# MAGIC       <h3>Hybrid search</h3>
# MAGIC       <p class="sub">Runs both, merges the results</p>
# MAGIC       <ul>
# MAGIC         <li>Combines both result sets into <strong>one ranked list</strong>.</li>
# MAGIC         <li>Merges with <strong>Reciprocal Rank Fusion (RRF)</strong>.</li>
# MAGIC         <li>Best when queries mix intent with specific terms &mdash; a strong default.</li>
# MAGIC       </ul>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <p class="hint">Click a card to spotlight that path in the diagram &mdash; click it again to show everything.</p>
# MAGIC   <script>
# MAGIC   (function () {
# MAGIC     var root = document.currentScript.closest('.aism');
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
# MAGIC ## B. Managed vs. Custom RAG

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Two Paths to RAG
# MAGIC
# MAGIC Now that you've built the full custom pipeline (parse, chunk, index, search), let's zoom out. Databricks provides two complementary approaches to building retrieval systems:
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 320" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Managed RAG vs Custom RAG: two paths from documents to answers</title>
# MAGIC   <defs>
# MAGIC     <marker id="rag-arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#FF5F46" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC     <marker id="rag-arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#2272B4" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC
# MAGIC   <!-- Documents (shared start) -->
# MAGIC   <rect x="10" y="110" width="130" height="90" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="75" y="140" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Documents</text>
# MAGIC   <text x="75" y="160" text-anchor="middle" font-size="13" fill="#618794">PDFs, tables,</text>
# MAGIC   <text x="75" y="176" text-anchor="middle" font-size="13" fill="#618794">UC Volumes</text>
# MAGIC
# MAGIC   <!-- Fork arrows -->
# MAGIC   <path d="M140 135 L210 65" fill="none" stroke="#FF5F46" stroke-width="1.8" marker-end="url(#rag-arrow-green)"/>
# MAGIC   <path d="M140 175 L210 235" fill="none" stroke="#2272B4" stroke-width="1.8" marker-end="url(#rag-arrow-blue)"/>
# MAGIC
# MAGIC   <!-- MANAGED PATH (top) -->
# MAGIC   <rect x="215" y="15" width="310" height="100" rx="10" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="370" y="42" text-anchor="middle" font-size="17" font-weight="700" fill="#FF5F46">Managed RAG</text>
# MAGIC   <text x="370" y="62" text-anchor="middle" font-size="14" fill="#0b2026">Genie Agent</text>
# MAGIC   <text x="370" y="82" text-anchor="middle" font-size="13" fill="#618794">Parsing, chunking, embedding,</text>
# MAGIC   <text x="370" y="98" text-anchor="middle" font-size="13" fill="#618794">indexing, serving are automatically implemented</text>
# MAGIC
# MAGIC   <!-- CUSTOM PATH (bottom) -->
# MAGIC   <rect x="215" y="190" width="310" height="110" rx="10" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="370" y="218" text-anchor="middle" font-size="17" font-weight="700" fill="#2272B4">Custom RAG</text>
# MAGIC   <text x="370" y="238" text-anchor="middle" font-size="14" fill="#0b2026">AI SQL Functions + Additional Logic</text>
# MAGIC   <text x="370" y="258" text-anchor="middle" font-size="13" fill="#618794">ai_parse_document → ai_prep_search</text>
# MAGIC   <text x="370" y="274" text-anchor="middle" font-size="13" fill="#618794">→ Delta table → AI Search index</text>
# MAGIC   <text x="370" y="290" text-anchor="middle" font-size="13" fill="#618794">Full control over every component</text>
# MAGIC
# MAGIC   <!-- Converge arrows -->
# MAGIC   <path d="M525 65 L600 135" fill="none" stroke="#FF5F46" stroke-width="1.8" marker-end="url(#rag-arrow-green)"/>
# MAGIC   <path d="M525 245 L600 175" fill="none" stroke="#2272B4" stroke-width="1.8" marker-end="url(#rag-arrow-blue)"/>
# MAGIC
# MAGIC   <!-- Answers (shared end) -->
# MAGIC   <rect x="605" y="110" width="160" height="90" rx="8" fill="#ffffff" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="685" y="140" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Answers</text>
# MAGIC   <text x="685" y="160" text-anchor="middle" font-size="13" fill="#618794">LLM + retrieved</text>
# MAGIC   <text x="685" y="176" text-anchor="middle" font-size="13" fill="#618794">context</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC **Managed RAG (Genie Agents):**
# MAGIC - You provide documents (in a UC Volume or AI Search Index)
# MAGIC - The system handles parsing, chunking, embedding, indexing, and serving
# MAGIC - Optimized automatically via feedback loops
# MAGIC - Best for: standard RAG use cases where speed to production matters
# MAGIC
# MAGIC **Custom RAG (Agent Framework + AI Search):**
# MAGIC - You build the pipeline yourself with full ownership of each step
# MAGIC - Full control over parsing strategy, chunking approach, embedding model, and retrieval logic
# MAGIC - Use AI Search indexes as retrieval tools in your custom agent
# MAGIC - Best for: unique requirements, complex multi-step reasoning, novel architectures

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Governance with Unity Catalog
# MAGIC
# MAGIC AI Search indexes are governed by Unity Catalog. Access to Unity Catalog objects in the retrieval pipeline is controlled with Unity Catalog privileges, while AI search endpoints are controlled with endpoint ACLs. Audit visibility is available through system tables, and lineage can be captured across Unity Catalog tables and paths used in the retrieval pipeline.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 820 200" role="img" style="font-family: sans-serif;">
# MAGIC  <title>Governance across the retrieval pipeline with Unity Catalog and AI search endpoint ACLs</title>
# MAGIC  <defs>
# MAGIC  <marker id="uc-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC  <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC  </marker>
# MAGIC  </defs>
# MAGIC
# MAGIC  <!-- Governance banner -->
# MAGIC  <rect x="10" y="10" width="800" height="180" rx="10" fill="none" stroke="#1B3139" stroke-width="1" stroke-opacity="0.3"/>
# MAGIC  <text x="410" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#1B3139">Governance across the retrieval pipeline</text>
# MAGIC
# MAGIC  <!-- UC Volume -->
# MAGIC  <rect x="30" y="58" width="120" height="56" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.2"/>
# MAGIC  <text x="90" y="82" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">UC Volume</text>
# MAGIC  <text x="90" y="100" text-anchor="middle" font-size="12" fill="#618794">Source PDFs</text>
# MAGIC
# MAGIC  <path d="M150 86 L185 86" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#uc-arrow)"/>
# MAGIC
# MAGIC  <!-- Delta Table -->
# MAGIC  <rect x="190" y="58" width="145" height="56" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.2"/>
# MAGIC  <text x="262.5" y="82" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Delta Table</text>
# MAGIC  <text x="262.5" y="100" text-anchor="middle" font-size="12" fill="#618794">Chunks + metadata</text>
# MAGIC
# MAGIC  <path d="M335 86 L370 86" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#uc-arrow)"/>
# MAGIC
# MAGIC  <!-- VS Index -->
# MAGIC  <rect x="375" y="58" width="135" height="56" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC  <text x="442.5" y="82" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">VS Index</text>
# MAGIC  <text x="442.5" y="100" text-anchor="middle" font-size="12" fill="#618794">Vectors + synced cols</text>
# MAGIC
# MAGIC  <path d="M510 86 L555 86" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#uc-arrow)"/>
# MAGIC
# MAGIC  <!-- VS Endpoint -->
# MAGIC  <rect x="560" y="58" width="120" height="56" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.2"/>
# MAGIC  <text x="620" y="82" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">VS Endpoint</text>
# MAGIC  <text x="620" y="100" text-anchor="middle" font-size="12" fill="#618794">Query interface</text>
# MAGIC
# MAGIC  <path d="M680 86 L715 86" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#uc-arrow)"/>
# MAGIC
# MAGIC  <!-- KA / Agent -->
# MAGIC  <rect x="720" y="58" width="70" height="56" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.2"/>
# MAGIC  <text x="755" y="82" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Agent</text>
# MAGIC  <text x="755" y="100" text-anchor="middle" font-size="12" fill="#618794">RAG</text>
# MAGIC
# MAGIC  <!-- Governance labels -->
# MAGIC  <g>
# MAGIC  <text x="90" y="132" text-anchor="middle" font-size="12" fill="#618794">UC privileges</text>
# MAGIC  <text x="262.5" y="132" text-anchor="middle" font-size="12" fill="#618794">UC privileges</text>
# MAGIC  <text x="442.5" y="132" text-anchor="middle" font-size="12" fill="#618794">UC privileges</text>
# MAGIC  <text x="620" y="132" text-anchor="middle" font-size="12" fill="#618794">Endpoint ACLs</text>
# MAGIC  <text x="755" y="132" text-anchor="middle" font-size="12" fill="#618794">Queries via</text>
# MAGIC  <text x="755" y="146" text-anchor="middle" font-size="12" fill="#618794">endpoint</text>
# MAGIC
# MAGIC </svg>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. AI Search vs. Lakebase Search
# MAGIC
# MAGIC Databricks now offers two search options for retrieval. They complement each other and serve different architectures:
# MAGIC
# MAGIC |  | **AI Search** | **Lakebase Search** |
# MAGIC |---|---|---|
# MAGIC | **What it is** | Managed search service over Unity Catalog tables and volumes — handles embedding, indexing, and retrieval as a fully managed stack | Postgres extensions adding vector + keyword search inside a Lakebase project |
# MAGIC | **Primary use case** | Production search — search bars, recommendation systems, and document RAG for unstructured content (PDFs, docs, slides) | Operational search — query structured and semi-structured data already in Postgres |
# MAGIC | **Used by** | Search bars, recommender systems, Genie Agents, custom RAG agents, any application calling the AI Search API | Applications backed by Lakebase — customer-facing apps, internal tools, agent memory |
# MAGIC | **Search modes** | Vector (ANN), keyword (BM25), hybrid (RRF), and hybrid + reranker | Vector (ANN), keyword (BM25), and hybrid (RRF fusion) |
# MAGIC | **JOINs with operational data** | No — results come back as a separate retrieval call | Yes — search, JOIN, and filter in one SQL query, one transaction |
# MAGIC | **Permissions model** | Unity Catalog ACLs on the index | Postgres row-level GRANTs on the same tables |
# MAGIC | **Index management** | Fully managed by Databricks | You create indexes with `CREATE INDEX ... USING lakebase_ann / lakebase_bm25` |
# MAGIC | **Lakehouse integration** | Native — indexes sync from UC tables/volumes via Delta Sync | Via synced tables — sync a UC table into Lakebase with type mapping for search columns |
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">When to choose which</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Use <strong>AI Search</strong> when you want a fully managed search engine — search bars, recommendations, or document RAG — with built-in hybrid retrieval, reranking, and quality evaluation out of the box. Use <strong>Lakebase Search</strong> when your data already lives in Lakebase and you want low-latency search co-located with your operational tables — for example, searching customer reviews, product catalogs, or support tickets alongside structured metadata in a single SQL query.
# MAGIC </p>
# MAGIC </div>

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
# MAGIC     Want to learn more about AI Search and embedding strategies? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/building-rag-agents-with-agent-bricks-en_us-2.0.0/images/20261014T152401Z/Building RAG Agents with Agent Bricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #F9F7F4; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-query" style="flex: 1;">How does Databricks AI Search handle hybrid keyword and semantic search?</span>
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

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC