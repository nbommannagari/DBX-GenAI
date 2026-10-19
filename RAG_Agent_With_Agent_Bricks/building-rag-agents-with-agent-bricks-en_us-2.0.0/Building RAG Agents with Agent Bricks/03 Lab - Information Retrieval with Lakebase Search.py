# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Lab - Information Retrieval with Lakebase Search
# MAGIC #### Duration: 30 minutes
# MAGIC ## Overview
# MAGIC In this lab you will work with **Lakebase Search** on a **Lakebase Postgres** branch that has been pre-loaded with Airbnb listing document chunks (with pre-computed **1024-dimensional embeddings** and **tsvector** values). You will use copyable SQL to explore three search workloads: **vector similarity**, **keyword (tsvector)**, and a **combined hybrid** query.
# MAGIC
# MAGIC This gives you hands-on practice with the Lakebase retrieval workflow — connecting to a pre-provisioned database and running search queries.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lab, you will be able to:
# MAGIC 1. **Connect** to a pre-provisioned Lakebase project using the `lb_connect()` factory
# MAGIC 2. **Run** vector similarity search using cosine distance (`<=>`) over pgvector embeddings
# MAGIC 3. **Run** keyword search using pre-computed `tsvector` columns and `ts_rank`
# MAGIC 4. **Run** hybrid search combining vector and keyword signals with weighted scoring
# MAGIC 5. **Compare** the results of different search strategies

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
# MAGIC <li><strong>Serverless Compute, Version 6</strong>: How to select an environment version (<a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V6</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Classroom Setup

# COMMAND ----------

# Get the current user's username
username = spark.sql("SELECT current_user()").first()[0]
print(f"{username.split('@')[0].replace(".","-").replace("_","-")}-lb-search")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Lakebase Project Setup
# MAGIC Before running the classroom setup, we will need to do 2 things: 
# MAGIC 1. Create a Lakebase Project
# MAGIC     - Navigate to the waffle at the top right, click on it, right click on **Lakebase Postgres** and select **Open Link in New Tab**.
# MAGIC     - On the next screen, click **⊕ New Project**
# MAGIC     - Give it the name `<labusername>-lb-search`. Running the previous cell will output what you should copy and paste into the name field.  
# MAGIC         - Example: `labuser_123abc-lb-search`
# MAGIC     - Choose Postgres version 18 (other version may work but this course has been tested on 18)
# MAGIC     - Click **Create**
# MAGIC
# MAGIC 2. Turn on Lakebase Search
# MAGIC     - Navigate into your Lakebase Project and click on **Settings**
# MAGIC     - Scroll down to the very bottom and click **Enable Lakebase Search** and confirm to enable Lakebase Search
# MAGIC
# MAGIC 3. Navigate back to the workspace, but leave an additional tab open so we can demonstrate different search methods.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <strong style="color: #c62828; font-size: 1.1em;">Important</strong>
# MAGIC   <div style="color: #333;">
# MAGIC
# MAGIC You must name your Lakebase project correctly, or the classroom setup below will not run.
# MAGIC
# MAGIC If you see the following error, your Lakebase project was named incorrectly.
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC RuntimeError: 
# MAGIC ====================================================================
# MAGIC Lakebase project not found: labuser-xxx-lb-search
# MAGIC ```
# MAGIC
# MAGIC   </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Classroom Setup
# MAGIC Run the cell below to set up your catalog, schema, and other classroom assets needed to this notebook.

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-03

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A3. Lakebase Tables: Under the Hood
# MAGIC The previous setup tasks did two important things for you:
# MAGIC 1. Create the Lakebase Project with the UI and enabled the **Lakebase Search** feature
# MAGIC
# MAGIC 2. Populated our project with a table called **doc_chunks** in the **databricks_postgres.public** schema.
# MAGIC
# MAGIC This was performed for you to bypass some heavy code, but it's worth understanding different methods for getting these tables situated. We summarize them below, but the technical details go beyond the scope of this course. Please see our [additional trainings](https://www.databricks.com/training/catalog?search=lakebase) for a deeper dive into these concepts as well as our docs pages ([AWS](https://docs.databricks.com/aws/en/oltp/projects/lakebase-search) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/oltp/projects/lakebase-search))
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <div class="lbss">
# MAGIC   <style>
# MAGIC     .lbss{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1120px;margin:0 auto;color:#1B3139;}
# MAGIC     .lbss *{box-sizing:border-box;}
# MAGIC     .lbss code{font-family:ui-monospace,Menlo,Consolas,monospace;}
# MAGIC     .lbss .mono{font-family:ui-monospace,Menlo,Consolas,monospace;background:#EEEDE9;border-radius:5px;padding:2px 8px;font-size:13.5px;color:#0b2026;white-space:nowrap;}
# MAGIC     .lbss .head{text-align:center;margin-bottom:6px;}
# MAGIC     .lbss .h1{font-size:26px;font-weight:800;color:#FF3621;}
# MAGIC     .lbss .sub{font-size:15px;color:#5A6A70;margin-top:4px;}
# MAGIC     .lbss .eyebrow{font-size:12px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;color:#5A6A70;margin:22px 0 10px;}
# MAGIC     .lbss .ing{display:flex;gap:18px;flex-wrap:nowrap;}
# MAGIC     .lbss .card{flex:1 1 0;min-width:0;background:#fff;border:1.5px solid #C6DCEA;border-radius:14px;padding:16px 18px 18px;box-shadow:0 2px 8px rgba(27,49,57,.07);}
# MAGIC     .lbss .bar{height:5px;border-radius:5px;margin:-4px 0 12px;}
# MAGIC     .lbss .cap{font-size:16px;font-weight:800;}
# MAGIC     .lbss .flow{display:flex;align-items:center;gap:10px;margin:12px 0 10px;flex-wrap:wrap;}
# MAGIC     .lbss .arrow{color:#9DB0BA;font-size:18px;font-weight:800;}
# MAGIC     .lbss .note{font-size:12.5px;color:#5A6A70;line-height:1.45;}
# MAGIC     .lbss .tag{font-size:11px;color:#5A6A70;}
# MAGIC     .lbss>input{position:absolute;opacity:0;pointer-events:none;}
# MAGIC     .lbss .tabs{display:flex;gap:12px;flex-wrap:nowrap;}
# MAGIC     .lbss .btn{flex:1 1 0;min-width:0;display:block;cursor:pointer;background:#fff;border:1.5px solid #C6DCEA;border-radius:12px;padding:13px 15px;transition:all .12s;}
# MAGIC     .lbss .btn .n{display:inline-flex;width:22px;height:22px;border-radius:50%;background:#EEF1F3;color:#5A6A70;font-size:12px;font-weight:800;align-items:center;justify-content:center;margin-bottom:7px;}
# MAGIC     .lbss .btn .t{font-size:15px;font-weight:800;color:#1B3139;line-height:1.2;}
# MAGIC     .lbss .btn .w{font-size:12px;color:#5A6A70;margin-top:3px;line-height:1.35;}
# MAGIC     .lbss .btn:hover{border-color:#FF5F46;}
# MAGIC     .lbss .panel{display:none;background:#fff;border:1.5px solid #FF5F46;border-radius:14px;padding:18px 20px;margin-top:14px;box-shadow:0 4px 14px rgba(255,95,70,.12);}
# MAGIC     .lbss .step{display:flex;gap:12px;margin-bottom:12px;}
# MAGIC     .lbss .step:last-child{margin-bottom:0;}
# MAGIC     .lbss .sn{flex:none;width:24px;height:24px;border-radius:50%;background:#FF5F46;color:#fff;font-size:13px;font-weight:800;display:flex;align-items:center;justify-content:center;}
# MAGIC     .lbss .sd{font-size:14px;color:#1B3139;line-height:1.45;padding-top:2px;}
# MAGIC     .lbss #lbss-m1:checked ~ .tabs label[for=lbss-m1],
# MAGIC     .lbss #lbss-m2:checked ~ .tabs label[for=lbss-m2],
# MAGIC     .lbss #lbss-m3:checked ~ .tabs label[for=lbss-m3]{background:#FF5F46;border-color:#FF5F46;box-shadow:0 4px 12px rgba(255,95,70,.22);}
# MAGIC     .lbss #lbss-m1:checked ~ .tabs label[for=lbss-m1] .t,
# MAGIC     .lbss #lbss-m2:checked ~ .tabs label[for=lbss-m2] .t,
# MAGIC     .lbss #lbss-m3:checked ~ .tabs label[for=lbss-m3] .t,
# MAGIC     .lbss #lbss-m1:checked ~ .tabs label[for=lbss-m1] .w,
# MAGIC     .lbss #lbss-m2:checked ~ .tabs label[for=lbss-m2] .w,
# MAGIC     .lbss #lbss-m3:checked ~ .tabs label[for=lbss-m3] .w{color:#fff;}
# MAGIC     .lbss #lbss-m1:checked ~ .tabs label[for=lbss-m1] .n,
# MAGIC     .lbss #lbss-m2:checked ~ .tabs label[for=lbss-m2] .n,
# MAGIC     .lbss #lbss-m3:checked ~ .tabs label[for=lbss-m3] .n{background:rgba(255,255,255,.28);color:#fff;}
# MAGIC     .lbss #lbss-m1:checked ~ .panels #lbss-p1,
# MAGIC     .lbss #lbss-m2:checked ~ .panels #lbss-p2,
# MAGIC     .lbss #lbss-m3:checked ~ .panels #lbss-p3{display:block;}
# MAGIC   </style>
# MAGIC
# MAGIC   <div class="eyebrow">Click each card to view different methods</div>
# MAGIC   <input type="radio" name="lbss-m" id="lbss-m1" checked>
# MAGIC   <input type="radio" name="lbss-m" id="lbss-m2">
# MAGIC   <input type="radio" name="lbss-m" id="lbss-m3">
# MAGIC   <div class="tabs">
# MAGIC     <label class="btn" for="lbss-m1"><span class="n">1</span><div class="t">Start from scratch</div><div class="w">No table yet &mdash; build it in Lakebase</div></label>
# MAGIC     <label class="btn" for="lbss-m2"><span class="n">2</span><div class="t">Existing Lakebase table</div><div class="w">Columns already there &mdash; just index</div></label>
# MAGIC     <label class="btn" for="lbss-m3"><span class="n">3</span><div class="t">Sync from Unity Catalog</div><div class="w">Data lives in UC &mdash; sync it in</div></label>
# MAGIC   </div>
# MAGIC   <div class="panels">
# MAGIC     <div class="panel" id="lbss-p1">
# MAGIC       <div class="step"><div class="sn">1</div><div class="sd">Create a new table with <span class="mono">VECTOR(n)</span> and <span class="mono">TSVECTOR</span> columns.</div></div>
# MAGIC       <div class="step"><div class="sn">2</div><div class="sd">Insert your data directly into the table.</div></div>
# MAGIC       <div class="step"><div class="sn">3</div><div class="sd">Build the <span class="mono" style="background:#FFF0ED;color:#C0341D;">lakebase_ann</span> and <span class="mono" style="background:#E7EEF1;color:#1B5162;">lakebase_bm25</span> indexes.</div></div>
# MAGIC     </div>
# MAGIC     <div class="panel" id="lbss-p2">
# MAGIC       <div class="step"><div class="sn">1</div><div class="sd">You already have a Lakebase table with the proper <span class="mono">VECTOR</span> and <span class="mono">TSVECTOR</span> columns.</div></div>
# MAGIC       <div class="step"><div class="sn">2</div><div class="sd">Just create the <span class="mono" style="background:#FFF0ED;color:#C0341D;">lakebase_ann</span> and <span class="mono" style="background:#E7EEF1;color:#1B5162;">lakebase_bm25</span> indexes on the existing columns.</div></div>
# MAGIC     </div>
# MAGIC     <div class="panel" id="lbss-p3">
# MAGIC       <div class="step"><div class="sn">1</div><div class="sd">Your embeddings and source text live in a <b>Unity Catalog</b> table.</div></div>
# MAGIC       <div class="step"><div class="sn">2</div><div class="sd">Use <b>synced tables</b> with custom type mappings: map the embedding column to <span class="mono">VECTOR(n)</span> and generate a <span class="mono">TSVECTOR</span> column from a source text column during the sync.</div></div>
# MAGIC       <div class="step"><div class="sn">3</div><div class="sd">Build the <span class="mono" style="background:#FFF0ED;color:#C0341D;">lakebase_ann</span> and <span class="mono" style="background:#E7EEF1;color:#1B5162;">lakebase_bm25</span> indexes on the synced columns.</div></div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Querying Lakebase Tables
# MAGIC
# MAGIC More generally, when querying Lakebase tables with a managed SQL editor on Databricks, there are two options: 
# MAGIC - **Lakehouse SQL Editor**
# MAGIC     - Navigate to the left side menu and, in a new tab, open the SQL Editor. 
# MAGIC         1. **Select the compute resource**: Click on the compute dropdown at the top right and select **More...**
# MAGIC         2. Under **Compute type**, select **Lakebase Postgres**
# MAGIC         3. Select **Autoscaling** and choose the Lakebase project name shown in the output table from running the classroom setup script above
# MAGIC         4. Click **Attach**
# MAGIC         5. Begin querying
# MAGIC
# MAGIC - **Lakebase SQL Editor**
# MAGIC     - Navigate to the waffle at the top right and click on **Lakebase Postrgres**. 
# MAGIC         1. Locate your project by searching for the same project name that is shown in the output of the classroom setup. Be sure to leverage the copy button provided. 
# MAGIC         2. Click on **Tables**. You will see the table **doc_chunks**. This table is actually a snapshot of a volume that has been indexed via enabling **Content Search**.  
# MAGIC
# MAGIC You can choose either **SQL Editor** in what follows.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Querying Lakebase Search
# MAGIC
# MAGIC With the data loaded and indexed by the classroom setup, the next three blocks provide copyable Lakebase SQL versions of the search workloads against the **doc_chunks** table.
# MAGIC
# MAGIC Instead of using an existing chunk as the probe, we will use a natural-language query that matches the Airbnb data loaded into the table. A good example from this dataset is:
# MAGIC
# MAGIC `I want a Victorian house in Hayes Valley for a month-long stay.`
# MAGIC
# MAGIC That question should surface content about the Victorian listings that mention Hayes Valley and minimum one-month stays.
# MAGIC
# MAGIC <div class="scn">
# MAGIC   <style>
# MAGIC     .scn { max-width: 1280px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #0b2026; }
# MAGIC     .scn svg { display: block; width: 100%; height: auto; }
# MAGIC   </style>
# MAGIC   <svg viewBox="0 0 1280 384" role="img" aria-label="User query is embedded outside the database, the vector is passed into Lakebase Postgres for ANN search, and results return to the user.">
# MAGIC     <defs>
# MAGIC       <marker id="scn-ar-g" viewBox="0 0 10 10" refX="0" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#618794"/></marker>
# MAGIC       <marker id="scn-ar-l" viewBox="0 0 10 10" refX="0" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#FF5F46"/></marker>
# MAGIC       <marker id="scn-ar-n" viewBox="0 0 10 10" refX="0" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#1B5162"/></marker>
# MAGIC     </defs>
# MAGIC     <!-- ================================================================
# MAGIC          LAYOUT GRID
# MAGIC          card row:      y 100 -> 244  (center y = 172)
# MAGIC          zones:         y 56  -> 268
# MAGIC          cards 1-3:     192w at x 160 / 402 / 644  (uniform 50px gaps)
# MAGIC          gap 1: 352 -> 402, center 377  | arrow line 358 -> 389, tip 400
# MAGIC          gap 2: 594 -> 644, center 619  | arrow line 600 -> 631, tip 642
# MAGIC          outside zone:  x 136 -> 860
# MAGIC          inside zone:   x 960 -> 1264
# MAGIC          all h-arrows:  y 172
# MAGIC          return loop:   y 332
# MAGIC        ================================================================ -->
# MAGIC     <!-- ===== User ===== -->
# MAGIC     <text x="56" y="110" text-anchor="middle" font-size="14" font-weight="700" fill="#0b2026">User</text>
# MAGIC     <circle cx="56" cy="142" r="13" fill="#1B5162"/>
# MAGIC     <path d="M33 192 C33 172 43 159 56 159 C69 159 79 172 79 192 Z" fill="#1B5162"/>
# MAGIC     <line x1="88" y1="172" x2="114" y2="172" stroke="#618794" stroke-width="2.4" marker-end="url(#scn-ar-g)"/>
# MAGIC     <!-- ===== OUTSIDE zone ===== -->
# MAGIC     <rect x="136" y="56" width="724" height="212" rx="14" fill="#F8F9FC" stroke="#C4CCD6" stroke-width="1.5" stroke-dasharray="6 5"/>
# MAGIC     <text x="160" y="84" font-size="14" font-weight="800" letter-spacing="0.6" fill="#5A6F77">YOUR APP &#183; NOTEBOOK &#183; PIPELINE</text>
# MAGIC     <!-- box 1: x 160 -> 352, center 256 -->
# MAGIC     <rect x="160" y="100" width="192" height="144" rx="10" fill="rgba(66,153,224,0.12)" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC     <circle cx="256" cy="132" r="13" fill="rgba(66,153,224,0.22)"/><text x="256" y="137" text-anchor="middle" font-size="14" font-weight="800" fill="#2272B4">1</text>
# MAGIC     <text x="256" y="174" text-anchor="middle" font-size="16" font-weight="800" fill="#0b2026">User query</text>
# MAGIC     <text x="256" y="202" text-anchor="middle" font-size="14" fill="#618794">plain text</text>
# MAGIC     <!-- arrow 1 -> 2 : centered in gap 352-402 -->
# MAGIC     <line x1="358" y1="172" x2="389" y2="172" stroke="#618794" stroke-width="2.4" marker-end="url(#scn-ar-g)"/>
# MAGIC     <!-- box 2: x 402 -> 594, center 498 -->
# MAGIC     <rect x="402" y="100" width="192" height="144" rx="10" fill="rgba(0,169,114,0.12)" stroke="#00A972" stroke-width="1.5"/>
# MAGIC     <circle cx="498" cy="132" r="13" fill="rgba(0,169,114,0.22)"/><text x="498" y="137" text-anchor="middle" font-size="14" font-weight="800" fill="#00A972">2</text>
# MAGIC     <text x="498" y="172" text-anchor="middle" font-size="16" font-weight="800" fill="#0b2026">Embed the text</text>
# MAGIC     <text x="498" y="196" text-anchor="middle" font-size="14" fill="#618794">text &#8594; 1024-float vector</text>
# MAGIC     <rect x="412" y="208" width="172" height="24" rx="6" fill="rgba(0,169,114,0.18)"/>
# MAGIC     <text x="498" y="225" text-anchor="middle" font-size="14" font-family="ui-monospace,Menlo,Consolas,monospace" fill="#0b2026">ai_query() &#183; API</text>
# MAGIC     <!-- arrow 2 -> 3 : centered in gap 594-644 -->
# MAGIC     <line x1="600" y1="172" x2="631" y2="172" stroke="#618794" stroke-width="2.4" marker-end="url(#scn-ar-g)"/>
# MAGIC     <!-- box 3: x 644 -> 836, center 740 -->
# MAGIC     <rect x="644" y="100" width="192" height="144" rx="10" fill="rgba(255,171,0,0.14)" stroke="#FFAB00" stroke-width="1.5"/>
# MAGIC     <circle cx="740" cy="132" r="13" fill="rgba(255,171,0,0.26)"/><text x="740" y="137" text-anchor="middle" font-size="14" font-weight="800" fill="#0b2026">3</text>
# MAGIC     <text x="740" y="174" text-anchor="middle" font-size="16" font-weight="800" fill="#0b2026">Pass vector to SQL</text>
# MAGIC     <text x="740" y="202" text-anchor="middle" font-size="14" fill="#618794">the literal to compare</text>
# MAGIC     <!-- ===== boundary arrow: gap 860 -> 960, centered ===== -->
# MAGIC     <line x1="878" y1="172" x2="930" y2="172" stroke="#FF5F46" stroke-width="3.2" marker-end="url(#scn-ar-l)"/>
# MAGIC     <!-- ===== INSIDE zone ===== -->
# MAGIC     <rect x="960" y="56" width="304" height="212" rx="14" fill="#FFF6F4" stroke="#FF5F46" stroke-width="2"/>
# MAGIC     <text x="984" y="84" font-size="14" font-weight="800" letter-spacing="0.6" fill="#FF5F46">LAKEBASE (POSTGRES)</text>
# MAGIC     <!-- ANN box: x 984 -> 1240, center 1112 -->
# MAGIC     <rect x="984" y="100" width="256" height="144" rx="10" fill="rgba(255,95,70,0.12)" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC     <circle cx="1112" cy="132" r="13" fill="#FF5F46"/><text x="1112" y="137" text-anchor="middle" font-size="14" font-weight="800" fill="#FFFFFF">4</text>
# MAGIC     <text x="1112" y="172" text-anchor="middle" font-size="16" font-weight="800" fill="#0b2026">ANN search</text>
# MAGIC     <text x="1112" y="196" text-anchor="middle" font-size="14" fill="#618794">returns nearest rows</text>
# MAGIC     <rect x="1026" y="208" width="172" height="24" rx="6" fill="rgba(255,95,70,0.22)"/>
# MAGIC     <text x="1112" y="225" text-anchor="middle" font-size="14" font-family="ui-monospace,Menlo,Consolas,monospace" fill="#98102A">lakebase_ann</text>
# MAGIC     <!-- ===== return loop ===== -->
# MAGIC     <path d="M1112 268 L1112 332 L840 332" fill="none" stroke="#1B5162" stroke-width="2.4"/>
# MAGIC     <path d="M444 332 L56 332 L56 200" fill="none" stroke="#1B5162" stroke-width="2.4" marker-end="url(#scn-ar-n)"/>
# MAGIC     <text x="642" y="338" text-anchor="middle" font-size="14" font-weight="700" fill="#1B5162">5 &#183; Results go back to the user</text>
# MAGIC   </svg>
# MAGIC </div>
# MAGIC
# MAGIC Here you will use natural language to create an embedding with `ai_query()` and you will pass the resulting embedding using SQL in the **Lakebase SQL Editor**.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Following the diagram from the cell above, we're first going to pass our query to `ai_query()` to render the embedding and save. 
# MAGIC
# MAGIC Now, of course in production, we would wire this into an application, perform proper tracing, etc. But this will give you an idea of the workflow using vector search with Lakebase. Copy and paste the following into the SQL block below to generate the embedding. 
# MAGIC
# MAGIC <div class="code-card">
# MAGIC       <div class="code-card-header">
# MAGIC         <span class="code-card-lang">Prompt</span>
# MAGIC         <button class="code-card-copy" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent,t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.classList.add('copied');setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.classList.remove('copied')},2000)}})(this)">
# MAGIC           <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
# MAGIC             <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
# MAGIC           </svg>
# MAGIC           <svg class="check-icon" style="display:none" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
# MAGIC             <polyline points="20 6 9 17 4 12"/>
# MAGIC           </svg>
# MAGIC           <span class="copy-label">Copy</span>
# MAGIC         </button>
# MAGIC       </div>
# MAGIC       <div class="code-card-body">I want a Victorian house in Hayes Valley for a month-long stay.</div>
# MAGIC     </div>
# MAGIC
# MAGIC <style>
# MAGIC   .code-card { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; border: 1px solid #DCE0E2; border-radius: 0.5rem; overflow: hidden; background: #F9F7F4; max-width: 100%; margin: 0.5rem 0; }
# MAGIC   .code-card-header { display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0.75rem; background: #EEEDE9; border-bottom: 1px solid #DCE0E2; }
# MAGIC   .code-card-lang { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #5A6F77; }
# MAGIC   .code-card-copy { display: inline-flex; align-items: center; gap: 0.3rem; background: none; border: none; color: #5A6F77; cursor: pointer; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-size: 0.7rem; font-weight: 500; transition: color 0.15s, background 0.15s; }
# MAGIC   .code-card-copy:hover { color: #1B3139; background: rgba(0,0,0,0.05); }
# MAGIC   .code-card-copy.copied { color: #2272B4; }
# MAGIC   .code-card-copy svg { width: 14px; height: 14px; }
# MAGIC   .code-card-body { margin: 0; padding: 1rem; overflow-x: auto; font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace; font-size: 0.82rem; line-height: 1.55; color: #1B3139; white-space: pre-wrap; word-break: break-word; }
# MAGIC   .catalog-input-card { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; border: 2px solid #1976d2; background: #e3f2fd; padding: 14px 20px; border-radius: 8px; margin: 16px 0; font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; }
# MAGIC   .catalog-input-card label { font-weight: 700; color: #0d47a1; font-size: 0.95rem; white-space: nowrap; }
# MAGIC   .catalog-input-card input { font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace; font-size: 0.9rem; padding: 8px 14px; border: 2px solid #1976d2; border-radius: 6px; background: #fff; min-width: 280px; color: #1B3139; outline: none; transition: border-color 0.2s, box-shadow 0.2s; }
# MAGIC   .catalog-input-card input:focus { border-color: #0d47a1; box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.15); }
# MAGIC   .catalog-fill { background: #fff3cd; padding: 1px 4px; border-radius: 3px; border-bottom: 2px solid #ffc107; font-weight: 600; }
# MAGIC </style>
# MAGIC
# MAGIC You will need to leverage a hosted model within Databricks to intiate `ai_query()`. 
# MAGIC
# MAGIC More importantly, the embedding model _must_ match dimensions in our lakebase table. 
# MAGIC
# MAGIC 1. Navgate back to the **doc_chunks** table and determine the dimensionality of your embedding column **embedding**. What dimension do you see? 
# MAGIC
# MAGIC 2. Copy and paste the following embedding model into `ai_query()`
# MAGIC
# MAGIC The function `ai_query()` takes `(endpoint, request)` as inputs. 
# MAGIC
# MAGIC
# MAGIC <div class="code-card">
# MAGIC       <div class="code-card-header">
# MAGIC         <span class="code-card-lang">Prompt</span>
# MAGIC         <button class="code-card-copy" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent,t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.classList.add('copied');setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.classList.remove('copied')},2000)}})(this)">
# MAGIC           <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
# MAGIC             <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
# MAGIC           </svg>
# MAGIC           <svg class="check-icon" style="display:none" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
# MAGIC             <polyline points="20 6 9 17 4 12"/>
# MAGIC           </svg>
# MAGIC           <span class="copy-label">Copy</span>
# MAGIC         </button>
# MAGIC       </div>
# MAGIC       <div class="code-card-body"><code>databricks-qwen3-embedding-0-6b</code></div>
# MAGIC     </div>
# MAGIC
# MAGIC Documentation: Please see the following official documentation for `ai_query` [AWS](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_query) | [GCP](https://docs.databricks.com/gcp/en/sql/language-manual/functions/ai_query)

# COMMAND ----------

## TODO

embedding_df = spark.sql(f"""
  SELECT ai_query(
    <FILL_IN>,
    <FILL_IN>
  ) AS embedding
""")

display(embedding_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC embedding_df = spark.sql(f"""
# MAGIC   SELECT ai_query(
# MAGIC     'databricks-qwen3-embedding-0-6b',
# MAGIC     'I want a Victorian house in Hayes Valley for a month-long stay.'
# MAGIC   ) AS embedding
# MAGIC """)
# MAGIC
# MAGIC display(embedding_df)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Save the Embedding
# MAGIC Next, we're going to save this embedding locally for demonstration purposes. 
# MAGIC
# MAGIC Of course, you would probably save your embeddings in a table in Unity Catalog or have them computed on-the-fly and stored in ephemeral memory in your app. 
# MAGIC
# MAGIC Running the next code will create a file called `embedding.txt` in your Workspace folder. After running the next cell, go to the Workspace folder to the left and inspect the file.

# COMMAND ----------

import json

# Collect the embedding from the DataFrame
embedding = embedding_df.collect()[0]["embedding"]

# Save to embedding.txt in this notebook's folder
file_path = "./embedding.txt"

with open(file_path, "w") as f:
    if isinstance(embedding, (list, tuple)):
        f.write(json.dumps(embedding))
    else:
        f.write(str(embedding))

print(f"Embedding saved to: {file_path}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B3. Vector Similarity Search
# MAGIC
# MAGIC #### NOTE: Run the query in Lakebase
# MAGIC
# MAGIC Now let's perform a vector similarity search using the embedding you just saved. 
# MAGIC
# MAGIC There are a few operator classes to be familiar with, whose definitions can be found in the docs ([AWS](https://docs.databricks.com/aws/en/oltp/projects/lakebase-vector#operator-classes) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/oltp/projects/lakebase-vector)). We will be using the `vector_cosine_ops` class, which computes the cosine similarity between embeddings. Its query operator is denoted by `<=>` and returns a **cosine distance** &mdash; lower values mean higher similarity. We alias this as `distance` so you can see the score alongside each result, just like the `score` column in the BM25 query later.
# MAGIC
# MAGIC Fill in the three `FILL_IN` placeholders below:
# MAGIC 1. The **embedding column** (`embedding`) on the left side of `<=>`
# MAGIC
# MAGIC 2. The **1024-dimensional embedding** from `embedding.txt` &mdash; paste it on the right side of `<=>`
# MAGIC
# MAGIC 3. The **table name** (`doc_chunks`)
# MAGIC
# MAGIC 4. The output should return a chunk of text along with a distance score of ~ 0.501
# MAGIC
# MAGIC <div class="code-card" style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;background:#F9F7F4;max-width:100%;margin:12px 0;">
# MAGIC   <div style="display:flex;align-items:center;justify-content:space-between;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;">
# MAGIC     <div style="display:inline-flex;align-items:center;gap:0.45rem;">
# MAGIC       <svg width="20" height="17" viewBox="0 0 20 17" fill="none" style="display:inline-block;vertical-align:middle;"><rect x="1.5" y="1" width="17" height="15" rx="2.5" stroke="#5A6F77" stroke-width="2"/><line x1="1.5" y1="5.5" x2="18.5" y2="5.5" stroke="#5A6F77" stroke-width="1.5"/><path d="M5.5 8.5L8.5 11L5.5 13.5" stroke="#5A6F77" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><line x1="10.5" y1="13" x2="14.5" y2="13" stroke="#5A6F77" stroke-width="1.8" stroke-linecap="round"/></svg>
# MAGIC       <span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">SQL Query</span>
# MAGIC     </div>
# MAGIC     <button style="display:inline-flex;align-items:center;gap:0.3rem;background:none;border:none;color:#5A6F77;cursor:pointer;padding:0.2rem 0.4rem;border-radius:0.25rem;font-size:0.7rem;font-weight:500;" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent,t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.style.color='#2272B4';setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.style.color='#5A6F77'},2000)}})(this)">
# MAGIC       <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width:14px;height:14px" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
# MAGIC       <svg class="check-icon" style="display:none;width:14px;height:14px" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
# MAGIC       <span class="copy-label">Copy</span>
# MAGIC     </button>
# MAGIC   </div>
# MAGIC   <div class="code-card-body" style="margin:0;padding:1rem;overflow-x:auto;font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;font-size:0.82rem;line-height:1.55;color:#1B3139;white-space:pre-wrap;word-break:break-word;">SELECT chunk_text, &amp;lt;FILL_IN&amp;gt; &lt;=&gt; '&amp;lt;FILL_IN&amp;gt;' AS distance
# MAGIC FROM &amp;lt;FILL_IN&amp;gt;
# MAGIC ORDER BY distance;</div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```sql
# MAGIC SELECT chunk_text, embedding <=> 'paste your 1024-dim embedding from embedding.txt' AS distance
# MAGIC FROM doc_chunks
# MAGIC ORDER BY distance;
# MAGIC ```
# MAGIC
# MAGIC **Note:** Paste the full 1024-dimensional embedding vector from your `embedding.txt` file in place of the placeholder above. The `distance` column shows cosine distance &mdash; chunks with the **lowest** values are the most semantically similar to your query.
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B4. Keyword Search with `lakebase_bm25`
# MAGIC
# MAGIC #### NOTE: Run the query in Lakebase
# MAGIC
# MAGIC Now let's perform a keyword search using the `lakebase_bm25` index. 
# MAGIC
# MAGIC The key operator here is `<@>`, whose definition can be found in the docs ([AWS](https://docs.databricks.com/aws/en/oltp/projects/lakebase-text#operators) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/oltp/projects/lakebase-text)). Its signature is `tsvector <@> bm25query_tsvector` and it returns a `double precision` value representing a **negative BM25 score** &mdash; order **ascending** to get the most relevant results first.
# MAGIC
# MAGIC Fill in the four `&lt;FILL_IN&gt;` placeholders below:
# MAGIC
# MAGIC 1. All columns and the BM25 score (`*,`)
# MAGIC
# MAGIC 2. The **tsvector column** (`chunk_tsvector`) on the left side of `<@>`
# MAGIC
# MAGIC 3. The **search term** to match against (e.g. `'Victorian'`)
# MAGIC
# MAGIC 4. The **BM25 index name** (`doc_chunks_ts_idx`)
# MAGIC
# MAGIC 5. The output should return a chunk of text along with a distance score of ~ -0.556.
# MAGIC
# MAGIC <div class="code-card" style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;background:#F9F7F4;max-width:100%;margin:12px 0;">
# MAGIC   <div style="display:flex;align-items:center;justify-content:space-between;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;">
# MAGIC     <div style="display:inline-flex;align-items:center;gap:0.45rem;">
# MAGIC       <svg width="20" height="17" viewBox="0 0 20 17" fill="none" style="display:inline-block;vertical-align:middle;"><rect x="1.5" y="1" width="17" height="15" rx="2.5" stroke="#5A6F77" stroke-width="2"/><line x1="1.5" y1="5.5" x2="18.5" y2="5.5" stroke="#5A6F77" stroke-width="1.5"/><path d="M5.5 8.5L8.5 11L5.5 13.5" stroke="#5A6F77" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><line x1="10.5" y1="13" x2="14.5" y2="13" stroke="#5A6F77" stroke-width="1.8" stroke-linecap="round"/></svg>
# MAGIC       <span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">SQL Query</span>
# MAGIC     </div>
# MAGIC     <button style="display:inline-flex;align-items:center;gap:0.3rem;background:none;border:none;color:#5A6F77;cursor:pointer;padding:0.2rem 0.4rem;border-radius:0.25rem;font-size:0.7rem;font-weight:500;" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent,t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.style.color='#2272B4';setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.style.color='#5A6F77'},2000)}})(this)">
# MAGIC       <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width:14px;height:14px" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
# MAGIC       <svg class="check-icon" style="display:none;width:14px;height:14px" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
# MAGIC       <span class="copy-label">Copy</span>
# MAGIC     </button>
# MAGIC   </div>
# MAGIC   <div class="code-card-body" style="margin:0;padding:1rem;overflow-x:auto;font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;font-size:0.82rem;line-height:1.55;color:#1B3139;white-space:pre-wrap;word-break:break-word;">SELECT &amp;lt;FILL_IN&amp;gt;,
# MAGIC   &amp;lt;FILL_IN&amp;gt; &lt;@&gt; to_bm25query(to_tsvector('english', '&amp;lt;FILL_IN&amp;gt;'), '&amp;lt;FILL_IN&amp;gt;') AS score
# MAGIC FROM doc_chunks
# MAGIC ORDER BY score;</div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```sql
# MAGIC SELECT *,
# MAGIC   chunk_tsvector <@> to_bm25query(to_tsvector('english', 'Victorian'), 'doc_chunks_ts_idx') AS score
# MAGIC FROM doc_chunks
# MAGIC ORDER BY score;
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B5. Hybrid Search
# MAGIC
# MAGIC #### NOTE: Run the query in Lakebase
# MAGIC
# MAGIC Finally, we will perform a hybrid search. 
# MAGIC
# MAGIC It retrieves the top candidates from each search _independently_, then combines them into a single ranking using **Reciprocal Rank Fusion (RRF)**. Results that rank well in either or both searches score higher. You can read more here ([AWS](https://docs.databricks.com/aws/en/oltp/projects/lakebase-search#combine-results-with-hybrid-search) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/oltp/projects/lakebase-search)).
# MAGIC
# MAGIC Using what you learned in B3 and B4, fill in the four `&lt;FILL_IN&gt;` placeholders below:
# MAGIC 1. The **1024-dimensional embedding** from `embedding.txt` &mdash; paste it on the right side of `<=>`
# MAGIC
# MAGIC 2. The **tsvector column** (`chunk_tsvector`) on the left side of `<@>`
# MAGIC
# MAGIC 3. The **search term** (e.g. `'Victorian'`)
# MAGIC
# MAGIC 4. The **BM25 index name** (`doc_chunks_ts_idx`)
# MAGIC
# MAGIC 5. You will see the `rrf_score` column, ranked from best search results to least. 
# MAGIC
# MAGIC <div class="code-card" style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;border:1px solid #DCE0E2;border-radius:8px;overflow:hidden;background:#F9F7F4;max-width:100%;margin:12px 0;">
# MAGIC   <div style="display:flex;align-items:center;justify-content:space-between;padding:0.45rem 0.75rem;background:#EEEDE9;border-bottom:1px solid #DCE0E2;">
# MAGIC     <div style="display:inline-flex;align-items:center;gap:0.45rem;">
# MAGIC       <svg width="20" height="17" viewBox="0 0 20 17" fill="none" style="display:inline-block;vertical-align:middle;"><rect x="1.5" y="1" width="17" height="15" rx="2.5" stroke="#5A6F77" stroke-width="2"/><line x1="1.5" y1="5.5" x2="18.5" y2="5.5" stroke="#5A6F77" stroke-width="1.5"/><path d="M5.5 8.5L8.5 11L5.5 13.5" stroke="#5A6F77" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><line x1="10.5" y1="13" x2="14.5" y2="13" stroke="#5A6F77" stroke-width="1.8" stroke-linecap="round"/></svg>
# MAGIC       <span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A6F77;">SQL Query</span>
# MAGIC     </div>
# MAGIC     <button style="display:inline-flex;align-items:center;gap:0.3rem;background:none;border:none;color:#5A6F77;cursor:pointer;padding:0.2rem 0.4rem;border-radius:0.25rem;font-size:0.7rem;font-weight:500;" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent,t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.style.color='#2272B4';setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.style.color='#5A6F77'},2000)}})(this)">
# MAGIC       <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width:14px;height:14px" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
# MAGIC       <svg class="check-icon" style="display:none;width:14px;height:14px" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
# MAGIC       <span class="copy-label">Copy</span>
# MAGIC     </button>
# MAGIC   </div>
# MAGIC   <div class="code-card-body" style="margin:0;padding:1rem;overflow-x:auto;font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;font-size:0.82rem;line-height:1.55;color:#1B3139;white-space:pre-wrap;word-break:break-word;">WITH vector_ranked AS (
# MAGIC   SELECT chunk_id, RANK() OVER (ORDER BY dist) AS rank
# MAGIC   FROM (
# MAGIC     SELECT chunk_id, embedding &lt;=&gt; '&amp;lt;FILL_IN&amp;gt;' AS dist
# MAGIC     FROM doc_chunks
# MAGIC     ORDER BY dist
# MAGIC     LIMIT 40
# MAGIC   ) v
# MAGIC ),
# MAGIC keyword_ranked AS (
# MAGIC   SELECT chunk_id, RANK() OVER (ORDER BY score) AS rank
# MAGIC   FROM (
# MAGIC     SELECT chunk_id,
# MAGIC       &amp;lt;FILL_IN&amp;gt; &lt;@&gt; to_bm25query(to_tsvector('english', '&amp;lt;FILL_IN&amp;gt;'), '&amp;lt;FILL_IN&amp;gt;') AS score
# MAGIC     FROM doc_chunks
# MAGIC     ORDER BY score
# MAGIC     LIMIT 40
# MAGIC   ) k
# MAGIC )
# MAGIC SELECT d.chunk_id, LEFT(d.chunk_text, 120) AS preview,
# MAGIC   COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS rrf_score
# MAGIC FROM doc_chunks d
# MAGIC LEFT JOIN vector_ranked v ON d.chunk_id = v.chunk_id
# MAGIC LEFT JOIN keyword_ranked k ON d.chunk_id = k.chunk_id
# MAGIC WHERE v.chunk_id IS NOT NULL OR k.chunk_id IS NOT NULL
# MAGIC ORDER BY rrf_score DESC, d.chunk_id
# MAGIC LIMIT 10;</div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none;">
# MAGIC Click to view answer
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```sql
# MAGIC WITH vector_ranked AS (
# MAGIC   SELECT chunk_id, RANK() OVER (ORDER BY dist) AS rank
# MAGIC   FROM (
# MAGIC     SELECT chunk_id, embedding <=> '<paste your 1024-dim embedding from embedding.txt>' AS dist
# MAGIC     FROM doc_chunks
# MAGIC     ORDER BY dist
# MAGIC     LIMIT 40
# MAGIC   ) v
# MAGIC ),
# MAGIC keyword_ranked AS (
# MAGIC   SELECT chunk_id, RANK() OVER (ORDER BY score) AS rank
# MAGIC   FROM (
# MAGIC     SELECT chunk_id,
# MAGIC       chunk_tsvector <@> to_bm25query(to_tsvector('english', 'Victorian'), 'doc_chunks_ts_idx') AS score
# MAGIC     FROM doc_chunks
# MAGIC     ORDER BY score
# MAGIC     LIMIT 40
# MAGIC   ) k
# MAGIC )
# MAGIC SELECT d.chunk_id, LEFT(d.chunk_text, 120) AS preview,
# MAGIC   COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS rrf_score
# MAGIC FROM doc_chunks d
# MAGIC LEFT JOIN vector_ranked v ON d.chunk_id = v.chunk_id
# MAGIC LEFT JOIN keyword_ranked k ON d.chunk_id = k.chunk_id
# MAGIC WHERE v.chunk_id IS NOT NULL OR k.chunk_id IS NOT NULL
# MAGIC ORDER BY rrf_score DESC, d.chunk_id
# MAGIC LIMIT 10;
# MAGIC ```
# MAGIC
# MAGIC **Note:** Paste the full 1024-dimensional embedding vector from your `embedding.txt` file in place of the placeholder above.
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>