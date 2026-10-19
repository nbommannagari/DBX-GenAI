# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Demo Overview and Learning Objectives
# MAGIC %md
# MAGIC # Demo - Building Genie Agents Programmatically
# MAGIC #### Duration: 20 minutes
# MAGIC ## Overview
# MAGIC A **Genie Agent** is a Databricks-managed, single agent that answers natural-language questions about governed data by *calling a tool* — text-to-SQL over Unity Catalog — and can be equipped with the same trusted UC functions you already registered. This notebook looks at a Genie Agent through the lens of the single-agent concepts you already know — **tool calling**, **grounded responses with sources**, and a **human feedback loop** and, most importantly, for an agent developer, how to **create and drive that agent entirely from code** (notebook- and app-driven) rather than from the UI.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC - Explain how a Genie Agent maps onto the single-agent pattern: one agent, tool calling, grounded answers, and a human-in-the-loop feedback loop
# MAGIC - Create a Genie Agent programmatically and **attach a recycled Unity Catalog function** from the previous demo as a trusted tool
# MAGIC - Query the agent from a notebook using the **Genie Conversation API** in the Databricks SDK
# MAGIC - Inspect an agent response's **sources** — the generated SQL and the returned rows
# MAGIC - Continue a conversation with **multi-turn follow-ups** to see the human-in-the-loop feedback loop
# MAGIC - (Optional) Stream an agent's reasoning and tool calls with the **Agent Mode API**

# COMMAND ----------

# DBTITLE 1,Required - Select a Compute Environment
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

# MAGIC %run ./Includes/Classroom-Setup-4

# COMMAND ----------

# DBTITLE 1,A3 - Initialize the Workspace Client
# MAGIC %md-sandbox
# MAGIC ### A3. Initialize the Workspace Client
# MAGIC
# MAGIC Create an authenticated `WorkspaceClient`. This single client gives us everything we need in this notebook: the Genie Conversation API (`w.genie`) to create conversations and read responses, and access to SQL warehouses. Authentication is handled automatically from the notebook context — no manual token handling required.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
workspace_host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"

print(f"Host    : {workspace_host}")

# COMMAND ----------

# DBTITLE 1,Section B - Create a Genie Agent and Connect Tools
# MAGIC %md-sandbox
# MAGIC ## B. Create a Genie Agent and Connect Tools
# MAGIC
# MAGIC We will now create a Genie Agent **from code** and attach the recycled `avg_neigh_price` Unity Catalog function as a trusted tool. Creating the agent programmatically (rather than clicking through the UI) is exactly what you would do to version an agent in source control or provision it from an app or CI/CD pipeline.
# MAGIC
# MAGIC A Genie Agent is defined by a `serialized_space` JSON document plus a warehouse to run its SQL. The `serialized_space` is where you encode the agent's curated context: the tables it can see, sample questions, example SQL, text instructions, and — the key piece for us — the **SQL functions** it is allowed to call.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Learn More</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC For the full Genie Agent authoring surface and the Conversation API, see the Databricks documentation (<a href="https://docs.databricks.com/aws/en/genie/" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/genie/" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/genie/" style="color: #1976d2; text-decoration: underline;">GCP</a>).
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,B1 - Resolve a SQL Warehouse
# MAGIC %md-sandbox
# MAGIC ### B1. Resolve a SQL Warehouse
# MAGIC
# MAGIC A Genie Agent executes its generated SQL on a **SQL warehouse**, so we need one to attach at creation time. The next cell picks a warehouse available to you and captures its ID. As a part of the classroom setup for this notebook, a warehouse called `shared_warehouse` has been provided.

# COMMAND ----------

# Get the "shared_warehouse" SQL warehouse
warehouses = list(w.warehouses.list())
match = [wh for wh in warehouses if wh.name == "shared_warehouse"]
if not match:
    raise RuntimeError("No SQL warehouse named 'shared_warehouse' found. Please create it and re-run.")

warehouse = match[0]
warehouse_id = warehouse.id
print(f"Using warehouse: {warehouse.name}  (id={warehouse_id}, state={warehouse.state})")

# COMMAND ----------

# DBTITLE 1,B2 - Build the Agent Definition
# MAGIC %md-sandbox
# MAGIC ### B2. Build the Agent Definition (`serialized_space`)
# MAGIC
# MAGIC Here we assemble the agent's curated context. Note the two things that make this behave like a governed single agent:
# MAGIC
# MAGIC - **`data_sources.tables`** — the one table the agent may query (`sf_airbnb_listings`).
# MAGIC - **`instructions.sql_functions`** — the **recycled `avg_neigh_price` UC function** attached as a trusted tool. When a question is best answered by that function, Genie can call it instead of writing the aggregation from scratch.
# MAGIC
# MAGIC We also add a sample question, one example SQL that teaches a common query shape, and a short text instruction. Every id-keyed entry needs a unique 32-character hex `id`, and each list must be sorted by that `id`.

# COMMAND ----------

# DBTITLE 1,Cell 12
# MAGIC %md
# MAGIC 1. Define a `hex_id()` helper function to generate the unique 32-char hexadecimal IDs required by the `serialized_space` schema. This will be used to build the curated-context payloads:
# MAGIC     - `sample_questions`: example questions shown to users in the Genie Agent UI
# MAGIC     - `example_sqls`: example SQL queries the Genie Agent will leverage as few-shot guidance
# MAGIC     - `text_instructions`: natural-language instructions that steer the agent's behavior
# MAGIC     - `sql_functions`: UC SQL functions equipped to the Genie Agent as trusted tools

# COMMAND ----------

import json, uuid

def hex_id() -> str:
    """32-char lowercase hex id, as required by the serialized_space schema."""
    return uuid.uuid4().hex

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Set the `source_table` and `avg_price_fn` fully-qualified names

# COMMAND ----------

source_table = f"{catalog_name}.{schema_name}.sf_airbnb_listings"
avg_price_fn = f"{catalog_name}.{schema_name}.avg_neigh_price"

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Build the curated-context payloads

# COMMAND ----------

sample_questions = sorted(
    [{"id": hex_id(), "question": ["What is the average price in the Mission neighborhood?"]}],
    key=lambda x: x["id"],
)
example_sqls = sorted(
    [{
        "id": hex_id(),
        "question": ["What are the five most expensive neighborhoods by average price?"],
        "sql": [
            f"SELECT neighbourhood_cleansed, AVG(price) AS avg_price "
            f"FROM {source_table} "
            f"WHERE price IS NOT NULL "
            f"GROUP BY neighbourhood_cleansed ORDER BY avg_price DESC LIMIT 5"
        ],
    }],
    key=lambda x: x["id"],
)
text_instructions = [{
    "id": hex_id(),
    "content": [
        "This agent answers questions about San Francisco Airbnb listings.",
        "Round monetary values to 2 decimals and always state the neighborhood used.",
    ],
}]

sql_functions = sorted(
    [{"id": hex_id(), "identifier": avg_price_fn}],
    key=lambda x: x["id"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Assemble everything into the `serialized_space` JSON document that defines the agent

# COMMAND ----------

serialized_space = json.dumps({
    "version": 2,
    "config": {"sample_questions": sample_questions},
    "data_sources": {"tables": [{"identifier": source_table}]},
    "instructions": {
        "example_question_sqls": example_sqls,
        "text_instructions": text_instructions,
        "sql_functions": sql_functions,
    },
})

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Print a summary of what was created

# COMMAND ----------

print("Attached table   :", source_table)
print("Attached function:", avg_price_fn)
print(json.dumps(json.loads(serialized_space), indent=2))

# COMMAND ----------

# DBTITLE 1,B3 - Create the Genie Agent
# MAGIC %md-sandbox
# MAGIC ### B3. Create the Genie Agent
# MAGIC
# MAGIC With the definition and warehouse ready, we create the agent via the Genie Spaces REST endpoint (`POST /api/2.0/genie/spaces`). We authenticate with the notebook's own context token, place the agent in the user's workspace folder, and capture the returned `space_id` — the handle we use to query it for the rest of the notebook.

# COMMAND ----------

# DBTITLE 1,B3 Step 1
# MAGIC %md
# MAGIC 1. Import `requests` and resolve the API URL, token, and authorization headers from the notebook context — no manual token handling required

# COMMAND ----------

# DBTITLE 1,Cell 23
import requests

# Resolve host + token from the notebook context
notebook_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
api_url = notebook_ctx.apiUrl().get()
api_token = notebook_ctx.apiToken().get()
headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

# COMMAND ----------

# DBTITLE 1,B3 Step 2
# MAGIC %md
# MAGIC 2. Build the creation payload with the agent's title, description, parent workspace path, warehouse ID, and the `serialized_space` we assembled in B2

# COMMAND ----------

# DBTITLE 1,B3 Step 2 Code
create_payload = {
    "title": f"{username.split('@')[0]} - SF Airbnb Genie Agent",
    "description": "Managed agent answering questions about San Francisco Airbnb listings.",
    "parent_path": f"/Workspace/Users/{username}",
    "warehouse_id": warehouse_id,
    "serialized_space": serialized_space,
}

# COMMAND ----------

# DBTITLE 1,B3 Step 3
# MAGIC %md
# MAGIC 3. POST to the Genie Spaces REST endpoint (`/api/2.0/genie/spaces`) and capture the returned `space_id` — the handle we use to query the agent for the rest of the notebook

# COMMAND ----------

# DBTITLE 1,B3 Step 3 Code
resp = requests.post(f"{api_url}/api/2.0/genie/spaces", headers=headers, json=create_payload)
resp.raise_for_status()
space = resp.json()
space_id = space.get("space_id") or space.get("id")

# COMMAND ----------

# DBTITLE 1,B3 Step 4
# MAGIC %md
# MAGIC 4. Print confirmation and a direct link to the agent in the Genie UI

# COMMAND ----------

# DBTITLE 1,B3 Step 4 Code
print(f"Genie Agent created. space_id = {space_id}")
print(f"Open in UI: {workspace_host}/genie/rooms/{space_id}")

# COMMAND ----------

# DBTITLE 1,Section C - Query the Agent Programmatically
# MAGIC %md-sandbox
# MAGIC ## C. Query the Agent Programmatically
# MAGIC
# MAGIC This is the core of the demo for an agent developer: driving the managed agent from code with the **Genie Conversation API**. Everything the UI does — start a conversation, get an answer, see the SQL and rows — is available through `w.genie`, so you can embed a Genie Agent in an application, a notebook workflow, or a larger multi-agent system.
# MAGIC
# MAGIC A message moves through states (`SUBMITTED` &#8594; `FILTERING_CONTEXT` &#8594; `ASKING_AI` &#8594; `EXECUTING_QUERY` &#8594; `COMPLETED`). The `_and_wait` helpers below poll for us and return once the message reaches a terminal state.

# COMMAND ----------

# DBTITLE 1,C1 - Ask a Question
# MAGIC %md-sandbox
# MAGIC ### C1. Ask a Question
# MAGIC
# MAGIC We start a new conversation with a question that the agent should answer using the attached `avg_neigh_price` function. `start_conversation_and_wait` blocks until the agent finishes, then we pull the natural-language answer out of the message attachments.

# COMMAND ----------

def print_answer(message):
    """Print the natural-language text from a Genie message's attachments."""
    for att in (message.attachments or []):
        text = getattr(att, "text", None)
        if text and getattr(text, "content", None):
            print(text.content)

question = "What is the average price in the Mission neighborhood?"
print(f"Q: {question}\n")

message = w.genie.start_conversation_and_wait(space_id, question)

print(f"Status: {message.status}\n")
print_answer(message)

# Save IDs so we can inspect sources and ask a follow-up later
conversation_id = message.conversation_id
message_id = getattr(message, "id", None) or getattr(message, "message_id", None)

# COMMAND ----------

# DBTITLE 1,C2 - View the Sources
# MAGIC %md-sandbox
# MAGIC ### C2. View the Sources (Generated SQL and Rows)
# MAGIC
# MAGIC A grounded answer is only as trustworthy as its sources. Each query attachment carries the **SQL the agent generated and executed** — this is the agent's tool call — and we can fetch the **rows** that SQL returned. Surfacing both is how you build user trust and how you debug an agent's reasoning.

# COMMAND ----------

def show_sources(message):
    """Show the generated SQL (the tool call) and the returned rows (the source)."""
    for att in (message.attachments or []):
        query = getattr(att, "query", None)
        if not query:
            continue
        print("Description:", getattr(query, "description", None))
        print("\nGenerated SQL (the agent's tool call):")
        print(getattr(query, "query", None))

        attachment_id = getattr(att, "attachment_id", None)
        result = w.genie.get_message_attachment_query_result(
            space_id, conversation_id, message_id, attachment_id
        )
        sr = getattr(result, "statement_response", None)
        if sr and getattr(sr, "result", None) and getattr(sr, "manifest", None):
            cols = [c.name for c in sr.manifest.schema.columns]
            rows = sr.result.data_array or []
            print("\nRows returned (the source):")
            print(cols)
            for row in rows[:10]:
                print(row)

show_sources(message)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC The generated SQL is the Genie Agent's <strong>tool call</strong>, and the returned rows are the <strong>source</strong> grounding its answer. Because the agent runs on a SQL warehouse over Unity Catalog, every one of those calls is governed and auditable — the same permissions that protect your tables protect the agent's access to them.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Section D - Human-in-the-Loop Follow-ups
# MAGIC %md-sandbox
# MAGIC ## D. Human-in-the-Loop: Multi-Turn Follow-ups
# MAGIC
# MAGIC Single agents rarely answer in one shot — users refine. Because a conversation is **stateful**, a follow-up question keeps the prior context, so the user can steer the agent conversationally. This is the human-in-the-loop feedback loop in action.

# COMMAND ----------

# DBTITLE 1,D1 - Ask a Follow-up
# MAGIC %md-sandbox
# MAGIC ### D1. Ask a Follow-up in the Same Conversation
# MAGIC
# MAGIC We reuse `conversation_id` and call `create_message_and_wait`. Notice the follow-up refers back to the previous answer ("compare that to...") without restating the full context — the agent remembers.

# COMMAND ----------

follow_up = "How does that compare to the Noe Valley neighborhood?"
print(f"Q: {follow_up}\n")

follow_message = w.genie.create_message_and_wait(space_id, conversation_id, follow_up)

print(f"Status: {follow_message.status}\n")
print_answer(follow_message)
message_id = getattr(follow_message, "id", None) or getattr(follow_message, "message_id", None)
show_sources(follow_message)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC In the Genie UI, users also give <strong>thumbs-up / thumbs-down feedback</strong> on answers, and curators promote correct queries to trusted "instructions." That review signal — combined with these multi-turn conversations — is the feedback loop that improves a managed agent over time, without you retraining anything.
# MAGIC </p>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# DBTITLE 1,Section E - Optional - Agent Mode API
# MAGIC %md-sandbox
# MAGIC ## E. (Optional) Agent Mode API: Stream Reasoning and Tool Calls
# MAGIC
# MAGIC The Conversation API returns the final answer plus its SQL. If you want to watch the agent **think** — its reasoning plan, each SQL tool call, and the synthesized answer — Databricks offers the **Agent Mode API**, which streams these as events. This is the fullest expression of the "single agent with tool calling" idea.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Beta &amp; preview-gated</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC The Agent Mode API is in <strong>Beta</strong> and requires the preview <em>"Agent Mode APIs for Genie Agents"</em> to be enrolled for your workspace, plus the <code>databricks-openai</code> package. The cell below is wrapped so it degrades gracefully if either is missing — it will print guidance rather than fail. See the Agent Mode API reference (<a href="https://docs.databricks.com/aws/en/genie/" style="color: #1976d2; text-decoration: underline;">docs</a>).
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# Optional: stream the agent's reasoning and tool calls (Beta; requires preview + databricks-openai)
try:
    from databricks_openai import DatabricksOpenAI

    client = DatabricksOpenAI(workspace_client=w)
    client.base_url = f"{workspace_host}/api/2.0/genie/agents/{space_id}"

    stream = client.responses.create(
        model="genie-agent",
        input=[{"type": "message", "role": "user",
                "content": [{"type": "input_text",
                             "text": "Which neighborhood offers the best value, and why?"}]}],
        stream=True,
    )

    for event in stream:
        etype = getattr(event, "type", "")
        if etype == "reasoning":
            print("[reasoning]", getattr(event, "text", ""))
        elif etype == "response.output_item.done" and getattr(event.item, "type", "") == "function_call":
            import json as _json
            sql = _json.loads(event.item.arguments).get("sql")
            if sql:
                print("[tool call / SQL]\n", sql)
        elif etype == "response.completed":
            for item in event.response.output:
                if getattr(item, "type", "") == "message":
                    print("\n[final answer]\n", item.content[0].text)

except ImportError:
    print("databricks-openai is not installed. To try Agent Mode, run: %pip install databricks-openai")
except Exception as e:
    print(f"Agent Mode API unavailable (likely preview not enrolled): {type(e).__name__}: {e}")

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC You built and drove a **managed single agent** entirely from code. Along the way you saw how a Genie Agent maps onto the single-agent pattern you already know:
# MAGIC
# MAGIC - **Tool calling** — its text-to-SQL engine plus the recycled `avg_neigh_price` UC function you attached as a trusted tool
# MAGIC - **Grounded responses with sources** — every answer carries the generated SQL (the tool call) and the rows it returned (the source), all governed by Unity Catalog
# MAGIC - **A human feedback loop** — stateful, multi-turn conversations and UI feedback that curators use to improve the agent
# MAGIC - **Programmatic control** — creating the agent and querying it with the Genie Conversation API in the Databricks SDK, ready to embed in an app, workflow, or larger system
# MAGIC
# MAGIC The key takeaway for an agent developer: a Genie Agent gives you a governed, single-agent RAG-over-structured-data capability without writing the orchestration loop — and because it is fully API-driven, it drops into the same applications and multi-agent systems you build with code-first agents.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>