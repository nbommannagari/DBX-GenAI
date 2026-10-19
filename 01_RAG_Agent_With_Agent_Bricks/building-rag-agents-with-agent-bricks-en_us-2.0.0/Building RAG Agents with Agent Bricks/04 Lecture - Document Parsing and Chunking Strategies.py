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
# MAGIC # Lecture - Document Parsing and Chunking Strategies for AI Search
# MAGIC #### Duration: 30 minutes
# MAGIC ## Overview
# MAGIC Before a retrieval agent can answer questions about your documents, those documents must be transformed from raw files into searchable, structured data. This lecture covers the key Databricks AI Functions that power this pipeline: `ai_parse_document` for extracting text from PDFs, `ai_classify` and `ai_extract` for structuring content, and `ai_prep_search` for chunking text for AI Search or leveraging reverse ETL for Lakebase Search integration. We'll also cover how chunks flow into an AI Search index: what happens when you create one, how it stays in sync, and why the text you embed differs from the text you retrieve.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lecture, you will be able to:
# MAGIC 1. Describe how `ai_parse_document` extracts structured content from PDFs
# MAGIC 2. Explain the role of `ai_classify` and `ai_extract` in document processing
# MAGIC 3. Use `ai_prep_search` to chunk text for AI Search
# MAGIC 4. Compare different chunking strategies and their trade-offs
# MAGIC 5. Explain what happens when an AI Search index is created and how it stays in sync
# MAGIC 6. Distinguish between `chunk_to_embed` and `chunk_to_retrieve` and why they differ

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. The Document Processing Pipeline

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ### A1. From Raw Files to Searchable Data
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 855 245" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Document Processing Pipeline: parallel branches from parsed document</title>
# MAGIC   <defs>
# MAGIC     <marker id="a1c-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC
# MAGIC   <!-- Raw PDF -->
# MAGIC   <rect x="10" y="75" width="125" height="95" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="72" y="112" text-anchor="middle" font-size="17" font-weight="600" fill="#0b2026">Raw PDF</text>
# MAGIC   <text x="72" y="132" text-anchor="middle" font-size="15" fill="#618794">UC Volume</text>
# MAGIC
# MAGIC   <!-- Arrow to parse -->
# MAGIC   <path d="M135 122 L178 122" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#a1c-arrow)"/>
# MAGIC
# MAGIC   <!-- ai_parse_document -->
# MAGIC   <rect x="183" y="58" width="180" height="128" rx="12" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="273" y="100" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">ai_parse_document</text>
# MAGIC   <text x="273" y="120" text-anchor="middle" font-size="15" fill="#618794">Extract text + layout</text>
# MAGIC   <text x="273" y="140" text-anchor="middle" font-size="15" fill="#618794">-> VARIANT output</text>
# MAGIC
# MAGIC   <!-- Fan-out arrows (parallel from parsed doc) -->
# MAGIC   <path d="M363 90 L410 41" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#a1c-arrow)"/>
# MAGIC   <path d="M363 122 L410 121" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#a1c-arrow)"/>
# MAGIC   <path d="M363 154 L410 201" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#a1c-arrow)"/>
# MAGIC
# MAGIC   <!-- ai_prep_search (red) -->
# MAGIC   <rect x="418" y="12" width="170" height="58" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="503" y="36" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">ai_prep_search</text>
# MAGIC   <text x="503" y="56" text-anchor="middle" font-size="15" fill="#618794">Chunk for VS</text>
# MAGIC
# MAGIC   <!-- ai_classify (green) -->
# MAGIC   <rect x="418" y="92" width="170" height="58" rx="8" fill="#4299E0" fill-opacity="0.12" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="503" y="116" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">ai_classify</text>
# MAGIC   <text x="503" y="136" text-anchor="middle" font-size="15" fill="#618794">Categorize content</text>
# MAGIC
# MAGIC   <!-- ai_extract (green) -->
# MAGIC   <rect x="418" y="172" width="170" height="58" rx="8" fill="#4299E0" fill-opacity="0.12" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="503" y="196" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">ai_extract</text>
# MAGIC   <text x="503" y="216" text-anchor="middle" font-size="15" fill="#618794">Pull structured fields</text>
# MAGIC
# MAGIC   <!-- Converge arrows (join into enriched chunks) -->
# MAGIC   <path d="M588 41 L648 98" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#a1c-arrow)"/>
# MAGIC   <path d="M588 121 L648 121" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#a1c-arrow)"/>
# MAGIC   <path d="M588 201 L648 144" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#a1c-arrow)"/>
# MAGIC
# MAGIC   <!-- Enriched Chunk Table (output) -->
# MAGIC   <rect x="653" y="56" width="190" height="132" rx="8" fill="#ffffff" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="748" y="82" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Enriched Chunks</text>
# MAGIC   <text x="748" y="106" text-anchor="middle" font-size="13" fill="#FF5F46">chunks</text>
# MAGIC   <text x="748" y="124" text-anchor="middle" font-size="13" fill="#4299E0">+ category labels</text>
# MAGIC   <text x="748" y="142" text-anchor="middle" font-size="13" fill="#4299E0">+ extracted fields</text>
# MAGIC   <text x="748" y="170" text-anchor="middle" font-size="14" fill="#618794">-> AI Search</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC How does the document processing pipeline work?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p style="margin-top: 10px;">RAG systems are only as good as the data they retrieve. Before documents can power a RAG agent, they must be processed through a pipeline that extracts, classifies, and chunks the content. Databricks Knowledge Assistant can ingest UC Volumes and tables directly. It runs its own parsing and chunking pipeline under the hood, so you don't need to wire these AI Functions yourself.</p>
# MAGIC
# MAGIC <p>Databricks provides a set of AI Functions (<a href="https://docs.databricks.com/aws/en/large-language-models/ai-functions/" style="color: #1B5162; font-weight: 600;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/large-language-models/ai-functions/" style="color: #1B5162; font-weight: 600;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/large-language-models/ai-functions/" style="color: #1B5162; font-weight: 600;">GCP</a>), SQL-callable functions powered by foundation models, that handle each stage. As an example workflow, we might have:</p>
# MAGIC
# MAGIC <ul>
# MAGIC <li><code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">ai_parse_document</code> produce a <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">VARIANT</code> output</li>
# MAGIC <li><code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">ai_prep_search</code>, <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">ai_classify</code>, and <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">ai_extract</code> run <strong>in parallel</strong> off that parsed document</li>
# MAGIC <li>Join the classifications and extracted fields onto the chunk table as extra metadata columns for filtering or ranking in AI Search</li>
# MAGIC </ul>
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Parsing Documents with `ai_parse_document`

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ### B1. How It Works
# MAGIC
# MAGIC `ai_parse_document` ([AWS](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_parse_document) | [GCP](https://docs.databricks.com/gcp/en/sql/language-manual/functions/ai_parse_document)) invokes state-of-the-art generative AI models to extract structured content from PDFs and images. It returns a structured JSON object (VARIANT type) with layout-aware text extraction.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 700 200" role="img" style="font-family: sans-serif;">
# MAGIC   <title>ai_parse_document: from raw PDF to structured output</title>
# MAGIC   <defs>
# MAGIC     <marker id="pd-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- Input: Raw PDF -->
# MAGIC   <rect x="15" y="30" width="130" height="130" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="80" y="56" text-anchor="middle" font-size="17" font-weight="600" fill="#0b2026">Raw PDF</text>
# MAGIC   <rect x="30" y="68" width="100" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <rect x="30" y="84" width="80" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <rect x="30" y="100" width="60" height="24" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="60" y="116" text-anchor="middle" font-size="14" fill="#2272B4">table</text>
# MAGIC   <rect x="96" y="100" width="34" height="24" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="113" y="116" text-anchor="middle" font-size="14" fill="#2272B4">img</text>
# MAGIC   <rect x="30" y="130" width="100" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <!-- Arrow -->
# MAGIC   <path d="M145 95 L210 95" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#pd-arrow)"/>
# MAGIC   <!-- Function box -->
# MAGIC   <rect x="215" y="32" width="220" height="128" rx="12" fill="#4299E0" fill-opacity="0.10" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="325" y="72" text-anchor="middle" font-size="18" font-weight="700" fill="#0b2026">ai_parse_document</text>
# MAGIC   <text x="325" y="96" text-anchor="middle" font-size="15" fill="#618794">OCR + layout analysis,</text>
# MAGIC   <text x="325" y="116" text-anchor="middle" font-size="15" fill="#618794">figure descriptions,</text>
# MAGIC   <text x="325" y="136" text-anchor="middle" font-size="15" fill="#618794">bounding boxes</text>
# MAGIC   <!-- Arrow -->
# MAGIC   <path d="M435 95 L490 95" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#pd-arrow)"/>
# MAGIC   <!-- Output -->
# MAGIC   <rect x="495" y="30" width="190" height="140" rx="8" fill="#ffffff" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="590" y="55" text-anchor="middle" font-size="17" font-weight="600" fill="#4299E0">Structured JSON</text>
# MAGIC   <text x="514" y="78" font-size="14" fill="#0b2026" font-family="monospace">document:</text>
# MAGIC   <text x="526" y="94" font-size="14" fill="#618794" font-family="monospace">  elements: [...]</text>
# MAGIC   <text x="526" y="110" font-size="14" fill="#618794" font-family="monospace">  pages: [...]</text>
# MAGIC   <text x="514" y="130" font-size="14" fill="#0b2026" font-family="monospace">  error_status: [...]</text>
# MAGIC   <text x="514" y="150" font-size="14" fill="#0b2026" font-family="monospace">metadata: {...}</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC What are the four stages of a Knowledge Assistant pipeline?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p style="margin-top: 10px;">Databricks AI Functions transform raw PDFs into searchable data in three stages: <li><strong>Parse (ai_parse_document)</strong>  extracts text and layout as structured JSON</li>
# MAGIC <li><strong>Classify + Extract (ai_classify, ai_extract)</strong>  categorize and pull structured fields</li>
# MAGIC <li><strong>Chunk (ai_prep_search)</strong> splits text into semantic chunks for AI Search.</li>
# MAGIC <li><code>version</code> is an option flag that tells <code>ai_parse_document</code> which schema/output version to use.
# MAGIC <li><strong>v2 ('2.0')</strong> is the recommended version: it gives you the newer output format (HTML tables, improved layout, etc.) and is what v2 <code>ai_extract</code>/<code>ai_classify</code> examples are tuned for.
# MAGIC </li></p>
# MAGIC
# MAGIC <pre style="background: #FFFFFF; color: #0b2026; padding: 12px 14px; border: 1px solid #D8D4CA; border-radius: 6px; margin: 10px 0; font-size: 11pt; line-height: 1.5; overflow-x: auto;"><code>SELECT
# MAGIC   path,
# MAGIC   ai_parse_document(content, MAP('version', '2.0')) AS parsed_content
# MAGIC FROM READ_FILES('/Volumes/...', format => 'binaryFile')
# MAGIC </code></pre>
# MAGIC <p style="margin: 6px 0 0 0;">The output includes <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">document:elements</code> (individual content blocks) and <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">document:pages</code> (page-level breakdown), giving you flexible access to the parsed content.</p>
# MAGIC </li>
# MAGIC </ol>
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Classifying and Extracting with AI Functions
# MAGIC Next, we will look at some examples of how to use `ai_classify()` and `ai_extract()`. Parsing the raw PDF into a structured JSON is straightforward as indicated above. However, we now have to use the table-valude function `variant_explode` ([AWS](https://docs.databricks.com/aws/en/sql/language-manual/functions/variant_explode) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/variant_explode) | [GCP](https://docs.databricks.com/gcp/en/sql/language-manual/functions/variant_explode)) to un-nest our parsed document.
# MAGIC
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC How does variant_explode work?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p style="margin-top: 10px;">The <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">variant_explode</code> table-valued function turns a <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">VARIANT</code> array or object into a set of rows with three columns:</p>
# MAGIC
# MAGIC <ul>
# MAGIC <li><code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">pos INT</code> &mdash; the position of the element within the array or object</li>
# MAGIC <li><code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">key STRING</code> &mdash; the field name (objects only; <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">NULL</code> for arrays)</li>
# MAGIC <li><code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">value VARIANT</code> &mdash; the field value or array element</li>
# MAGIC </ul>
# MAGIC
# MAGIC <p>Behavior depends on the input shape:</p>
# MAGIC <ul>
# MAGIC <li><strong>VARIANT object</strong> &rarr; <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">key</code> and <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">value</code> are the object's keys and values.</li>
# MAGIC <li><strong>VARIANT array</strong> &rarr; <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">key</code> is always <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">NULL</code>; <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">value</code> holds each element.</li>
# MAGIC <li><strong>NULL or non-VARIANT-array/object input</strong> &rarr; <strong>no rows are produced</strong>. To preserve the row with <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">NULL</code> values instead, use variant_explode_outer (<a href="https://docs.databricks.com/aws/en/sql/language-manual/functions/variant_explode_outer" style="color: #1B5162; font-weight: 600;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/variant_explode_outer" style="color: #1B5162; font-weight: 600;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/sql/language-manual/functions/variant_explode_outer" style="color: #1B5162; font-weight: 600;">GCP</a>).</li>
# MAGIC </ul>
# MAGIC
# MAGIC <pre style="background: #FFFFFF; color: #0b2026; padding: 12px 14px; border: 1px solid #D8D4CA; border-radius: 6px; margin: 10px 0; font-size: 11pt; line-height: 1.5; overflow-x: auto;"><code>SELECT pos, key, value
# MAGIC FROM variant_explode(parse_json('{"name": "Ada", "age": 36}'));
# MAGIC
# MAGIC -- pos | key  | value
# MAGIC -- ----+------+--------
# MAGIC --  0  | name | "Ada"
# MAGIC --  1  | age  | 36
# MAGIC </code></pre>
# MAGIC
# MAGIC <p style="margin: 6px 0 0 0;">Use <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">variant_explode</code> as a lateral source (e.g., in a <code style="background: #EEEDE9; padding: 1px 6px; border-radius: 3px; font-size: 11pt;">LATERAL VIEW</code> or comma join) to flatten semi-structured VARIANT columns into queryable rows.</p>
# MAGIC
# MAGIC </div>
# MAGIC </details>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ### C1. Categorize Content: `ai_classify`
# MAGIC
# MAGIC `ai_classify` ([AWS](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_classify) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_classify) | [GCP](https://docs.databricks.com/gcp/en/sql/language-manual/functions/ai_classify)) assigns a label to text from a set of categories you define. It's useful for routing documents through different processing paths.
# MAGIC - `ai_parsed_document` produces a `VARIANT` column called `parsed_content`. Passing `parsed_content` lets ai_classify see the full, parsed document instead of just a single text string.
# MAGIC - `["invoice", "receipt", "contract", "report"]` is the set of categories you want the model to choose from.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 720 165" role="img" style="font-family: sans-serif;">
# MAGIC   <title>ai_classify: assigns a category label to text</title>
# MAGIC   <defs>
# MAGIC     <marker id="cl-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- Input -->
# MAGIC   <rect x="15" y="18" width="170" height="126" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="100" y="42" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Parsed Text</text>
# MAGIC   <!-- Gray bar above -->
# MAGIC   <rect x="25" y="50" width="150" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <!-- Chip 1 -->
# MAGIC   <rect x="25" y="66" width="150" height="22" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="100" y="81" text-anchor="middle" font-size="11" fill="#2272B4">"Total due: $4,092..."</text>
# MAGIC   <!-- Chip 2 -->
# MAGIC   <rect x="25" y="94" width="150" height="22" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="100" y="109" text-anchor="middle" font-size="11" fill="#2272B4">"Payment terms: Net 30"</text>
# MAGIC   <!-- Gray bar below -->
# MAGIC   <rect x="25" y="122" width="150" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <path d="M185 81 L240 81" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#cl-arrow)"/>
# MAGIC   <!-- Function -->
# MAGIC   <rect x="245" y="37" width="200" height="80" rx="12" fill="#4299E0" fill-opacity="0.10" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="345" y="67" text-anchor="middle" font-size="18" font-weight="700" fill="#0b2026">ai_classify</text>
# MAGIC   <text x="345" y="87" text-anchor="middle" font-size="14" fill="#618794">parsed_content,</text>
# MAGIC   <text x="345" y="101" text-anchor="middle" font-size="14" fill="#618794">["invoice",...]</text>
# MAGIC   <path d="M445 77 L505 77" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#cl-arrow)"/>
# MAGIC   <!-- Output -->
# MAGIC   <rect x="510" y="47" width="170" height="56" rx="8" fill="#ffffff" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="595" y="71" text-anchor="middle" font-size="17" font-weight="600" fill="#4299E0">invoice</text>
# MAGIC   <text x="595" y="89" text-anchor="middle" font-size="15" fill="#618794">Single label returned</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC Sample Code:
# MAGIC <pre style="background: #FFFFFF; color: #0b2026; padding: 12px 14px; border: 1px solid #D8D4CA; border-radius: 6px; margin: 10px 0; font-size: 11pt; line-height: 1.5; overflow-x: auto;"><code>WITH elements AS (
# MAGIC   &amp;lt;read_file_logic&amp;gt; t,
# MAGIC   LATERAL variant_explode(parsed:document:elements) AS element
# MAGIC   &amp;lt;additional_WHERE_logic&amp;gt;
# MAGIC ),
# MAGIC classified AS (
# MAGIC   SELECT
# MAGIC     content,
# MAGIC     ai_classify(
# MAGIC       content,
# MAGIC       '["property_description","pricing_information","location_details","amenities","host_information","other"]'
# MAGIC     ) AS cls
# MAGIC   FROM elements
# MAGIC )
# MAGIC SELECT
# MAGIC   content,
# MAGIC   cls:response[0] AS category -- Parse the results from ai_classify()
# MAGIC FROM classified</pre>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ### C2. Pull Structured Fields: `ai_extract`
# MAGIC
# MAGIC `ai_extract` ([AWS](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_extract) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_extract) | [GCP](https://docs.databricks.com/gcp/en/sql/language-manual/functions/ai_extract)) extracts specific fields from unstructured text. You define what to extract, and the AI model finds the values.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 720 180" role="img" style="font-family: sans-serif;">
# MAGIC   <title>ai_extract: pulls structured fields from text</title>
# MAGIC   <defs>
# MAGIC     <marker id="ex-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- Input -->
# MAGIC   <rect x="15" y="15" width="170" height="148" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="100" y="40" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Parsed Text</text>
# MAGIC   <!-- Gray bar above -->
# MAGIC   <rect x="25" y="48" width="150" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <!-- Chip 1 -->
# MAGIC   <rect x="25" y="63" width="150" height="22" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="100" y="78" text-anchor="middle" font-size="11" fill="#2272B4">"From: Apex Roofing"</text>
# MAGIC   <!-- Chip 2 -->
# MAGIC   <rect x="25" y="89" width="150" height="22" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="100" y="104" text-anchor="middle" font-size="11" fill="#2272B4">"Date: Jan 15, 2026"</text>
# MAGIC   <!-- Chip 3 -->
# MAGIC   <rect x="25" y="115" width="150" height="22" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="100" y="130" text-anchor="middle" font-size="11" fill="#2272B4">"Total: $4,092.64"</text>
# MAGIC   <!-- Gray bar below -->
# MAGIC   <rect x="25" y="142" width="150" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <path d="M185 89 L240 89" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#ex-arrow)"/>
# MAGIC   <!-- Function -->
# MAGIC   <rect x="245" y="44" width="200" height="90" rx="12" fill="#4299E0" fill-opacity="0.10" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="345" y="74" text-anchor="middle" font-size="18" font-weight="700" fill="#0b2026">ai_extract</text>
# MAGIC   <text x="345" y="94" text-anchor="middle" font-size="14" fill="#618794">parsed_content,</text>
# MAGIC   <text x="345" y="108" text-anchor="middle" font-size="14" fill="#618794">["invoice_id",...]</text>
# MAGIC   <path d="M445 89 L505 89" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#ex-arrow)"/>
# MAGIC   <!-- Output -->
# MAGIC   <rect x="510" y="20" width="190" height="145" rx="8" fill="#ffffff" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="605" y="43" text-anchor="middle" font-size="16" font-weight="600" fill="#4299E0">Structured JSON</text>
# MAGIC   <text x="522" y="66" font-size="13" fill="#0b2026" font-family="monospace">error_message: null</text>
# MAGIC   <text x="522" y="86" font-size="13" fill="#0b2026" font-family="monospace">metadata: {...}</text>
# MAGIC   <text x="522" y="106" font-size="13" fill="#0b2026" font-family="monospace">response:</text>
# MAGIC   <text x="534" y="124" font-size="13" fill="#618794" font-family="monospace">  invoice_id: "1234"</text>
# MAGIC   <text x="534" y="142" font-size="13" fill="#618794" font-family="monospace">  date: "Jan 15, 2026"</text>
# MAGIC   <text x="534" y="160" font-size="13" fill="#618794" font-family="monospace">  total: "$4,092.64"</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC Sample Code:
# MAGIC <pre style="background: #FFFFFF; color: #0b2026; padding: 12px 14px; border: 1px solid #D8D4CA; border-radius: 6px; margin: 10px 0; font-size: 11pt; line-height: 1.5; overflow-x: auto;"><code>WITH elements AS (
# MAGIC   &amp;lt;read_file_logic&amp;gt; t,
# MAGIC   &amp;lt;variant_explode_logic&amp;gt;
# MAGIC ),
# MAGIC extracted AS (
# MAGIC   SELECT
# MAGIC     content,
# MAGIC     ai_extract(
# MAGIC       content,
# MAGIC       '["invoice_id", "date", "total"]'
# MAGIC     ) AS ext
# MAGIC   FROM elements
# MAGIC )
# MAGIC SELECT
# MAGIC   content,
# MAGIC   ext:response.invoice_id   AS invoice_id,
# MAGIC   ext:response.date    AS date,
# MAGIC   ext:response.total   AS total
# MAGIC FROM extracted
# MAGIC LIMIT 15</code></pre>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Chunking with `ai_prep_search`

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### D1. What is `ai_prep_search`?
# MAGIC
# MAGIC `ai_prep_search` ([AWS](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_prep_search) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_prep_search) | [GCP](https://docs.databricks.com/gcp/en/sql/language-manual/functions/ai_prep_search)) is a Databricks AI Function that **chunks text for AI Search**. Instead of manually splitting text by character count or using custom chunking logic, `ai_prep_search` uses AI models to split text into semantically meaningful chunks.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 720 180" role="img" style="font-family: sans-serif;">
# MAGIC   <title>ai_prep_search: chunks text at semantic boundaries for AI Search</title>
# MAGIC   <defs>
# MAGIC     <marker id="ps-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- Input -->
# MAGIC   <rect x="15" y="15" width="170" height="148" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="100" y="40" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Parsed Text</text>
# MAGIC   <!-- Gray bar above -->
# MAGIC   <rect x="25" y="48" width="150" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <!-- Chip 1 -->
# MAGIC   <rect x="25" y="63" width="150" height="22" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="100" y="78" text-anchor="middle" font-size="11" fill="#2272B4">"Property is located in..."</text>
# MAGIC   <!-- Chip 2 -->
# MAGIC   <rect x="25" y="89" width="150" height="22" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="100" y="104" text-anchor="middle" font-size="11" fill="#2272B4">"House rules include..."</text>
# MAGIC   <!-- Chip 3 -->
# MAGIC   <rect x="25" y="115" width="150" height="22" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="100" y="130" text-anchor="middle" font-size="11" fill="#2272B4">"Host info: Superhost..."</text>
# MAGIC   <!-- Gray bar below -->
# MAGIC   <rect x="25" y="142" width="150" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <path d="M185 89 L240 89" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#ps-arrow)"/>
# MAGIC   <!-- Function -->
# MAGIC   <rect x="245" y="44" width="200" height="90" rx="12" fill="#4299E0" fill-opacity="0.10" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="345" y="74" text-anchor="middle" font-size="18" font-weight="700" fill="#0b2026">ai_prep_search</text>
# MAGIC   <text x="345" y="94" text-anchor="middle" font-size="14" fill="#618794">AI-powered semantic</text>
# MAGIC   <text x="345" y="108" text-anchor="middle" font-size="14" fill="#618794">boundary detection</text>
# MAGIC   <path d="M445 89 L505 89" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#ps-arrow)"/>
# MAGIC   <!-- Output -->
# MAGIC   <rect x="510" y="20" width="195" height="145" rx="8" fill="#ffffff" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="607" y="43" text-anchor="middle" font-size="16" font-weight="600" fill="#4299E0">Chunk Array</text>
# MAGIC   <rect x="522" y="53" width="172" height="24" rx="6" fill="#4299E0" fill-opacity="0.12" stroke="#4299E0" stroke-width="1"/>
# MAGIC   <text x="530" y="69" text-anchor="start" font-size="12" fill="#0b2026">Chunk 1: "Property is..."</text>
# MAGIC   <rect x="522" y="83" width="172" height="24" rx="6" fill="#4299E0" fill-opacity="0.12" stroke="#4299E0" stroke-width="1"/>
# MAGIC   <text x="530" y="99" text-anchor="start" font-size="12" fill="#0b2026">Chunk 2: "House rules..."</text>
# MAGIC   <rect x="522" y="113" width="172" height="24" rx="6" fill="#4299E0" fill-opacity="0.12" stroke="#4299E0" stroke-width="1"/>
# MAGIC   <text x="530" y="129" text-anchor="start" font-size="12" fill="#0b2026">Chunk 3: "Host info..."</text>
# MAGIC   <!-- Gray bar in output -->
# MAGIC   <rect x="522" y="143" width="172" height="10" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC </svg>
# MAGIC </div>
# MAGIC Sample Code:
# MAGIC <pre style="background: #FFFFFF; color: #0b2026; padding: 12px 14px; border: 1px solid #D8D4CA; border-radius: 6px; margin: 10px 0; font-size: 11pt; line-height: 1.5; overflow-x: auto;"><code>SELECT ai_prep_search(parsed_text) AS chunks
# MAGIC FROM parsed_documents</code></pre>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D2. Chunking Strategy Considerations
# MAGIC
# MAGIC Regardless of the chunking method, keep these principles in mind:
# MAGIC
# MAGIC - **Chunk size vs. embedding model limits** - chunks must fit within the embedding model's context window (e.g., 512 tokens for many models). Text beyond the limit is silently truncated.
# MAGIC - **Overlap** - including a small overlap between consecutive chunks (e.g. 10-20%) prevents information from being lost at chunk boundaries.
# MAGIC - **Granularity trade-off** - smaller chunks are more precise for retrieval but may lack surrounding context. Larger chunks capture more context but may dilute the signal.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Lost in the middle</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC The "Lost in the Middle" phenomenon means LLMs tend to overlook information buried deep in long contexts. Creating smaller, focused chunks ensures critical details aren't missed during retrieval.
# MAGIC </p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ### D3. Embed vs. Retrieve: Why They're Different Columns
# MAGIC
# MAGIC From `ai_prep_search` we can produce two distinct columns from the same parsed content. Understanding this separation is important before you create an AI Search Index.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 310" role="img" style="font-family: sans-serif;">
# MAGIC   <title>chunk_to_embed vs chunk_to_retrieve: different text for different purposes</title>
# MAGIC   <defs>
# MAGIC     <marker id="er-arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#2272B4" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC     <marker id="er-arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#4299E0" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC
# MAGIC   <!-- ai_prep_search output box -->
# MAGIC   <rect x="10" y="80" width="180" height="140" rx="12" fill="#4299E0" fill-opacity="0.10" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="100" y="108" text-anchor="middle" font-size="16" font-weight="700" fill="#0b2026">ai_prep_search</text>
# MAGIC   <text x="100" y="130" text-anchor="middle" font-size="14" fill="#618794">Example output per chunk:</text>
# MAGIC   <text x="100" y="154" text-anchor="middle" font-size="13" font-family="monospace" fill="#0b2026">chunk_to_embed</text>
# MAGIC   <text x="100" y="174" text-anchor="middle" font-size="13" font-family="monospace" fill="#0b2026">chunk_to_retrieve</text>
# MAGIC   <text x="100" y="194" text-anchor="middle" font-size="13" font-family="monospace" fill="#0b2026">chunk_id</text>
# MAGIC   <text x="100" y="210" text-anchor="middle" font-size="13" font-family="monospace" fill="#0b2026">chunk_position</text>
# MAGIC
# MAGIC   <!-- Arrow to embed path -->
# MAGIC   <path d="M190 130 L260 60" fill="none" stroke="#2272B4" stroke-width="1.8" marker-end="url(#er-arrow-blue)"/>
# MAGIC   <!-- Arrow to retrieve path -->
# MAGIC   <path d="M190 170 L260 240" fill="none" stroke="#4299E0" stroke-width="1.8" marker-end="url(#er-arrow-green)"/>
# MAGIC
# MAGIC   <!-- EMBED PATH (top) -->
# MAGIC   <rect x="265" y="10" width="240" height="100" rx="8" fill="#2272B4" fill-opacity="0.10" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="385" y="36" text-anchor="middle" font-size="16" font-weight="600" fill="#2272B4">chunk_to_embed</text>
# MAGIC   <text x="385" y="58" text-anchor="middle" font-size="13" fill="#618794">Context-enriched text:</text>
# MAGIC   <text x="385" y="76" text-anchor="middle" font-size="12" font-family="monospace" fill="#0b2026">"Title: House Rules</text>
# MAGIC   <text x="385" y="92" text-anchor="middle" font-size="12" font-family="monospace" fill="#0b2026"> Section: Pets > The fee..."</text>
# MAGIC
# MAGIC   <path d="M505 60 L560 60" fill="none" stroke="#2272B4" stroke-width="1.8" marker-end="url(#er-arrow-blue)"/>
# MAGIC
# MAGIC   <!-- Embedding model -->
# MAGIC   <rect x="565" y="25" width="195" height="70" rx="8" fill="#ffffff" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="662" y="52" text-anchor="middle" font-size="14" font-weight="600" fill="#2272B4">Embedding Model</text>
# MAGIC   <text x="662" y="72" text-anchor="middle" font-size="13" fill="#618794">[0.23, -0.41, 0.08, ...]</text>
# MAGIC
# MAGIC   <!-- RETRIEVE PATH (bottom) -->
# MAGIC   <rect x="265" y="190" width="240" height="100" rx="8" fill="#4299E0" fill-opacity="0.10" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="385" y="216" text-anchor="middle" font-size="16" font-weight="600" fill="#4299E0">chunk_to_retrieve</text>
# MAGIC   <text x="385" y="238" text-anchor="middle" font-size="13" fill="#618794">Raw original text:</text>
# MAGIC   <text x="385" y="256" text-anchor="middle" font-size="12" font-family="monospace" fill="#0b2026">"The fee is $50 per pet</text>
# MAGIC   <text x="385" y="272" text-anchor="middle" font-size="12" font-family="monospace" fill="#0b2026"> per night. Max 2 pets."</text>
# MAGIC
# MAGIC   <path d="M505 240 L560 240" fill="none" stroke="#4299E0" stroke-width="1.8" marker-end="url(#er-arrow-green)"/>
# MAGIC
# MAGIC   <!-- LLM -->
# MAGIC   <rect x="565" y="205" width="195" height="70" rx="8" fill="#ffffff" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="662" y="228" text-anchor="middle" font-size="14" font-weight="600" fill="#4299E0">LLM Context</text>
# MAGIC   <text x="662" y="246" text-anchor="middle" font-size="13" fill="#618794">Generates answer from</text>
# MAGIC   <text x="662" y="262" text-anchor="middle" font-size="13" fill="#618794">clean, original text</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC Why not use the same text for both?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p style="margin-top: 10px;">Embedding and retrieval serve different goals:</p>
# MAGIC
# MAGIC <ul>
# MAGIC <li><strong>Embedding</strong> needs to answer: <em>"What is this chunk about?"</em> Adding surrounding context (headers, section titles) helps the embedding model place the chunk in the right region of vector space, even if the chunk text alone is ambiguous. A chunk that says "The fee is $50" is meaningless without context &mdash; but prepending "Cleaning Policy > Late Checkout" makes it findable.</li>
# MAGIC <li><strong>Retrieval</strong> needs to answer: <em>"What should the LLM read?"</em> The LLM needs the original text to generate accurate answers. Prepending metadata here would waste context window tokens and could confuse the model with formatting artifacts.</li>
# MAGIC </ul>
# MAGIC
# MAGIC <p>This separation is a key retrieval quality technique: <strong>optimize for findability when embedding, optimize for readability when retrieving.</strong></p>
# MAGIC
# MAGIC </div>
# MAGIC </details>

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
# MAGIC     Want to explore more about document parsing and chunking strategies? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/building-rag-agents-with-agent-bricks-en_us-2.0.0/images/20261014T152401Z/Building RAG Agents with Agent Bricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #F9F7F4; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-query" style="flex: 1;">How can I combine custom parsing strategies with AI Functions?</span>
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

# MAGIC %md
# MAGIC ## E. From Chunks to AI Search Index

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E1. What Happens When You Create an Index
# MAGIC
# MAGIC Once your chunks are in a Delta table, you create an AI Search index to make them searchable. 
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 855 245" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Index creation lifecycle</title>
# MAGIC   <defs>
# MAGIC     <marker id="e1-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- Step 1: Input -->
# MAGIC   <rect x="10" y="58" width="160" height="128" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="90" y="80" text-anchor="middle" font-size="14" font-weight="700" fill="#618794">Step 1</text>
# MAGIC   <text x="90" y="98" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">Read Source Table</text>
# MAGIC   <!-- Gray bar above -->
# MAGIC   <rect x="22" y="106" width="136" height="8" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <!-- Chip 1 -->
# MAGIC   <rect x="22" y="118" width="136" height="20" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="90" y="132" text-anchor="middle" font-size="10" fill="#2272B4">"Property is located..."</text>
# MAGIC   <!-- Chip 2 -->
# MAGIC   <rect x="22" y="142" width="136" height="20" rx="3" fill="#2272B4" fill-opacity="0.15" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="90" y="156" text-anchor="middle" font-size="10" fill="#2272B4">"House rules include..."</text>
# MAGIC   <!-- Gray bar below -->
# MAGIC   <rect x="22" y="166" width="136" height="8" rx="2" fill="#618794" fill-opacity="0.3"/>
# MAGIC   <path d="M170 122 L213 122" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#e1-arrow)"/>
# MAGIC   <!-- Step 2: Compute Embeddings -->
# MAGIC   <rect x="218" y="58" width="180" height="128" rx="12" fill="#4299E0" fill-opacity="0.10" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="308" y="82" text-anchor="middle" font-size="14" font-weight="700" fill="#618794">Step 2</text>
# MAGIC   <text x="308" y="102" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Compute</text>
# MAGIC   <text x="308" y="120" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Embeddings</text>
# MAGIC   <text x="308" y="142" text-anchor="middle" font-size="13" fill="#618794">Calls embedding model</text>
# MAGIC   <text x="308" y="158" text-anchor="middle" font-size="13" fill="#618794">on each chunk_text row</text>
# MAGIC   <text x="308" y="176" text-anchor="middle" font-size="13" fill="#4299E0">[0.23, -0.41, ...]</text>
# MAGIC   <path d="M398 122 L441 122" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#e1-arrow)"/>
# MAGIC   <!-- Step 3: ANN Index -->
# MAGIC   <rect x="446" y="58" width="180" height="128" rx="12" fill="#4299E0" fill-opacity="0.10" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="536" y="82" text-anchor="middle" font-size="14" font-weight="700" fill="#618794">Step 3</text>
# MAGIC   <text x="536" y="102" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">AI Search (ANN)</text>
# MAGIC   <text x="536" y="120" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Index</text>
# MAGIC   <text x="536" y="142" text-anchor="middle" font-size="13" fill="#618794">Organizes vectors for</text>
# MAGIC   <text x="536" y="158" text-anchor="middle" font-size="13" fill="#618794">fast nearest-neighbor</text>
# MAGIC   <text x="536" y="174" text-anchor="middle" font-size="13" fill="#618794">lookup</text>
# MAGIC   <path d="M626 122 L669 122" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#e1-arrow)"/>
# MAGIC   <!-- Step 4: Output -->
# MAGIC   <rect x="674" y="58" width="170" height="128" rx="8" fill="#ffffff" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="759" y="82" text-anchor="middle" font-size="14" font-weight="700" fill="#618794">Step 4</text>
# MAGIC   <text x="759" y="102" text-anchor="middle" font-size="16" font-weight="600" fill="#4299E0">Store Synced</text>
# MAGIC   <text x="759" y="120" text-anchor="middle" font-size="16" font-weight="600" fill="#4299E0">Columns</text>
# MAGIC   <text x="759" y="142" text-anchor="middle" font-size="13" fill="#618794">chunk_id</text>
# MAGIC   <text x="759" y="158" text-anchor="middle" font-size="13" fill="#618794">chunk_text</text>
# MAGIC   <text x="759" y="174" text-anchor="middle" font-size="13" fill="#618794">source_path</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC **Parameter to pay attention when creating a VS endpoint**
# MAGIC
# MAGIC | Parameters | What It Does |
# MAGIC |---|---|
# MAGIC | `endpoint_name` | Which VS endpoint serves the index (shared compute in this course) |
# MAGIC | `source_table_name` | The Delta table containing your chunks |
# MAGIC | `index_name` | Fully-qualified name for your index in Unity Catalog |
# MAGIC | `pipeline_type` | `TRIGGERED` (sync on demand) or `CONTINUOUS` (sync automatically) |
# MAGIC | `primary_key` | Unique row identifier used to track which rows have been processed |
# MAGIC | `embedding_source_column` | Which column to send to the embedding model |
# MAGIC | `embedding_model_endpoint_name` | The model that converts text to vectors (e.g., `databricks-gte-large-en`) |
# MAGIC | `columns_to_sync` | Which columns to store alongside vectors so they can be returned in search results |
# MAGIC
# MAGIC This initial sync can take several minutes depending on the number of rows and chunk sizes.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E2. Keeping the Index in Sync with Change Data Feed
# MAGIC
# MAGIC When you write chunks to a Delta table for AI Search, you'll enable **Change Data Feed (CDF)**:
# MAGIC
# MAGIC ```sql
# MAGIC ALTER TABLE chunks_table SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
# MAGIC ```
# MAGIC
# MAGIC For Delta Sync indexes on standard endpoints, CDF is required and is what enables incremental sync behavior. With CDF enabled, AI Search processes only rows changed since the last sync, rather than rebuilding the index the way storage-optimized endpoints partially do on each sync.
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 220" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Change Data Feed: Delta table tracks changes, VS index reads only the diff</title>
# MAGIC   <defs>
# MAGIC     <marker id="cdf-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC
# MAGIC   <!-- Delta Table -->
# MAGIC   <rect x="10" y="30" width="190" height="160" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.5"/>
# MAGIC   <text x="105" y="56" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Delta Table</text>
# MAGIC   <text x="105" y="76" text-anchor="middle" font-size="13" fill="#618794">(chunks)</text>
# MAGIC
# MAGIC   <!-- Rows showing changes -->
# MAGIC   <rect x="24" y="90" width="162" height="20" rx="3" fill="#618794" fill-opacity="0.15"/>
# MAGIC   <text x="105" y="104" text-anchor="middle" font-size="12" fill="#618794">existing row (unchanged)</text>
# MAGIC   <rect x="24" y="114" width="162" height="20" rx="3" fill="#4299E0" fill-opacity="0.2" stroke="#4299E0" stroke-width="0.8"/>
# MAGIC   <text x="105" y="128" text-anchor="middle" font-size="12" fill="#4299E0">+ new row (INSERT)</text>
# MAGIC   <rect x="24" y="138" width="162" height="20" rx="3" fill="#2272B4" fill-opacity="0.2" stroke="#2272B4" stroke-width="0.8"/>
# MAGIC   <text x="105" y="152" text-anchor="middle" font-size="12" fill="#2272B4">~ updated row (UPDATE)</text>
# MAGIC   <rect x="24" y="162" width="162" height="20" rx="3" fill="#FF5F46" fill-opacity="0.2" stroke="#FF5F46" stroke-width="0.8"/>
# MAGIC   <text x="105" y="176" text-anchor="middle" font-size="12" fill="#FF5F46">- deleted row (DELETE)</text>
# MAGIC
# MAGIC   <!-- Arrow: CDF change log -->
# MAGIC   <path d="M200 110 L280 110" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#cdf-arrow)"/>
# MAGIC   <text x="240" y="100" text-anchor="middle" font-size="12" font-weight="600" fill="#618794">CDF log</text>
# MAGIC
# MAGIC   <!-- CDF Change Log -->
# MAGIC   <rect x="285" y="50" width="200" height="120" rx="8" fill="#FF5F46" fill-opacity="0.06" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="385" y="76" text-anchor="middle" font-size="15" font-weight="600" fill="#0b2026">Change Data Feed</text>
# MAGIC   <text x="385" y="98" text-anchor="middle" font-size="13" fill="#618794">Records only the diff:</text>
# MAGIC   <text x="385" y="118" text-anchor="middle" font-size="12" font-family="monospace" fill="#0b2026">_change_type</text>
# MAGIC   <text x="385" y="136" text-anchor="middle" font-size="12" font-family="monospace" fill="#0b2026">_commit_version</text>
# MAGIC   <text x="385" y="154" text-anchor="middle" font-size="12" font-family="monospace" fill="#0b2026">_commit_timestamp</text>
# MAGIC
# MAGIC   <!-- Arrow: sync -->
# MAGIC   <path d="M485 110 L555 110" fill="none" stroke="#1B3139" stroke-width="1.8" marker-end="url(#cdf-arrow)"/>
# MAGIC   <text x="520" y="100" text-anchor="middle" font-size="12" font-weight="600" fill="#618794">sync</text>
# MAGIC
# MAGIC   <!-- VS Index -->
# MAGIC   <rect x="560" y="40" width="200" height="140" rx="8" fill="#4299E0" fill-opacity="0.08" stroke="#4299E0" stroke-width="1.5"/>
# MAGIC   <text x="660" y="68" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">AI Search</text>
# MAGIC   <text x="660" y="86" text-anchor="middle" font-size="16" font-weight="600" fill="#0b2026">Index</text>
# MAGIC   <text x="660" y="110" text-anchor="middle" font-size="13" fill="#618794">Processes only changed</text>
# MAGIC   <text x="660" y="126" text-anchor="middle" font-size="13" fill="#618794">rows, not the full</text>
# MAGIC   <text x="660" y="142" text-anchor="middle" font-size="13" fill="#618794">table on every sync</text>
# MAGIC   <text x="660" y="168" text-anchor="middle" font-size="12" fill="#4299E0" font-weight="600">Fast incremental updates</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC **Sync modes** control *when* those changes flow to the index:
# MAGIC
# MAGIC | Mode | Behavior | Trade-off |
# MAGIC |---|---|---|
# MAGIC | **`TRIGGERED`** | Syncs only when you explicitly call `index.sync()` or through the UI | You control when updates happen, which is good for batch workflows where you update chunks periodically |
# MAGIC | **`CONTINUOUS`** | Syncs automatically as the source table changes | Near-real-time freshness, but uses more compute |

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Other AI SQL Functions
# MAGIC To see the full suite of AI functions, you can view the latest list of task-specific AI functions here ([AWS](https://docs.databricks.com/aws/en/large-language-models/ai-functions#-task-specific-ai-functions) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/large-language-models/ai-functions) | [GCP](https://docs.databricks.com/gcp/en/large-language-models/ai-functions)).

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC