# DBX-GenAI
Covers a different sector of the building process, giving you a comprehensive understanding of Generative AI Engineering with Databricks. 


#### How does Databricks AI Search handle hybrid keyword and semantic search?

Databricks AI Search supports three retrieval modes — ANN (semantic, the default), HYBRID, and FULL_TEXT (keyword only, beta) — and hybrid is the recommended starting point for most use cases.

How hybrid search works: It runs both an ANN vector similarity search and a BM25 keyword search in parallel, then merges the two ranked lists using Reciprocal Rank Fusion (RRF). This gives you the semantic understanding of embeddings (catching paraphrases and conceptual matches) combined with the precision of exact keyword matching (catching SKUs, error codes, proper nouns, and technical terms).

When to use it:

Queries mix conceptual intent with specific identifiers (e.g., "SPARK-12345 executor memory error")
Users search with product codes, technical jargon, or proper nouns that pure semantic search might miss
You want a robust general-purpose strategy without tuning per-query type
Enabling it is a one-line change — set query_type="HYBRID" when calling query_index or similarity_search. For managed-embedding indexes, you only need query_text; for self-managed indexes without a model endpoint, you provide both query_vector (for ANN) and query_text (for BM25).

There's also a Hybrid + Reranker option that re-scores the merged results with a cross-encoder model for higher precision, at the cost of modest additional latency (typically under 1 second). The Databricks AI Search query documentation has the full parameter reference.


#### Explain ANN vs hybrid trade-offs

Here's how ANN and hybrid search compare in practice:

ANN (Approximate Nearest Neighbor) — semantic only

Matches on meaning, so it handles paraphrases, synonyms, and conceptual queries well
Fastest option — single vector similarity pass with no additional scoring
May miss results when the query contains exact identifiers (error codes, SKUs, proper nouns) that don't carry strong semantic signal in embedding space
Hybrid — ANN + BM25 keyword scoring

Runs both a vector search and a keyword search in parallel, then merges results with Reciprocal Rank Fusion (RRF)
Catches documents that share exact terms the semantic pass might miss, improving recall for mixed-intent queries
Roughly 2× the resource usage compared to ANN alone (two retrieval passes instead of one)
Capped at 200 results per query (ANN has no such cap)
Recommended as the default starting point by Databricks — it's the most robust general-purpose strategy
When to pick which:


| Signal | Choose |
| --- | --- |
| Conceptual or paraphrased queries ("how do I handle pipeline errors?") | ANN is sufficient |
| Queries with codes, IDs, or proper nouns ("SPARK-12345 memory error") | RHybrid |
| Mixed workload you can't predict | Hybrid — safer default |
| Latency/cost is critical and queries are purely conceptual | NN to save the extra scoring pass |

The lecture notebook's section A2 covers the ANN retrieval flow and cosine similarity mechanics in Cell 2: Lecture - AI Se.... 

The key takeaway: start with hybrid unless you have a specific reason to optimize for pure semantic retrieval, and switch to ANN only if profiling shows the keyword pass adds no recall benefit for your query workload.