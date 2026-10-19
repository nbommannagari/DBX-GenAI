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

# MAGIC %md-sandbox
# MAGIC # Lecture - The Challenge of Evaluating AI Agents
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC Traditional software testing approaches are fundamentally insufficient for AI agents due to their non-deterministic nature, emergent behaviors, and context-dependent responses. This lecture explores why conventional testing fails with AI agents and introduces the unique challenges that require specialized evaluation frameworks.
# MAGIC
# MAGIC We'll examine how AI agents break traditional testing paradigms, understand the complexity of multi-step reasoning evaluation, and learn why evaluation must be treated as a continuous process rather than a one-time validation step.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lecture, you will be able to:
# MAGIC - Explain why traditional software testing approaches are insufficient for AI agents
# MAGIC - Identify the key challenges unique to evaluating AI agents (non-determinism, emergent behavior, context dependency)
# MAGIC - Understand why evaluation must be treated as a continuous process
# MAGIC - Recognize the importance of proper evaluation dataset design
# MAGIC - Describe the operational setup requirements for systematic agent evaluation

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## A. Why Traditional Testing Falls Short

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A1. The Deterministic Testing Paradigm
# MAGIC
# MAGIC Traditional software testing relies on deterministic inputs and expected outputs. You write unit tests, integration tests, and end-to-end tests that verify your code produces the exact same result every time given the same input.
# MAGIC
# MAGIC <div style="display:flex;justify-content:center;gap:16px;margin:18px 0;font-family:system-ui,sans-serif;flex-wrap:wrap;">
# MAGIC   <div style="flex:1;min-width:200px;max-width:220px;border:1px solid #618794;border-radius:8px;padding:12px;text-align:center;background:#fff;">
# MAGIC     <div style="font-size:13px;font-weight:600;color:#618794;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Assumption</div>
# MAGIC     <div style="background:#EEEDE9;border-radius:6px;padding:8px;font-size:14px;margin-bottom:4px;">Same input &#x2192; same output</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:200px;max-width:220px;border:1px solid #618794;border-radius:8px;padding:12px;text-align:center;background:#fff;">
# MAGIC     <div style="font-size:13px;font-weight:600;color:#618794;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Verification</div>
# MAGIC     <div style="background:#EEEDE9;border-radius:6px;padding:8px;font-size:14px;margin-bottom:4px;">Exact string/number match</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:200px;max-width:220px;border:1px solid #618794;border-radius:8px;padding:12px;text-align:center;background:#fff;">
# MAGIC     <div style="font-size:13px;font-weight:600;color:#618794;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Behavior</div>
# MAGIC     <div style="background:#EEEDE9;border-radius:6px;padding:8px;font-size:14px;margin-bottom:4px;">Explicitly programmed</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:200px;max-width:220px;border:1px solid #618794;border-radius:8px;padding:12px;text-align:center;background:#fff;">
# MAGIC     <div style="font-size:13px;font-weight:600;color:#618794;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Edge Cases</div>
# MAGIC     <div style="background:#EEEDE9;border-radius:6px;padding:8px;font-size:14px;margin-bottom:4px;">Anticipated &amp; systematic</div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### A2. How AI Agents Break the Paradigm
# MAGIC
# MAGIC AI agents fundamentally break the deterministic testing paradigm:
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:12px;margin:18px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="flex:1;min-width:170px;background:#2272B4;color:#fff;padding:14px 16px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:15px;">Non-determinism</div>
# MAGIC     <div style="font-size:13px;margin-top:6px;opacity:0.9;">Same input can produce different outputs due to temperature &amp; sampling</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:170px;background:#00A972;color:#fff;padding:14px 16px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:15px;">Emergent Behavior</div>
# MAGIC     <div style="font-size:13px;margin-top:6px;opacity:0.9;">Autonomous tool &amp; reasoning decisions not explicitly programmed</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:170px;background:#FF5F46;color:#fff;padding:14px 16px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:15px;">Context Dependency</div>
# MAGIC     <div style="font-size:13px;margin-top:6px;opacity:0.9;">Responses depend on retrieval, history, &amp; external data</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:170px;background:#1B5162;color:#fff;padding:14px 16px;border-radius:8px;text-align:center;">
# MAGIC     <div style="font-weight:600;font-size:15px;">Qualitative Assessment</div>
# MAGIC     <div style="font-size:13px;margin-top:6px;opacity:0.9;">Success requires judgment, not exact string matching</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <details style="margin:12px 0;border:1.5px solid #1B5162;border-radius:10px;background:#F9F7F4;padding:0 16px;">
# MAGIC <summary style="cursor:pointer;padding:12px 0;font-weight:600;font-size:15px;color:#0b2026;">Example: Why exact matching fails</summary>
# MAGIC <div style="padding:0 0 12px 0;font-size:14px;color:#333;">
# MAGIC An agent answering "What's the weather in San Francisco?" might respond:
# MAGIC <ul style="margin:8px 0;">
# MAGIC <li>"It's currently 65°F and sunny in San Francisco."</li>
# MAGIC <li>"San Francisco weather: 65 degrees, clear skies."</li>
# MAGIC <li>"The temperature in SF is 65°F with no clouds."</li>
# MAGIC </ul>
# MAGIC All three are correct, helpful, and appropriate. Yet none match exactly. Traditional <code>assert output == "expected_response"</code> would fail on all three.
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. The Agent Evaluation Challenge

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Multi-Dimensional Complexity
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 780 200" role="img" style="font-family: sans-serif;">
# MAGIC   <title>The Agent Evaluation Challenge</title>
# MAGIC   <desc>Four evaluation dimensions: Multi-step Reasoning, Tool Calling, Retrieval Quality, and Safety, all feeding into a central Evaluation challenge.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="ec-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- Multi-step Reasoning -->
# MAGIC   <rect x="10" y="10" width="160" height="70" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="90" y="38" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Multi-step</text>
# MAGIC   <text x="90" y="56" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Reasoning</text>
# MAGIC   <!-- Tool Calling -->
# MAGIC   <rect x="10" y="110" width="160" height="70" rx="8" fill="#00A972" fill-opacity="0.12" stroke="#00A972" stroke-width="1.5"/>
# MAGIC   <text x="90" y="142" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Tool Calling</text>
# MAGIC   <text x="90" y="160" text-anchor="middle" font-size="13" fill="#618794">Accuracy</text>
# MAGIC   <!-- Retrieval Quality -->
# MAGIC   <rect x="610" y="10" width="160" height="70" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="690" y="38" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Retrieval</text>
# MAGIC   <text x="690" y="56" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Quality</text>
# MAGIC   <!-- Safety -->
# MAGIC   <rect x="610" y="110" width="160" height="70" rx="8" fill="#1B5162" fill-opacity="0.12" stroke="#1B5162" stroke-width="1.5"/>
# MAGIC   <text x="690" y="142" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Safety &amp;</text>
# MAGIC   <text x="690" y="160" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Alignment</text>
# MAGIC   <!-- Center: Evaluation Challenge -->
# MAGIC   <rect x="250" y="50" width="280" height="90" rx="12" fill="#F9F7F4" stroke="#1B3139" stroke-width="2"/>
# MAGIC   <text x="390" y="84" text-anchor="middle" font-size="17" font-weight="700" fill="#FF5F46">Agent Evaluation</text>
# MAGIC   <text x="390" y="106" text-anchor="middle" font-size="14" fill="#618794">Non-deterministic &amp; context-dependent</text>
# MAGIC   <!-- Arrows -->
# MAGIC   <path d="M170 45 L250 75" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ec-arrow)"/>
# MAGIC   <path d="M170 145 L250 115" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ec-arrow)"/>
# MAGIC   <path d="M610 45 L530 75" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ec-arrow)"/>
# MAGIC   <path d="M610 145 L530 115" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ec-arrow)"/>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC Evaluating AI agents introduces unique complexities beyond traditional software:
# MAGIC
# MAGIC **Multi-step reasoning**: Agents may invoke multiple tools, retrieve various documents, and build complex reasoning chains. Evaluation must assess not just the final answer but the quality of intermediate steps.
# MAGIC
# MAGIC **Tool calling accuracy**: Did the agent select the right tools? Did it pass appropriate parameters? Did it correctly interpret tool results?
# MAGIC
# MAGIC **Retrieval quality**: For RAG-based agents, evaluation must verify that retrieved documents contain relevant information and that the agent correctly synthesizes information from multiple sources.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Safety and Real-World Variability
# MAGIC
# MAGIC **Safety and alignment**: Agents must avoid harmful outputs, respect user boundaries, and decline inappropriate requests. These qualities require sophisticated evaluation beyond simple pass/fail tests.
# MAGIC
# MAGIC **Real-world variability**: Production agents encounter diverse user queries, unexpected phrasings, and edge cases that are difficult to anticipate during development.
# MAGIC
# MAGIC These challenges demand a more sophisticated evaluation framework specifically designed for the probabilistic, contextual nature of AI agents.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Evaluation as a Continuous Process

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. The Continuous Evaluation Cycle
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="background:#2272B4;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:140px;">Development<br/><span style="font-weight:400;font-size:13px;">Rapid iteration &amp;<br/>frequent evaluation</span></div>
# MAGIC   <div style="color:#999;font-size:22px;padding:0 8px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:140px;">Pre-deployment<br/><span style="font-weight:400;font-size:13px;">Comprehensive<br/>validation</span></div>
# MAGIC   <div style="color:#999;font-size:22px;padding:0 8px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:140px;">Production<br/><span style="font-weight:400;font-size:13px;">Continuous<br/>monitoring</span></div>
# MAGIC   <div style="color:#999;font-size:22px;padding:0 8px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:14px 20px;border-radius:8px;font-weight:600;font-size:15px;text-align:center;min-width:140px;">Dataset Evolution<br/><span style="font-weight:400;font-size:13px;">Feedback loops &amp;<br/>improvement</span></div>
# MAGIC </div>
# MAGIC
# MAGIC Unlike traditional software where comprehensive test suites provide stable quality signals, AI agent evaluation is an ongoing process:
# MAGIC
# MAGIC **Development phase**: Rapid iteration requires frequent evaluation to validate that changes improve quality without introducing regressions.
# MAGIC
# MAGIC **Pre-deployment validation**: Comprehensive evaluation across diverse test cases ensures agents meet quality bars before production release.
# MAGIC
# MAGIC **Production monitoring**: Continuous evaluation of live interactions identifies quality degradation, emerging failure patterns, and opportunities for improvement.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C2. Why Continuous Evaluation Matters
# MAGIC
# MAGIC This continuous evaluation cycle means your evaluation infrastructure must be scalable, automated, and integrated into your development workflow. The evaluation framework must support:
# MAGIC
# MAGIC - **Rapid feedback loops** during development
# MAGIC - **Comprehensive validation** before deployment
# MAGIC - **Ongoing monitoring** in production
# MAGIC - **Dataset evolution** as usage patterns change

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Preparing for Evaluation

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D1. Why Preparation Matters
# MAGIC
# MAGIC Before using any evaluation framework, it's important to make evaluation both purposeful and reproducible. AI agents are non-deterministic and context-dependent, so traditional assertion-style tests fall short. Instead, effective evaluation requires defining quality dimensions, assembling representative datasets, and enabling tracing so judges can assess not just answers, but how those answers were produced.
# MAGIC
# MAGIC By clarifying goals, curating datasets, and selecting appropriate judges up front, you create a feedback loop where metrics reflect real user needs and failures are diagnosable through traces and rationales: not just pass/fail scores.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ### D2. Designing Your Evaluation Dataset
# MAGIC
# MAGIC <div style="max-width: 900px; margin: 0 auto;">
# MAGIC <svg width="100%" viewBox="0 0 800 180" role="img" style="font-family: sans-serif;">
# MAGIC   <title>Designing Evaluation Datasets</title>
# MAGIC   <desc>Evaluation dataset structure: Inputs, Outputs, and Expectations feed into an Evaluation Dataset.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="ds-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
# MAGIC       <path d="M2 1L8 5L2 9" fill="none" stroke="#1B3139" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
# MAGIC     </marker>
# MAGIC   </defs>
# MAGIC   <!-- Input components -->
# MAGIC   <rect x="10" y="10" width="150" height="50" rx="8" fill="#2272B4" fill-opacity="0.12" stroke="#2272B4" stroke-width="1.5"/>
# MAGIC   <text x="85" y="40" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Inputs (Queries)</text>
# MAGIC   <rect x="10" y="70" width="150" height="50" rx="8" fill="#00A972" fill-opacity="0.12" stroke="#00A972" stroke-width="1.5"/>
# MAGIC   <text x="85" y="100" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Outputs (optional)</text>
# MAGIC   <rect x="10" y="130" width="150" height="50" rx="8" fill="#FF5F46" fill-opacity="0.12" stroke="#FF5F46" stroke-width="1.5"/>
# MAGIC   <text x="85" y="160" text-anchor="middle" font-size="14" font-weight="600" fill="#0b2026">Expectations</text>
# MAGIC   <!-- Arrows to center -->
# MAGIC   <path d="M160 35 L260 75" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ds-arrow)"/>
# MAGIC   <path d="M160 95 L260 90" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ds-arrow)"/>
# MAGIC   <path d="M160 155 L260 105" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ds-arrow)"/>
# MAGIC   <!-- Central dataset box -->
# MAGIC   <rect x="265" y="45" width="250" height="90" rx="12" fill="#F9F7F4" stroke="#1B3139" stroke-width="2"/>
# MAGIC   <text x="390" y="78" text-anchor="middle" font-size="16" font-weight="700" fill="#0b2026">Evaluation Dataset</text>
# MAGIC   <text x="390" y="100" text-anchor="middle" font-size="13" fill="#618794">JSON / DataFrame / Delta Table</text>
# MAGIC   <text x="390" y="118" text-anchor="middle" font-size="13" fill="#618794">Stored in Unity Catalog</text>
# MAGIC   <!-- Output properties -->
# MAGIC   <path d="M515 70 L580 35" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ds-arrow)"/>
# MAGIC   <path d="M515 90 L580 90" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ds-arrow)"/>
# MAGIC   <path d="M515 110 L580 145" fill="none" stroke="#1B3139" stroke-width="1.5" marker-end="url(#ds-arrow)"/>
# MAGIC   <rect x="585" y="10" width="210" height="50" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.2"/>
# MAGIC   <text x="690" y="30" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Representativeness</text>
# MAGIC   <text x="690" y="48" text-anchor="middle" font-size="14" fill="#618794">Common &amp; high-impact queries</text>
# MAGIC   <rect x="585" y="70" width="210" height="50" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.2"/>
# MAGIC   <text x="690" y="90" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Edge Cases</text>
# MAGIC   <text x="690" y="108" text-anchor="middle" font-size="14" fill="#618794">Adversarial &amp; ambiguous</text>
# MAGIC   <rect x="585" y="130" width="210" height="50" rx="8" fill="#EEEDE9" stroke="#1B3139" stroke-width="1.2"/>
# MAGIC   <text x="690" y="150" text-anchor="middle" font-size="13" font-weight="600" fill="#0b2026">Diversity</text>
# MAGIC   <text x="690" y="168" text-anchor="middle" font-size="14" fill="#618794">Length, complexity, domain</text>
# MAGIC </svg>
# MAGIC </div>
# MAGIC
# MAGIC Your evaluation dataset defines what you test and how trustworthy your signals will be. At minimum, it should include inputs (queries) and, where appropriate, expected answers, per-row guidelines, and metadata that reflects retrieval and tool usage.
# MAGIC
# MAGIC **Key principles:**
# MAGIC
# MAGIC - **Representativeness:** Include common, high-impact user queries so offline results generalize to production
# MAGIC - **Edge cases:** Add ambiguous, out-of-scope, and adversarial prompts to surface failure modes early
# MAGIC - **Diversity:** Vary length, complexity, domain, and user expertise to expose blind spots in reasoning and retrieval
# MAGIC - **Ground truth and/or guidelines:** Use expected answers or fact sets for objective questions; use natural-language guidelines where style, policy, or completeness matter
# MAGIC - **Storage and versioning:** MLflow evaluation datasets are stored in Unity Catalog, which provides built-in versioning, lineage, sharing, and governance

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### D3. Operational Setup Requirements
# MAGIC
# MAGIC Establish consistent scaffolding so results are comparable and auditable:
# MAGIC
# MAGIC <div style="display:flex;align-items:stretch;justify-content:center;gap:12px;margin:18px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="flex:1;min-width:220px;border:2px solid #2272B4;border-radius:10px;padding:14px;background:#fff;">
# MAGIC     <div style="font-size:14px;font-weight:700;color:#2272B4;margin-bottom:8px;">MLflow Experiments &amp; Runs</div>
# MAGIC     <div style="font-size:13px;color:#333;">Stable experiment names; tag runs with agent version, dataset version, and parameters; compare metrics in the UI</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:220px;border:2px solid #00A972;border-radius:10px;padding:14px;background:#fff;">
# MAGIC     <div style="font-size:14px;font-weight:700;color:#00A972;margin-bottom:8px;">Unity Catalog Integration</div>
# MAGIC     <div style="font-size:13px;color:#333;">Govern datasets and traces with access control, versioning, and lineage; register agents for end-to-end traceability</div>
# MAGIC   </div>
# MAGIC   <div style="flex:1;min-width:220px;border:2px solid #FF5F46;border-radius:10px;padding:14px;background:#fff;">
# MAGIC     <div style="font-size:14px;font-weight:700;color:#FF5F46;margin-bottom:8px;">Production Feedback Loop</div>
# MAGIC     <div style="font-size:13px;color:#333;">Enable Unity Gateway-enabled inference tables to log requests, responses, and traces for monitoring and mining new evaluation examples</div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;">
# MAGIC
# MAGIC <div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">
# MAGIC
# MAGIC   <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">
# MAGIC
# MAGIC   <div style="font-size: 15pt; color: #0b2026; line-height: 1.7; margin-bottom: 16px;">
# MAGIC     Want to explore more about agent evaluation challenges? Ask Genie Code. Click on the genie icon <img src="https://files.training.databricks.com/binder/prod_main/genai-agent-evaluation-on-databricks-en_us-1.1.1/images/20261006T164734Z/Agent Evaluation on Databricks/Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.
# MAGIC   </div>
# MAGIC
# MAGIC   <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0b2026;">
# MAGIC     <span id="genie-1-1" style="flex: 1;">What are the key differences between evaluating traditional software and evaluating AI agents?</span>
# MAGIC     <button onclick="
# MAGIC       var text = document.getElementById('genie-1-1').innerText;
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
# MAGIC ## E. Conclusion
# MAGIC
# MAGIC Traditional software testing approaches are fundamentally inadequate for AI agents due to their non-deterministic, emergent, and context-dependent nature.
# MAGIC
# MAGIC <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:18px 0;flex-wrap:wrap;font-family:system-ui,sans-serif;">
# MAGIC   <div style="background:#2272B4;color:#fff;padding:12px 16px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:150px;">1. Recognize<br/>Unique Challenges<br/><span style="font-weight:400;font-size:14px;">Non-determinism,<br/>emergent behavior</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#00A972;color:#fff;padding:12px 16px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:150px;">2. Continuous<br/>Evaluation<br/><span style="font-weight:400;font-size:14px;">Ongoing process,<br/>not one-time</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#FF5F46;color:#fff;padding:12px 16px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:150px;">3. Prepare<br/>Thoughtfully<br/><span style="font-weight:400;font-size:14px;">Dataset design,<br/>quality definitions</span></div>
# MAGIC   <div style="color:#999;font-size:20px;padding:0 6px;">&#x2192;</div>
# MAGIC   <div style="background:#1B5162;color:#fff;padding:12px 16px;border-radius:8px;font-weight:600;font-size:14px;text-align:center;min-width:150px;">4. Systematic<br/>Approach<br/><span style="font-weight:400;font-size:14px;">Scalable, automated,<br/>integrated</span></div>
# MAGIC </div>
# MAGIC
# MAGIC Understanding these challenges is the foundation for implementing effective agent evaluation. The next lectures will explore the tools and techniques that address these challenges systematically.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC
# MAGIC