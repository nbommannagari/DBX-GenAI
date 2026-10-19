# Databricks notebook source
# Change the course name as per your course name
COURSE_NAME = "Deploying and Monitoring Agents on Databricks"
RUN_QA_CHECKER = True
QA_TASK_RELATIVE_PATH = "./qa_content_checker"
QA_TASK_NAME = "QA Content Checker — Grammar, Deprecated, UI Steps"

# Tester emails — notified on job failure and success. Add as many as you like with comma separated emails
TESTER_EMAILS = [
    "aditya.baghel@databricks.com"
]

# Add your lab notebook paths relative from the CourseRunner folder as per your course
LAB_NOTEBOOKS = [
    "../../6 - Lab - Online Evaluation with Multi-Turn Conversations",
    "../../7 - Lab - Backfilling and Archiving Multi-Turn Conversations",
]

# Define the course notebook DAG here so this can be used for job scheduling as per your course
# All notebooks run completely (each notebook internally %run's its own Classroom-Setup)
COURSE_TASKS = [
    {
        "task_key": "02_Demo_Deploying_Observable_Agent",
        "name": "2 - Demo - Deploying an Observable Agent",
        "relative_path": "../../2 - Demo - Deploying an Observable Agent",
        "depends_on": [],
        "env_key": "course_02",
    },
    {
        "task_key": "04_Demo_Tracing_Production_Agents",
        "name": "4 - Demo - Tracing for Production Agents",
        "relative_path": "../../4 - Demo - Tracing for Production Agents",
        "depends_on": ["02_Demo_Deploying_Observable_Agent"],
        "env_key": "course_04",
    },
    {
        "task_key": "06_Lab_Online_Evaluation",
        "name": "6 - Lab - Online Evaluation with Multi-Turn Conversations",
        "relative_path": "../../6 - Lab - Online Evaluation with Multi-Turn Conversations",
        "depends_on": ["04_Demo_Tracing_Production_Agents"],
        "env_key": "course_06",
    },
    {
        "task_key": "07_Lab_Backfilling_Archiving",
        "name": "7 - Lab - Backfilling and Archiving Multi-Turn Conversations",
        "relative_path": "../../7 - Lab - Backfilling and Archiving Multi-Turn Conversations",
        "depends_on": ["06_Lab_Online_Evaluation"],
        "env_key": "course_07",
    },
]

# COMMAND ----------

# ── Auto-Fill Lab Notebooks ──────────────────────────────────────────────────
# This script finds unresolved lab placeholders in lab notebooks and replaces
# them with solution code extracted from the nearby solution blocks.
#
# HOW IT WORKS:
# 1. Exports each lab notebook as .ipynb JSON
# 2. Scans code cells for unresolved markers such as <FILL_IN>, <FILL-IN>,
#    <ToDo>, and TODO-only scaffolds
# 3. Looks ahead in subsequent markdown cells for <code>...</code> blocks
# 4. Extracts the solution and replaces the original cell source
# 5. Re-imports the patched notebook
#
# Safe to re-run: it only patches cells that still contain unresolved markers.
# ──────────────────────────────────────────────────────────────────────────────

import base64
import json
import os
import re
import html as html_lib
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat, ImportFormat, Language

w = WorkspaceClient()

# Derive course_root from this notebook's own path
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
course_root = "/".join(_nb_path.split("/")[:-1])

# LAB_NOTEBOOKS is configured in the course-specific inputs cell above.
# Resolve to absolute workspace paths
lab_abs_paths = [os.path.normpath(f"{course_root}/{p}") for p in LAB_NOTEBOOKS]

PLACEHOLDER_PATTERNS = [
    re.compile(r'<\s*fill[-_ ]?in\s*>', re.IGNORECASE),
    re.compile(r'--\s*fill[-_ ]?in\s*--', re.IGNORECASE),
    re.compile(r'<\s*todo\s*>', re.IGNORECASE),
    re.compile(r'(?mi)^\s*#+\s*todo\b'),
    re.compile(r'(?mi)^\s*#+.*\bTODO\b'),
]


def has_unresolved_placeholder(source: str) -> bool:
    """Return True when a code cell still contains a fill-in or TODO scaffold."""
    return any(pattern.search(source) for pattern in PLACEHOLDER_PATTERNS)


def extract_solution_from_source(source: str) -> str | None:
    """Extract code from a Databricks solution cell.

    Handles:
    - Fenced code blocks: ```python\\n...\\n``` (used in current labs)
    - HTML <code> tags: <code>...</code> (legacy fallback)
    - Strips %md / %md-sandbox magic prefixes before parsing.
    """
    # Strip Databricks magic prefix (%md, %md-sandbox, etc.)
    clean = re.sub(r'^%md[a-zA-Z-]*[ \t]*\n', '', source.lstrip(), count=1)

    # 1. Fenced code block (most common in current labs)
    match = re.search(r'```(?:python|sql|scala|r|sh)?\s*\n(.*?)\n[ \t]*```', clean, re.DOTALL)
    if match:
        code = match.group(1)
        lines = code.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return '\n'.join(lines) if lines else None

    # 2. HTML <code> tags (fallback for older lab format)
    match = re.search(r'<code[^>]*>\s*\n?(.*?)\n?\s*</code>', clean, re.DOTALL)
    if match:
        code = html_lib.unescape(match.group(1))
        lines = code.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return '\n'.join(lines) if lines else None

    return None


def is_solution_candidate(cell: dict) -> bool:
    """Return True if a cell is a potential answer/solution block.

    In Databricks ipynb exports, %md-sandbox cells are stored as code cells
    (not markdown), so we must check both cell_type=='markdown' and code cells
    that start with a %md magic prefix.
    """
    source = ''.join(cell.get("source", []))
    has_code_block = '```' in source or '<code>' in source
    is_markdown_type = cell.get("cell_type") == "markdown"
    is_md_magic = cell.get("cell_type") == "code" and bool(re.match(r'\s*%md', source))
    return (is_markdown_type or is_md_magic) and has_code_block


def fill_notebook(notebook_path: str) -> dict:
    """Fill all unresolved placeholder cells in a single notebook. Returns stats."""
    stats = {"path": notebook_path, "blanks_found": 0, "filled": 0, "no_answer": 0}

    # Export notebook as ipynb
    export_resp = w.workspace.export(path=notebook_path, format=ExportFormat.JUPYTER)
    nb_json = json.loads(base64.b64decode(export_resp.content))
    cells = nb_json.get("cells", [])

    for i, cell in enumerate(cells):
        source = ''.join(cell.get("source", []))

        # Check if this cell still has unresolved fill-in / TODO markers
        if not has_unresolved_placeholder(source):
            continue

        # Skip markdown cells (we only fill code cells)
        if cell.get("cell_type") != "code":
            continue

        stats["blanks_found"] += 1

        # Look ahead up to 8 cells for a solution block.
        # %md-sandbox solution cells may appear as 'code' type in ipynb exports,
        # so is_solution_candidate() checks both cell_type and %md magic prefix.
        solution = None
        for j in range(i + 1, min(i + 9, len(cells))):
            next_cell = cells[j]
            next_source = ''.join(next_cell.get("source", []))
            if is_solution_candidate(next_cell):
                solution = extract_solution_from_source(next_source)
                if solution:
                    break
            # Stop looking once we hit a real (non-magic) code cell
            elif next_cell.get("cell_type") == "code" and not re.match(r'\s*%', next_source):
                break

        if solution:
            # Check if solution has %sql magic — if so, handle cell language
            if solution.strip().startswith('%sql'):
                # Remove %sql prefix, keep the SQL code
                solution_lines = solution.strip().split('\n')
                # Keep %sql as first line for the cell magic
                cell["source"] = [solution + '\n']
            else:
                cell["source"] = [solution + '\n']
            stats["filled"] += 1
        else:
            stats["no_answer"] += 1
            print(f"  ⚠️  NO ANSWER FOUND for cell {i} in {notebook_path}")

    # Re-import the patched notebook
    if stats["filled"] > 0:
        patched_content = base64.b64encode(json.dumps(nb_json).encode()).decode()
        w.workspace.import_(
            path=notebook_path,
            content=patched_content,
            format=ImportFormat.JUPYTER,
            overwrite=True,
            language=Language.PYTHON,
        )

    return stats


# ── Run the filler ────────────────────────────────────────────────────────────
print("═" * 70)
print("AUTO-FILL LAB NOTEBOOKS")
print("═" * 70)

all_stats = []
for path in lab_abs_paths:
    print(f"\n▶ Processing: {path.split('/')[-1]}")
    try:
        s = fill_notebook(path)
        all_stats.append(s)
        print(f"   Blanks found: {s['blanks_found']}  |  Filled: {s['filled']}  |  No answer: {s['no_answer']}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        all_stats.append({"path": path, "blanks_found": 0, "filled": 0, "no_answer": 0, "error": str(e)})

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
total_blanks = sum(s["blanks_found"] for s in all_stats)
total_filled = sum(s["filled"] for s in all_stats)
total_no_ans = sum(s["no_answer"] for s in all_stats)
print(f"TOTAL:  {total_blanks} blanks found  |  {total_filled} filled  |  {total_no_ans} blockers")
print("═" * 70)

# COMMAND ----------

# Catalog / schema / table names
RESULTS_CATALOG   = "dbacademy"
RESULTS_SCHEMA    = spark.sql("SELECT current_user()").collect()[0][0].split("@")[0].replace(".", "_").replace("-", "_")
RESULTS_TABLE     = "notebook_run_results"
QA_FINDINGS_TABLE = "qa_content_findings"

# How long to wait for the entire job run (seconds)
TOTAL_RUN_TIMEOUT_SECONDS = 60 * 60 * 2   # 2 hour ceiling
POLL_INTERVAL_SECONDS     = 15

# Job and dashboard names
JOB_NAME       = f"[Course Validation] {COURSE_NAME}"
DASHBOARD_NAME = f"[Course Validation] {COURSE_NAME} Results Dashboard"

# Build the task list from the reusable course configuration above.
TASKS = []

if RUN_QA_CHECKER:
    TASKS.append({
        "task_key": "qa_content_checker",
        "course": "QA",
        "name": QA_TASK_NAME,
        "relative_path": QA_TASK_RELATIVE_PATH,
        "depends_on": [COURSE_TASKS[-1]["task_key"]] if COURSE_TASKS else [],
        "env_key": "qa_env",
    })

for task in COURSE_TASKS:
    TASKS.append({
        "task_key": task["task_key"],
        "course": COURSE_NAME,
        "name": task["name"],
        "relative_path": task["relative_path"],
        "depends_on": task.get("depends_on", []),
        "env_key": task["env_key"],
    })