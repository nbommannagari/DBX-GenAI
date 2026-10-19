# _lib/summary.py
#
# Renders an HTML summary card of the workspace assets created during setup.
# `build_setup_summary_html` is a pure string builder — no notebook globals
# (displayHTML/dbutils) — so it stays testable outside a notebook.
# `render_setup_summary` displays that HTML best-effort and is what the
# orchestrator calls at the end of setup; it silently no-ops outside a notebook
# (jobs, tests), so it's safe to call unconditionally.

from html import escape
from typing import Optional


def _link(url: str) -> str:
    """A clickable, new-tab link cell (URL is the visible text)."""
    safe = escape(url, quote=True)
    return (
        f'<a href="{safe}" target="_blank" rel="noopener noreferrer" '
        f'style="color:#2272B4;text-decoration:none;font-weight:600">{escape(url)} &#8599;</a>'
    )


def _status_badge(state: str) -> str:
    """Color-coded app status: green RUNNING, red CRASHED, amber otherwise."""
    s = (state or "").upper()
    color = {"RUNNING": "#00A972", "CRASHED": "#98102A"}.get(s, "#DC6222")
    return f'<span style="color:{color};font-weight:700">{escape(s or "UNKNOWN")}</span>'


def _copy_button(value: str) -> str:
    """A small "Copy" button that copies ``value`` to the clipboard.

    displayHTML renders inside a sandboxed iframe where the async Clipboard API is
    usually blocked, so we use a hidden <textarea> + document.execCommand('copy'),
    which works in that context. The raw value is JS-string-escaped and embedded
    in the onclick handler.
    """
    js_val = (
        value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "")
    )
    onclick = (
        "var t=document.createElement('textarea');"
        f"t.value='{js_val}';"
        "document.body.appendChild(t);t.select();"
        "try{{document.execCommand('copy');}}catch(e){{}}"
        "document.body.removeChild(t);"
        "var b=this;var o=b.innerText;b.innerText='Copied';"
        "setTimeout(function(){{b.innerText=o;}},1200);"
    )
    return (
        f'<button onclick="{escape(onclick, quote=True)}" title="Copy" '
        'style="flex:0 0 auto;padding:2px 9px;font-size:11px;font-family:inherit;'
        "color:#1B5162;background:#F9F7F4;border:1px solid #EEEDE9;border-radius:5px;"
        'cursor:pointer">Copy</button>'
    )


def build_setup_summary_html(env: dict, workspace_host: Optional[str] = None) -> str:
    """Build an HTML card listing the workspace assets created during setup.

    Parameters
    ----------
    env : dict
        The dict returned by ``setup_demo_environment``.
    workspace_host : str, optional
        Workspace URL (e.g. ``https://<ws>.cloud.databricks.com``). Used to build
        the clickable MLflow experiment link. When absent, the experiment row is
        shown as a plain id (or omitted if there's no id).

    Returns
    -------
    str
        HTML for ``displayHTML(...)``. Rows whose value is missing are skipped,
        so the card stays clean across configs (e.g. no volume, no app). Each row
        (except status) shows a Copy button that copies the raw value.
    """
    app = env.get("agent_app") or {}
    host = (workspace_host or "").rstrip("/")
    exp_id = app.get("experiment_id")
    exp_url = f"{host}/ml/experiments/{exp_id}" if (exp_id and host) else None

    # Registered system prompt (name + version + alias), shown as "name (v3 @champion)".
    prompt_name = app.get("system_prompt_name")
    prompt_version = app.get("system_prompt_version")
    prompt_alias = app.get("system_prompt_alias") or "champion"
    prompt_display = (
        f"{prompt_name} (v{prompt_version} @{prompt_alias})"
        if prompt_name and prompt_version
        else prompt_name
    )

    lb = env.get("lakebase_search") if isinstance(env.get("lakebase_search"), dict) else {}

    def _multi(label: str, items) -> Optional[tuple]:
        """A row for a list of assets: one value per line, count in the label."""
        vals = [str(x) for x in (items or []) if x]
        if not vals:
            return None
        lbl = f"{label} ({len(vals)})" if len(vals) > 1 else label
        cell = "<br>".join(escape(v) for v in vals)
        return (lbl, cell, "\n".join(vals))

    # (label, display_html, copy_value); None entries and empty-value rows are
    # dropped, so the card lists exactly what THIS config created. Ordered
    # roughly by setup sequence: environment → data → tools → agents → app.
    rows: list = [
        ("Catalog", None, env.get("catalog_name")),
        ("Schema", None, env.get("schema_name")),
        _multi("Volume", env.get("volumes")),
        _multi("Folder", env.get("folders")),
        _multi("File", env.get("files")),
        _multi("Table", env.get("tables")),
        _multi("SQL function", env.get("sql_functions")),
        _multi("Python function", env.get("python_functions")),
        ("PDF", None, env.get("pdf_path")),
        ("Genie space", None, env.get("genie_space_id")),
        ("Vector Search index", None, env.get("vs_index_name")),
        ("Model service", None, env.get("model_service_fqn")),
        ("KA assistant", None, env.get("ka_assistant_name")),
        ("Lakebase instance", None, env.get("lakebase_instance_name")),
        # Project name comes from either the agent-memory setup
        # (lakebase_autoscaling_project) or the lakebase_search step (project_id);
        # a course uses one or the other, so coalesce into one row.
        ("Lakebase project", None, env.get("lakebase_autoscaling_project") or lb.get("project_id")),
        ("Lakebase search table", None, lb.get("table_name")),
        _multi("Agent", env.get("agents_registered")),
        (
            "MLflow experiment",
            _link(exp_url) if exp_url else None,
            exp_url or env.get("mlflow_experiment_path") or (str(exp_id) if exp_id else None),
        ),
        ("Deployed endpoint", None, env.get("deployed_endpoint")),
        ("Agent app", None, app.get("app_name")),
        ("App URL", _link(app["app_url"]) if app.get("app_url") else None, app.get("app_url")),
        ("App status", _status_badge(app["app_status"]) if app.get("app_status") else None, None),
        ("System prompt", None, prompt_display),
    ]
    rows = [r for r in rows if r is not None]

    body = ""
    for label, display_html, copy_value in rows:
        # A row shows if it has display markup or a value to render.
        if display_html is None and (copy_value is None or copy_value == ""):
            continue
        cell = display_html if display_html is not None else escape(str(copy_value))
        button = _copy_button(str(copy_value)) if copy_value else ""
        body += (
            "<tr>"
            f'<td style="padding:8px 14px;white-space:nowrap;border-bottom:1px solid #EEEDE9;'
            f'font-weight:600;color:#0b2026;width:220px;vertical-align:middle">{escape(label)}</td>'
            f'<td style="padding:8px 14px;border-bottom:1px solid #EEEDE9;color:#0b2026;'
            f'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;gap:12px">'
            f'<span>{cell}</span>{button}</div></td>'
            "</tr>"
        )

    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;'
        'max-width:820px;margin:12px 0;border:1px solid #EEEDE9;border-radius:10px;overflow:hidden;'
        'box-shadow:0 2px 8px rgba(27,49,57,0.06)">'
        '<div style="background:#1B5162;color:#fff;padding:12px 16px;font-size:15px;font-weight:700">'
        "Workspace Assets Created</div>"
        '<table style="width:100%;border-collapse:collapse;font-size:14px">'
        f"{body}"
        "</table></div>"
    )


def _get_display_html():
    """Return the notebook's ``displayHTML`` callable, or None outside a notebook.

    ``displayHTML`` is a notebook REPL global not directly importable from a
    module. Both paths below are verified to work from within a _lib module on
    serverless; the import path is preferred. Returns None (rather than raising)
    when neither is available — e.g. in a job/test context — so callers no-op.
    """
    try:
        from dbruntime.display import displayHTML
        return displayHTML
    except Exception:
        pass
    try:
        import IPython
        ip = IPython.get_ipython()
        if ip is not None:
            fn = ip.user_ns.get("displayHTML")
            if callable(fn):
                return fn
    except Exception:
        pass
    return None


def render_setup_summary(env: dict, workspace_host: Optional[str] = None) -> None:
    """Build the summary card and display it, best-effort. Never raises.

    Called by ``setup_demo_environment`` at the end of setup so the summary is
    part of _lib itself — any notebook that runs setup gets the card without
    wiring up ``displayHTML`` by hand. Outside a notebook (jobs, test-runner,
    unit tests) there is no display, so this silently does nothing and setup
    proceeds normally.
    """
    try:
        html = build_setup_summary_html(env, workspace_host=workspace_host)
    except Exception as e:
        print(f"  (setup summary card skipped — could not build it: {e})")
        return
    display_fn = _get_display_html()
    if display_fn is None:
        return
    try:
        display_fn(html)
    except Exception:
        # A display failure must never fail setup.
        pass
