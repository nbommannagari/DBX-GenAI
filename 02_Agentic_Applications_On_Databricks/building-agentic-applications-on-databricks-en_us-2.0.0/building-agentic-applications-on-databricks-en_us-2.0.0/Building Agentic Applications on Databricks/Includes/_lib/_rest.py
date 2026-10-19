# _lib/_rest.py
#
# Shared HTTP plumbing for the Databricks REST calls scattered across _lib
# (genie_creation, genie_deletion, agent_bricks_manager, model_service_config).
# Centralizes auth-header construction, request timeouts, and retry/backoff so
# each caller stops re-implementing them (and so timeouts can't be forgotten).

import time
import requests


def auth_headers(w, content_type: str | None = "application/json") -> dict:
    """Auth headers from a WorkspaceClient (``w.config.authenticate()``).

    Pass ``content_type=None`` for verbs without a body (GET/DELETE).
    """
    headers = dict(w.config.authenticate())
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def notebook_auth_headers() -> tuple[str, dict]:
    """Resolve (host, headers) from the notebook context (dbutils).

    Used by callers that run inside a Databricks notebook and authenticate with
    the notebook's own API token rather than a WorkspaceClient. ``dbutils`` is
    imported lazily so importing this module never requires a notebook runtime.
    """
    from databricks.sdk.runtime import dbutils

    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    host = ctx.apiUrl().get()
    token = ctx.apiToken().get()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return host, headers


def request_with_retry(
    method: str,
    url: str,
    headers: dict,
    *,
    timeout: float = 60,
    ok_statuses: tuple = (200, 201, 204),
    max_attempts: int = 8,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    **kwargs,
) -> requests.Response:
    """HTTP request with a timeout and exponential backoff on 429/5xx.

    Returns the response for any status in ``ok_statuses``. Retries (bounded by
    ``max_attempts``) on 429 and 5xx. For any other status it calls
    ``raise_for_status()``; if that does not raise (an unexpected non-error
    status outside ``ok_statuses``), the response is returned rather than
    looping — the caller decides what to do with it.
    """
    attempt = 0
    while True:
        resp = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
        if resp.status_code in ok_statuses:
            return resp
        if (resp.status_code == 429 or resp.status_code >= 500) and attempt < max_attempts:
            attempt += 1
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            time.sleep(delay + 0.1 * delay)
            continue
        resp.raise_for_status()
        return resp
