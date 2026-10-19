# _lib/model_service_config.py
#
# Config-driven Unity Gateway model service setup for GenAI courses.
# Creates a model service in the user's catalog/schema that routes
# to a foundation model endpoint via Unity Gateway.

import requests
from databricks.sdk import WorkspaceClient

from ._rest import auth_headers

# REST behavior this module relies on (verified against the preview API,
# /api/2.1/unity-catalog/model-services):
#   POST (new)                -> 200
#   POST (duplicate)          -> 409, error_code ALREADY_EXISTS
#   GET   {fqn} (exists)      -> 200, full config.routing.destinations[].pay_per_token_config.model
#   GET   {fqn} (absent)      -> 404, error_code NOT_FOUND
#   PATCH {fqn}?update_mask=comment        -> 200
#   PATCH {fqn}?update_mask=config.routing -> 400 ("not yet supported by UpdateModelService")
#   DELETE {fqn}              -> 200
# The key constraint: routing (i.e. which model a service is served_by) CANNOT
# be changed in place — the only way to re-point an existing service is to
# delete and recreate it.

_API_BASE = "/api/2.1/unity-catalog/model-services"


def _model_ref(served_by: str) -> str:
    """Foundation-model reference used in the routing destination."""
    return f"models/system.ai.{served_by}"


def _routing_body(served_by: str, comment: str) -> dict:
    return {
        "comment": comment or f"Routes to {served_by} for agent training",
        "config": {
            "routing": {
                "destinations": [
                    {
                        "name": "primary",
                        "destination_type": "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL",
                        "pay_per_token_config": {"model": _model_ref(served_by)},
                        "traffic_percentage": 100,
                    }
                ]
            }
        },
    }


def _current_model(service: dict) -> str | None:
    """Extract the served model ref from a GET response, or None if unset."""
    dests = (service.get("config", {}).get("routing", {}) or {}).get("destinations", [])
    for d in dests:
        model = (d.get("pay_per_token_config") or {}).get("model")
        if model:
            return model
    return None


def _get_service(w: WorkspaceClient, fqn: str) -> dict | None:
    """Return the model service dict, or None if it does not exist (404)."""
    resp = requests.get(f"{w.config.host}{_API_BASE}/{fqn}", headers=auth_headers(w), timeout=60)
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code == 404:
        return None
    raise RuntimeError(
        f"Failed to read model service {fqn}: HTTP {resp.status_code} — {resp.text}"
    )


def _create_service(w: WorkspaceClient, catalog_name: str, schema_name: str,
                    service_name: str, served_by: str, comment: str) -> None:
    fqn = f"{catalog_name}.{schema_name}.{service_name}"
    resp = requests.post(
        f"{w.config.host}{_API_BASE}",
        params={"parent": f"schemas/{catalog_name}.{schema_name}",
                "model_service_id": service_name},
        json=_routing_body(served_by, comment),
        headers=auth_headers(w),
        timeout=60,
    )
    if resp.status_code == 200:
        print(f"  Model service created: {fqn} → {served_by}")
    elif resp.status_code == 409 or "ALREADY_EXISTS" in resp.text:
        # Race: created between our GET and POST. The caller's GET-first path
        # already handles the steady state; treat this as benign.
        print(f"  Model service already exists: {fqn}")
    else:
        raise RuntimeError(
            f"Failed to create model service {fqn}: "
            f"HTTP {resp.status_code} — {resp.text}"
        )


def _delete_service(w: WorkspaceClient, fqn: str) -> None:
    resp = requests.delete(f"{w.config.host}{_API_BASE}/{fqn}", headers=auth_headers(w), timeout=60)
    if resp.status_code not in (200, 404):
        raise RuntimeError(
            f"Failed to delete model service {fqn}: HTTP {resp.status_code} — {resp.text}"
        )


def _sync_comment(w: WorkspaceClient, fqn: str, comment: str) -> None:
    """Best-effort comment update (the one field UpdateModelService supports)."""
    resp = requests.patch(
        f"{w.config.host}{_API_BASE}/{fqn}",
        params={"update_mask": "comment"},
        json={"comment": comment},
        headers=auth_headers(w),
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"  Note: could not update comment on {fqn} "
              f"(HTTP {resp.status_code}); leaving as-is.")


def setup_model_service(
    catalog_name: str,
    schema_name: str,
    service_name: str,
    served_by: str,
    comment: str = "",
    recreate_on_drift: bool = False,
) -> str:
    """Ensure a Unity Gateway model service exists and routes to ``served_by``.

    Idempotent via a GET-first check (not a blind create):

    - **Absent** → create it.
    - **Exists, routes to the same model** → no-op (optionally sync the comment).
    - **Exists, routes to a different model** → routing is not updatable in
      place (the API rejects ``update_mask=config.routing``). By default this
      raises so the drift is not silently ignored. Pass
      ``recreate_on_drift=True`` to delete and recreate instead — destructive:
      the service gets a new id and any EXECUTE grants on it are dropped.

    Parameters
    ----------
    catalog_name, schema_name : str
        Target UC catalog / schema (must already exist).
    service_name : str
        Name for the model service.
    served_by : str
        Foundation model to route to (e.g. ``'databricks-gpt-5-2'``); resolved
        to ``models/system.ai.<served_by>``.
    comment : str, optional
        Description for the model service.
    recreate_on_drift : bool, optional
        When the service exists but routes to a different model, delete and
        recreate it instead of raising. Default False.

    Returns
    -------
    str
        Fully qualified model service name (``catalog.schema.service_name``).
    """
    w = WorkspaceClient()
    fqn = f"{catalog_name}.{schema_name}.{service_name}"
    desired = _model_ref(served_by)

    existing = _get_service(w, fqn)

    if existing is None:
        _create_service(w, catalog_name, schema_name, service_name, served_by, comment)
        return fqn

    current = _current_model(existing)
    if current == desired:
        print(f"  Model service already exists and matches: {fqn} → {served_by}")
        if comment and existing.get("comment") != comment:
            _sync_comment(w, fqn, comment)
        return fqn

    # Drift: exists but routes elsewhere, and routing can't be patched.
    if recreate_on_drift:
        print(f"  Model service {fqn} routes to '{current}', want '{desired}'. "
              f"recreate_on_drift=True → deleting and recreating.")
        _delete_service(w, fqn)
        _create_service(w, catalog_name, schema_name, service_name, served_by, comment)
        return fqn

    raise RuntimeError(
        f"Model service {fqn} already exists but routes to '{current}', "
        f"not the configured '{desired}'. Unity Gateway does not support "
        f"updating a model service's routing in place. Either drop the service "
        f"and re-run, or set model_service.recreate_on_drift: true in the config "
        f"to have setup delete and recreate it (destructive: new id, grants dropped)."
    )
