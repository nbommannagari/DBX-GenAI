from ._rest import notebook_auth_headers, request_with_retry


def list_spaces_by_title(space_title: str) -> list[dict]:
    """Return all Genie spaces whose title matches (case-insensitive)."""
    host, headers = notebook_auth_headers()
    resp = request_with_retry("GET", f"{host}/api/2.0/genie/spaces", headers=headers)
    spaces = resp.json().get("spaces", [])
    return [s for s in spaces if s.get("title", "").lower() == space_title.lower()]


def delete_genie_space(space_id: str) -> None:
    """Delete a Genie space by ID."""
    host, headers = notebook_auth_headers()
    request_with_retry("DELETE", f"{host}/api/2.0/genie/spaces/{space_id}", headers=headers)
    print(f"  Genie space deleted: {space_id}")


def delete_genie_space_by_title(title: str) -> None:
    """
    Find and delete all Genie spaces matching title (case-insensitive).

    Deletes all matches so the subsequent create always starts from a clean slate.
    Silently skips if no match is found.
    """
    spaces = list_spaces_by_title(title)
    if not spaces:
        print(f"  No existing Genie space found with title '{title}' — skipping deletion")
        return
    for space in spaces:
        delete_genie_space(space["space_id"])
