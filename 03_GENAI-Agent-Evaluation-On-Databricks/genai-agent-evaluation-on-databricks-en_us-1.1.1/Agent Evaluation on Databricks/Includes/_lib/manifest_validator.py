import json
import yaml
from pathlib import Path
from typing import Optional


def validate_manifest_alignment(
    course_name: str,
    vocareum_share_name: Optional[str] = None,
    manifest_path: Optional[str | Path] = None,
    medallion_labs: Optional[bool] = False,
) -> bool:
    """
    Validate that config values align with the manifest.

    Checks (standard labs)
    ----------------------
    1. course_name matches a lab 'name' in the manifest labs list.
    2. vocareum_share_name (if provided) matches a dataset 'name' for that lab.

    Checks (medallion_labs=True)
    ----------------------------
    1. Manifest file exists and is valid JSON.
    2. Manifest contains a top-level 'course' field.
    3. 'labs.lti' is set to True.

    Raises ValueError with a descriptive message on any mismatch.
    Returns True on success.
    """
    # --- Resolve manifest path ---
    if manifest_path is None:
        target = ".binder/manifest.json" if medallion_labs else "manifest.yaml"
        search = Path(__file__).resolve().parent
        while search != search.parent:
            candidate = search / target
            if candidate.exists():
                manifest_path = candidate
                break
            search = search.parent
        if manifest_path is None:
            raise FileNotFoundError(
                f"Manifest not found: walked from {Path(__file__).resolve().parent} "
                f"to filesystem root looking for '{target}'"
            )

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # --- Parse manifest (JSON for medallion labs, YAML otherwise) ---
    with open(manifest_path) as f:
        if medallion_labs:
            manifest = json.load(f)
        else:
            manifest = yaml.safe_load(f)

    # --- Medallion Labs validation branch ---
    if medallion_labs:
        return _validate_medallion_manifest(manifest, manifest_path)

    # --- Standard Labs validation ---
    return _validate_standard_manifest(manifest, course_name, vocareum_share_name)


def _validate_medallion_manifest(manifest: dict, manifest_path: Path) -> bool:
    """Validate structure of a Medallion Labs manifest.json."""

    # 1. Top-level 'course' field must be present
    course = manifest.get("course")
    if not course:
        raise ValueError(
            f"Medallion Labs manifest at '{manifest_path}' is missing a top-level 'course' field."
        )
    print(f"  [OK] Medallion Labs manifest found for course: '{course}'")

    # 2. labs.lti must be True
    labs = manifest.get("labs", {})
    if not labs.get("lti"):
        raise ValueError(
            f"Medallion Labs manifest for '{course}' does not have 'labs.lti' set to true.\n"
            f"  Found: {labs.get('lti')!r}"
        )
    print(f"  [OK] 'labs.lti' is True.")

    return True


def _validate_standard_manifest(
    manifest: dict,
    course_name: str,
    vocareum_share_name: Optional[str],
) -> bool:
    """Validate a standard YAML manifest against course_name and optional dataset."""

    # 1. Course name check -- try top-level 'course' field first, then labs list
    top_level_course = manifest.get("course", "")
    if top_level_course == course_name:
        print(f"  [OK] course_name '{course_name}' matches manifest top-level course.")
        # Use defaults config for dataset validation
        lab = manifest.get("labs", {}).get("defaults", {})
    else:
        labs = manifest.get("labs", {}).get("list", [])
        matching = [l for l in labs if l.get("name", "") == course_name or l.get("course", "") == course_name]
        if not matching:
            available = [top_level_course] if top_level_course else []
            available += [l.get("name", "") for l in labs]
            raise ValueError(
                f"course_name '{course_name}' not found in manifest.\n"
                f"  Available names: {available}"
            )
        lab = matching[0]
        print(f"  [OK] course_name '{course_name}' found in manifest.")

    # 2. Vocareum dataset check
    if vocareum_share_name:
        datasets = (lab.get("config") or {}).get("datasets") or []
        dataset_names = [d.get("name", "") for d in datasets]
        if vocareum_share_name not in dataset_names:
            raise ValueError(
                f"vocareum_share_name '{vocareum_share_name}' not found in manifest datasets for '{course_name}'.\n"
                f"  Available datasets: {dataset_names if dataset_names else '(none configured)'}\n"
                f"  Add a 'datasets' entry to the '{course_name}' config block in manifest.yaml."
            )
        print(f"  [OK] vocareum_share_name '{vocareum_share_name}' found in manifest datasets.")

    return True