from pathlib import Path
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def create_volume(
    catalog_name: str,
    schema_name: str,
    volume_name: str = "demo_vol",
) -> Path:
    """
    Create a Unity Catalog volume if it doesn't exist.

    Parameters
    ----------
    catalog_name : str
        The UC catalog name.
    schema_name : str
        The UC schema name.
    volume_name : str
        The volume name. Default: 'demo_vol'.

    Returns
    -------
    Path
        Path to the volume: /Volumes/{catalog}/{schema}/{volume}
    """
    spark.sql(f"USE CATALOG {catalog_name}")
    spark.sql(f"USE SCHEMA {schema_name}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_name}")

    volume_path = Path(f"/Volumes/{catalog_name}/{schema_name}/{volume_name}")
    print(f"  Volume ready: {volume_path}")
    return volume_path


def copy_from_folder_to_vol(
    source_dir: str | Path,
    volume_path: str | Path,
    extensions: tuple | None = None,
) -> list[Path]:
    """
    Copy files from a local folder into a UC volume path.

    Parameters
    ----------
    source_dir : str | Path
        Source directory to copy from.
    volume_path : str | Path
        Destination UC volume path.
    extensions : tuple | None
        Optional tuple of file extensions to filter (e.g., ('.json', '.csv')).
        If None, all files are copied.

    Returns
    -------
    list[Path]
        List of destination paths for copied files.
    """
    source = Path(source_dir)
    dest = Path(volume_path)

    if not source.exists():
        print(f"  Warning: Source folder not found: {source}")
        return []

    dest.mkdir(parents=True, exist_ok=True)

    copied = []
    for file_path in sorted(source.iterdir()):
        if not file_path.is_file():
            continue
        if extensions and file_path.suffix.lower() not in extensions:
            continue

        dest_file = dest / file_path.name
        dest_file.write_bytes(file_path.read_bytes())
        copied.append(dest_file)
        print(f"  Copied: {file_path.name} → {dest_file}")

    return copied


def copy_volume_assets(
    source_volume: str,
    items: list,
    catalog_name: str,
    schema_name: str,
    default_volume_path: Path = None,
) -> dict:
    """
    Copy files or folders from a source volume into user volumes.

    A richer alternative to ``copy_from_folder_to_vol`` for configs that need
    to fan assets out across multiple destination volumes. Gated in the
    orchestrator by the ``volume_copy`` config section (whereas
    ``copy_from_folder_to_vol`` is driven by ``volume_sources``); both remain
    available so either config style works.

    Each item in *items* is a dict with:
      - source      (required): relative path under source_volume (file or folder).
      - dest        (optional): relative path inside the destination volume.
                                Defaults to the source value, preserving structure.
      - dest_volume (optional): name of the destination volume.
                                The volume is created if it doesn't exist.
                                When omitted, falls back to *default_volume_path*.

    Folders are copied recursively.  Files are copied individually.

    Parameters
    ----------
    source_volume : str
        Base path of the source volume (e.g. /Volumes/dbacademy/ka_all/vol_all).
    items : list[dict]
        Items to copy.
    catalog_name : str
        User catalog — used to create volumes when dest_volume is specified.
    schema_name : str
        User schema — used to create volumes when dest_volume is specified.
    default_volume_path : Path, optional
        Fallback destination volume path when an item omits dest_volume.

    Returns
    -------
    tuple[dict, dict]
        A pair of (copied_files, created_volumes):
        - copied_files:    relative source path → absolute destination path
        - created_volumes: volume name → volume Path for every volume
                           created via dest_volume during this call.
    """
    import os
    import shutil

    # Cache volumes we've already created this run
    _created_volumes: dict[str, Path] = {}

    def _resolve_dest_volume(item: dict) -> Path:
        vol_name = item.get("dest_volume")
        if not vol_name:
            if default_volume_path is None:
                raise ValueError(
                    f"Item {item} has no dest_volume and no default volume was created "
                    "(catalog.volume_name is not set in config)."
                )
            return default_volume_path
        if vol_name not in _created_volumes:
            _created_volumes[vol_name] = create_volume(catalog_name, schema_name, vol_name)
        return _created_volumes[vol_name]

    copied = {}
    for item in items:
        source_rel = item["source"]
        dest_rel = item.get("dest", source_rel)
        dest_vol = _resolve_dest_volume(item)
        source_path = Path(f"{source_volume}/{source_rel}")
        dest_path = dest_vol / dest_rel

        if source_path.is_dir():
            # Folder copy — recursively walk and copy every file
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(str(source_path), str(dest_path))
            for root, _, files in os.walk(dest_path):
                for fname in files:
                    abs_dest = Path(root) / fname
                    rel_src = str(Path(source_rel) / abs_dest.relative_to(dest_path))
                    copied[rel_src] = str(abs_dest)
            print(f"  Copied folder: {source_rel} -> {dest_path}  ({len([f for f in dest_path.rglob('*') if f.is_file()])} files)")
        else:
            # Single-file copy
            os.makedirs(dest_path.parent, exist_ok=True)
            shutil.copy2(str(source_path), str(dest_path))
            copied[source_rel] = str(dest_path)
            print(f"  Copied file: {source_rel} -> {dest_path}")

    return copied, dict(_created_volumes)
