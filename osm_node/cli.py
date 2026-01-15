"""Command-line interface for osm-node."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import click

from osm_node.handler import extract_features
from osm_node.index import RoaringIndex, SortedU64Index
from osm_node.schema import get_feature_specs
from osm_node.utils import sort_and_unique_chunks
from osm_node.writers import RoaringWriter, SortedU64Writer


@click.group()
@click.version_option()
def main():
    """OSM Node - Extract and index control nodes from OSM PBF files."""
    pass


@main.command()
@click.option(
    "--pbf",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the OSM PBF file",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for index files",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["u64", "roar", "both"]),
    default="both",
    help="Index format to generate (default: both)",
)
@click.option(
    "--features",
    default="signals,stops,calming",
    help="Comma-separated list of features to extract (default: signals,stops,calming)",
)
@click.option(
    "--tmp",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Temporary directory for sorting (default: system temp)",
)
@click.option(
    "--flush-threshold",
    type=int,
    default=100_000,
    help="Number of IDs per feature before flushing to disk (default: 100000)",
)
def build(
    pbf: Path,
    out: Path,
    fmt: str,
    features: str,
    tmp: Path | None,
    flush_threshold: int,
):
    """Build indices from a PBF file.

    Examples:

        osm-node build --pbf region.osm.pbf --out ./indices

        osm-node build --pbf region.osm.pbf --out ./indices --format roar --features signals,stops
    """
    # Parse feature list
    feature_names = [f.strip() for f in features.split(",") if f.strip()]

    try:
        feature_specs = get_feature_specs(feature_names)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--features")

    click.echo(f"Extracting features: {', '.join(feature_specs.keys())}")
    click.echo(f"Input: {pbf}")
    click.echo(f"Output: {out}")
    click.echo(f"Format: {fmt}")

    # Create output directory
    out.mkdir(parents=True, exist_ok=True)

    # Create temp directory
    if tmp is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="osm_node_"))
        cleanup_tmp = True
    else:
        tmp_dir = tmp
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cleanup_tmp = False

    try:
        # Extract features from PBF
        click.echo("Scanning PBF file...")
        handler, chunk_paths = extract_features(
            pbf,
            feature_specs,
            tmp_dir,
            flush_threshold,
        )

        stats = handler.get_statistics()
        click.echo(f"Processed {stats['nodes_processed']:,} nodes")
        click.echo(f"Matched {stats['nodes_matched']:,} nodes")

        # Sort and write indices
        click.echo("Sorting and writing indices...")

        for feature_name in feature_specs:
            chunks = chunk_paths.get(feature_name, [])
            raw_count = stats["feature_counts"].get(feature_name, 0)

            if raw_count == 0:
                click.echo(f"  {feature_name}: 0 nodes (empty)")
                # Write empty files
                if fmt in ("u64", "both"):
                    (out / f"{feature_name}.u64").write_bytes(b"")
                if fmt in ("roar", "both"):
                    from pyroaring import BitMap

                    (out / f"{feature_name}.roar").write_bytes(BitMap().serialize())
                continue

            # Sort and unique the chunks
            sorted_path = tmp_dir / f"{feature_name}_sorted.u64"
            unique_count = sort_and_unique_chunks(chunks, sorted_path, tmp_dir)

            click.echo(f"  {feature_name}: {unique_count:,} unique nodes")

            # Read sorted array
            import numpy as np

            sorted_ids = np.fromfile(sorted_path, dtype="<u8")

            # Write in requested format(s)
            if fmt in ("u64", "both"):
                writer = SortedU64Writer(out)
                writer.write(feature_name, sorted_ids)
                click.echo(f"    -> {feature_name}.u64")

            if fmt in ("roar", "both"):
                writer = RoaringWriter(out)
                writer.write(feature_name, sorted_ids)
                click.echo(f"    -> {feature_name}.roar")

            # Clean up sorted temp file
            sorted_path.unlink(missing_ok=True)

        click.echo("Done!")

    finally:
        # Clean up temp directory
        if cleanup_tmp and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


@main.command()
@click.option(
    "--dir",
    "index_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing index files",
)
def inspect(index_dir: Path):
    """Inspect index files in a directory.

    Shows statistics about the indices including feature names and sizes.
    """
    click.echo(f"Inspecting indices in: {index_dir}")
    click.echo()

    # Check for u64 files
    u64_files = list(index_dir.glob("*.u64"))
    if u64_files:
        click.echo("Sorted uint64 indices (.u64):")
        idx = SortedU64Index.load_dir(index_dir)
        stats = idx.get_statistics()
        for feature, info in sorted(stats.items()):
            size_kb = info["size_bytes"] / 1024
            click.echo(f"  {feature}: {info['count']:,} nodes ({size_kb:.1f} KB)")
        click.echo()

    # Check for roar files
    roar_files = list(index_dir.glob("*.roar"))
    if roar_files:
        click.echo("Roaring bitmap indices (.roar):")
        idx = RoaringIndex.load_dir(index_dir)
        stats = idx.get_statistics()
        for feature, info in sorted(stats.items()):
            mode = "64-bit" if info["is_64bit"] else "32-bit"
            click.echo(f"  {feature}: {info['count']:,} nodes ({mode})")
        click.echo()

    if not u64_files and not roar_files:
        click.echo("No index files found.")


if __name__ == "__main__":
    main()
