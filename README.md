# osm-node

Extract and index control nodes (traffic signals, stops, traffic calming) from OpenStreetMap PBF files.

## Features

- **Streaming extraction**: Uses PyOsmium to stream large PBF files with minimal RAM
- **Dual index formats**:
  - Sorted uint64 files (portable, numpy memmap for near-zero RAM loading)
  - Roaring bitmaps (fastest membership tests, very compact)
- **CLI and library API**: Build indices from command line, query from Python

## Installation

```bash
pip install osm-node
```

Or for development:

```bash
git clone https://github.com/fedor-shi/osm-node.git
cd osm-node
pip install -e ".[dev]"
```

## Quick Start

### Build indices from PBF

```bash
# Build both u64 and roaring indices
osm-node build --pbf germany-latest.osm.pbf --out ./indices --format both

# Build only roaring indices for specific features
osm-node build --pbf region.osm.pbf --out ./indices --format roar --features signals,stops

# Inspect existing indices
osm-node inspect --dir ./indices
```

### Use in Python

```python
from osm_node.index import SortedU64Index, RoaringIndex

# Load indices
idx = SortedU64Index.load_dir("./indices")
# or
idx = RoaringIndex.load_dir("./indices")

# Count features along a route (node IDs from OSRM annotation)
route_nodes = [123456, 789012, 345678, ...]
signal_count = idx.count("signals", route_nodes)
stop_count = idx.count("stops", route_nodes)

# Check individual node
if idx.contains("signals", 123456):
    print("Node is a traffic signal")
```

## Indexed Features

| Feature | Tag Pattern |
|---------|-------------|
| `signals` | `highway=traffic_signals` OR `crossing=traffic_signals` |
| `stops` | `highway=stop` |
| `calming` | `traffic_calming=*` |

## Index Formats

### Sorted uint64 (`.u64`)

- Raw little-endian uint64 array of node IDs
- Can be memory-mapped with numpy (near-zero RAM)
- Membership via binary search (O(log n))

### Roaring bitmap (`.roar`)

- Compressed bitmap serialization
- Very compact for sparse sets
- O(1) membership tests

## CLI Reference

```
osm-node build --pbf FILE --out DIR [OPTIONS]

Options:
  --format {u64,roar,both}   Index format to generate (default: both)
  --features TEXT            Comma-separated feature list (default: signals,stops,calming)
  --tmp DIR                  Temporary directory for sorting (default: system temp)

osm-node inspect --dir DIR

  Show statistics about indices in a directory.
```

## License

MIT
