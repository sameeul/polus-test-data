"""Generate a small OME-Zarr (v2, NGFF 0.4) test image: bfio/small_5d.zarr(.zip).

Shape (T,C,Z,Y,X) = (1,4,1,300,302), uint8, chunks (1,1,1,128,128) so edge
chunks are partial. Seeded, so the output (and per-channel sums) are stable.
Usage: python make_small_zarr.py <out_dir>
"""
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import zarr

SHAPE = (1, 4, 1, 300, 302)
CHUNKS = (1, 1, 1, 128, 128)

out = Path(sys.argv[1])
store = out / "small_5d.zarr"
if store.exists():
    shutil.rmtree(store)

data = np.random.default_rng(0).integers(0, 256, SHAPE, dtype=np.uint8)

g = zarr.open_group(store, mode="w", zarr_format=2)
g.attrs["multiscales"] = [
    {
        "version": "0.4",
        "name": "small_5d",
        "axes": [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 1, 1, 1, 1]}
                ],
            }
        ],
    }
]
a = g.create_array("0", shape=SHAPE, chunks=CHUNKS, dtype=np.uint8)
a[:] = data

# Zip with deterministic ordering/timestamps; the zarr dir itself is the zip root.
zpath = out / "small_5d.zarr.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(store.rglob("*")):
        if p.is_file():
            zi = zipfile.ZipInfo(str(p.relative_to(out)), (1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, p.read_bytes())

print("sum all:", int(data.sum()))
for c in range(SHAPE[1]):
    print(f"sum c{c}:", int(data[0, c].sum()))
print("zip bytes:", zpath.stat().st_size)
