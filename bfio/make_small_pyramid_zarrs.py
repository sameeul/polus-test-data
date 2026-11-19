"""Generate small 2-level OME-Zarr test images for bfio tests.

Outputs (zip root is the zarr dir):
  bfio/small_5d_pyramid.zarr(.zip)  zarr v2, NGFF 0.4
  bfio/small_5d_v3.ome.zarr(.zip)   zarr v3, NGFF 0.5 (metadata under attrs["ome"])

Both: shape (T,C,Z,Y,X) = (1,4,1,300,302) uint8 at level "0", and
(1,4,1,150,151) at level "1" (data[..., ::2, ::2]). chunks (1,1,1,128,128).
Seeded, so output is stable. Requires zarr>=3.
Usage: python make_small_pyramid_zarrs.py <out_dir>
"""
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import zarr

SHAPE = (1, 4, 1, 300, 302)
CHUNKS = (1, 1, 1, 128, 128)
AXES = [
    {"name": "t", "type": "time"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]


def multiscales(name, version=None):
    ms = {
        "name": name,
        "axes": AXES,
        "datasets": [
            {
                "path": str(i),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 1, 1, 2**i, 2**i]}
                ],
            }
            for i in range(2)
        ],
    }
    if version:
        ms["version"] = version
    return [ms]


def make(out, name, zarr_format, data):
    store = out / name
    if store.exists():
        shutil.rmtree(store)
    g = zarr.open_group(store, mode="w", zarr_format=zarr_format)
    if zarr_format == 2:
        g.attrs["multiscales"] = multiscales(name, "0.4")
    else:
        g.attrs["ome"] = {"version": "0.5", "multiscales": multiscales(name)}
    for i, d in enumerate([data, data[..., ::2, ::2]]):
        a = g.create_array(str(i), shape=d.shape, chunks=CHUNKS, dtype=np.uint8)
        a[:] = d

    # Deterministic zip (sorted, fixed timestamps); the zarr dir is the zip root.
    zpath = out / f"{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(store.rglob("*")):
            if p.is_file():
                zi = zipfile.ZipInfo(str(p.relative_to(out)), (1980, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, p.read_bytes())
    print(name, "zip bytes:", zpath.stat().st_size)


out = Path(sys.argv[1])
data = np.random.default_rng(1).integers(0, 256, SHAPE, dtype=np.uint8)
make(out, "small_5d_pyramid.zarr", 2, data)
make(out, "small_5d_v3.ome.zarr", 3, data)
l0, l1 = data, data[..., ::2, ::2]
print("level0 sum:", int(l0.sum()), "shape:", l0.shape)
print("level1 sum:", int(l1.sum()), "shape:", l1.shape)
for c in range(SHAPE[1]):
    print(f"c{c} level0 sum:", int(l0[0, c].sum()))
