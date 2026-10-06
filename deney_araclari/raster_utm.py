#!/usr/bin/env python
"""Bir RGB raster'i benchmark izgarasina yeniden ornekler: UTM + 0.298582 m/px + alfa bandi.

    python raster_utm.py kaynak.tif hedef.tif [--epsg 32636] [--res 0.298582141739] [--referans diger.tif]

Benchmark (geospatial_model_benchmark.py) metre tabanli projeksiyon ve iki raster'da ayni
GSD ister; Urgup rasterlari 0.298582 m/px, EPSG:32636, 4 bant (RGB + alfa). Web Mercator
(EPSG:3857) kaynaklar bu araclas donusturulur. --referans verilirse hedef izgara (transform,
boyut) o raster'dan alinir; boylece iki kaynak piksel piksel ayni izgaraya oturur.
Alfa bandi gecerli (kaynaktan gelen) pikselleri isaretler; benchmark alfa=0 iceren karolari atlar.
"""
from __future__ import annotations

import argparse

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--epsg", type=int, default=32636)
    ap.add_argument("--res", type=float, default=0.298582141739)
    ap.add_argument("--referans", default="")
    args = ap.parse_args()
    dst_crs = CRS.from_epsg(args.epsg)
    with rasterio.open(args.src) as src:
        if args.referans:
            with rasterio.open(args.referans) as ref:
                transform, width, height = ref.transform, ref.width, ref.height
                dst_crs = ref.crs
        else:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=args.res
            )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height, count=4,
                       dtype="uint8", nodata=None, compress="LZW", tiled=True, BIGTIFF="IF_SAFER",
                       photometric="RGB")
        with rasterio.open(args.dst, "w", **profile) as dst:
            valid = np.zeros((height, width), dtype=np.uint8)
            for band in (1, 2, 3):
                out = np.zeros((height, width), dtype=np.uint8)
                reproject(source=rasterio.band(src, band), destination=out, src_transform=src.transform,
                          src_crs=src.crs, dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.bilinear)
                dst.write(out, band)
            ones = np.full((src.height, src.width), 255, dtype=np.uint8)
            reproject(source=ones, destination=valid, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest)
            dst.write(valid, 4)
    print(f"{args.dst}: {width}x{height} px, {dst_crs}, gecerli oran {float((valid > 0).mean()):.3f}")


if __name__ == "__main__":
    main()
