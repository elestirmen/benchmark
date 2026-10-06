#!/usr/bin/env python
"""Kaynaklar-arasi eslesme marjini olcen hizli on eleme.

Pahali benchmark'tan (checkpoint basina ~50 dk) once her adayi CPU'da bir
dakikada eler. Olculen nicelik, benchmark'in success_25m sonucunu belirleyen
seydir: ayni yerin iki kaynaktaki goruntusu, YANLIS yerlerden ne kadar daha
ayirt edilebilir sekilde benzesiyor.

Her merkez icin gercek konum NCC'si (s+) ve en iyi yanlis konum NCC'si (s-)
olculur. Uc istatistik uretilir:

    oran  = ortalama(s-) / ortalama(s+)      <- BIRINCIL olcut, kucuk olan iyi
    marj  = ortalama(s+ - s-)                <- ikincil, olcek bagimli
    z     = (ort(s+) - ort(s-)) / std(s-)    <- tepe/yan-lob, benchmark'in PSR'sine yakin

Birincil olcut ORAN'dir, marj degildir. Sebebi olculdu: gercek benchmark
~500 km2 icinde milyonlarca aday konum arar, dolayisiyla belirleyici olan
mutlak marj degil, yanlis tepenin gercek tepeye ne kadar yaklastigidir.
Yuksek s+ degeri 20 adayli bir testte avantaj saglar ama milyonlarca adayda
yuksek s- oranini telafi etmez.

Dogrulama (20 kirpma, seed 42, Urgup cevresi Google/Bing cifti):

    aday                              oran    marj    benchmark global (clean)
    RAW                               0.62   +0.115   g2b 0.616 / b2g 0.367
    top_modeller tpu_f48_k4 sigmoid   0.32   +0.146   g2b 0.691 / b2g 0.703
    top_modeller model_k_32_3 sigmoid 0.37   +0.108   g2b 0.652 / b2g 0.694
    models_pipeline classic_f32 tanh  2.87   -0.044   0.011

Oran, benchmark siralamasiyla birebir uyusur; marj uyusmaz (model_k_32_3
marjda RAW'in altinda kalir ama benchmark'ta iki katina yakin gecer). Vekil
olcut olarak oran kullanilmalidir.

Kullanim:

    python margin_gate.py --raw-only
    python margin_gate.py --model-dir top_modeller
    python margin_gate.py --model top_modeller/model_a.h5 --model models_pipeline/x/y.h5
    python margin_gate.py --model-dir top_modeller --samples 200 --aoi train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

# Bu ortamda HDF5, TensorFlow'dan once yuklenmelidir.
try:  # pragma: no cover - ortam bagimli
    import h5py  # noqa: F401
except ImportError:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

QUERY_RASTER = ROOT / "urgup_cevresi_harita" / "urgup_cevresi_gmap_utm_30cm.tif"
MAP_RASTER = ROOT / "urgup_cevresi_harita" / "urgup_cevresi_bingmap_utm_30cm.tif"

MODEL_SUFFIXES = {".h5", ".hdf5", ".keras"}

# Benchmark ile ayni geometri: 544 karo, 16 px kenar kirpma -> 512 sablon.
TILE_SIZE = 544
CROP_BORDER = 16

# Benchmark'in sablon reddi ile ayni esik (geospatial_model_benchmark.py:2638).
MIN_TEMPLATE_STD = 2.0

# Ortak alan, kenarlardan bir tam karo icerlek.
SHARED_EAST = (661500.0, 679500.0)
SHARED_NORTH = (4256000.0, 4279500.0)

# Cografi bolme: guney %25 test, arada 1000 m tampon. Egitim marji train
# bolgesinde olculmelidir, yoksa test alanina sizinti olur.
TEST_NORTH_MAX = 4262375.0
TRAIN_NORTH_MIN = 4263375.0

# Dusuk dokulu kirpmalar hicbir yontemi ayirt etmez; disari alinir.
MIN_SOURCE_STD = 8.0


def aoi_bounds(aoi: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if aoi == "all":
        return SHARED_EAST, SHARED_NORTH
    if aoi == "train":
        return SHARED_EAST, (TRAIN_NORTH_MIN, SHARED_NORTH[1])
    if aoi == "test":
        return SHARED_EAST, (SHARED_NORTH[0], TEST_NORTH_MAX)
    raise ValueError(f"Bilinmeyen AOI: {aoi}")


def sample_centers(
    count: int,
    *,
    seed: int,
    aoi: str,
) -> list[tuple[float, float]]:
    """Doku kapisini gecen, sabit seedli merkezler uret."""
    (east_min, east_max), (north_min, north_max) = aoi_bounds(aoi)
    rng = np.random.default_rng(seed)
    centers: list[tuple[float, float]] = []
    with rasterio.open(QUERY_RASTER) as query, rasterio.open(MAP_RASTER) as search:
        attempts = 0
        limit = count * 40
        while len(centers) < count and attempts < limit:
            attempts += 1
            easting = float(rng.uniform(east_min, east_max))
            northing = float(rng.uniform(north_min, north_max))
            try:
                left = read_gray(query, easting, northing)
                right = read_gray(search, easting, northing)
            except (ValueError, rasterio.errors.RasterioIOError):
                continue
            if left is None or right is None:
                continue
            if min(float(np.std(left)), float(np.std(right))) < MIN_SOURCE_STD:
                continue
            centers.append((easting, northing))
    if len(centers) < count:
        raise RuntimeError(
            f"Yeterli dokulu merkez bulunamadi: {len(centers)}/{count} (aoi={aoi})"
        )
    return centers


def read_rgb(dataset, easting: float, northing: float) -> np.ndarray | None:
    row, col = dataset.index(easting, northing)
    window = Window(
        col - TILE_SIZE // 2,
        row - TILE_SIZE // 2,
        TILE_SIZE,
        TILE_SIZE,
    )
    if (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + TILE_SIZE > dataset.width
        or window.row_off + TILE_SIZE > dataset.height
    ):
        return None
    pixels = dataset.read(indexes=[1, 2, 3], window=window)
    if pixels.shape[1:] != (TILE_SIZE, TILE_SIZE):
        return None
    # Alfa bandi varsa tamamen gecerli olmayan kirpmalari at.
    if dataset.count >= 4:
        alpha = dataset.read(indexes=[dataset.count], window=window)
        if int(alpha.min()) == 0:
            return None
    return np.transpose(pixels, (1, 2, 0))


def read_gray(dataset, easting: float, northing: float) -> np.ndarray | None:
    rgb = read_rgb(dataset, easting, northing)
    if rgb is None:
        return None
    return crop_border(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def crop_border(pixels: np.ndarray) -> np.ndarray:
    return pixels[
        CROP_BORDER : TILE_SIZE - CROP_BORDER,
        CROP_BORDER : TILE_SIZE - CROP_BORDER,
    ]


def ncc(left: np.ndarray, right: np.ndarray) -> float:
    """Benchmark'in TM_CCOEFF_NORMED'i ile ayni nicelik, sifir kaymada."""
    a = left.astype(np.float32)
    b = right.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator < 1e-9:
        return 0.0
    return float((a * b).sum() / denominator)


def ncc_matrix(
    query_templates: Sequence[np.ndarray],
    map_templates: Sequence[np.ndarray],
) -> np.ndarray:
    """Tum sorgu/harita ciftlerinin ncc() degeri tek matris carpimiyla.

    ncc() ile ayni nicelik (sifir ortalama, norm carpimina bolum, payda
    ~0 ise 0); 150 ornekte ikili donguden yuzlerce kat hizlidir.
    """

    def centered(templates: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        rows = np.stack([np.asarray(t, dtype=np.float64).ravel() for t in templates])
        rows -= rows.mean(axis=1, keepdims=True)
        return rows, np.sqrt((rows * rows).sum(axis=1))

    queries, query_norms = centered(query_templates)
    maps, map_norms = centered(map_templates)
    numerator = queries @ maps.T
    denominator = np.outer(query_norms, map_norms)
    safe = np.where(denominator < 1e-9, 1.0, denominator)
    return np.where(denominator < 1e-9, 0.0, numerator / safe)


def score_pairs(
    query_templates: Sequence[np.ndarray],
    map_templates: Sequence[np.ndarray],
) -> dict[str, object]:
    """Gercek konum ve en iyi yanlis konum NCC'lerinden marj istatistigi."""
    count = len(query_templates)
    scores = ncc_matrix(query_templates, map_templates)
    truth = np.diag(scores).astype(np.float64)
    off_diagonal = scores.copy()
    np.fill_diagonal(off_diagonal, -np.inf)
    wrong = off_diagonal.max(axis=1).astype(np.float64)
    margin = truth - wrong
    rejected = sum(
        1
        for template in list(query_templates) + list(map_templates)
        if float(np.std(template)) < MIN_TEMPLATE_STD
    )
    truth_mean = float(truth.mean())
    wrong_mean = float(wrong.mean())
    wrong_std = float(wrong.std(ddof=1)) if count > 1 else 0.0
    # Olcek bagimsiz ayirt edicilik. Gercek tepe cokmusse (s+ <= 0) oran
    # tanimsizdir; bu durumu buyuk bir ceza degeriyle temsil et.
    if truth_mean > 1e-6:
        ratio = wrong_mean / truth_mean
    else:
        ratio = float("inf")
    separation_z = (truth_mean - wrong_mean) / wrong_std if wrong_std > 1e-9 else 0.0
    return {
        "samples": count,
        "truth_ncc_mean": truth_mean,
        "truth_ncc_median": float(np.median(truth)),
        "wrong_ncc_mean": wrong_mean,
        "wrong_ncc_std": wrong_std,
        "false_peak_ratio": ratio,
        "separation_z": float(separation_z),
        "margin_mean": float(margin.mean()),
        "margin_median": float(np.median(margin)),
        "wins": int((margin > 0).sum()),
        "win_rate": float((margin > 0).mean()),
        "rejected_templates": rejected,
        "rejected_total": 2 * count,
        "query_std_mean": float(np.mean([np.std(t) for t in query_templates])),
        "map_std_mean": float(np.mean([np.std(t) for t in map_templates])),
    }


def load_source_chips(
    centers: Sequence[tuple[float, float]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    query_chips: list[np.ndarray] = []
    map_chips: list[np.ndarray] = []
    with rasterio.open(QUERY_RASTER) as query, rasterio.open(MAP_RASTER) as search:
        for easting, northing in centers:
            left = read_rgb(query, easting, northing)
            right = read_rgb(search, easting, northing)
            if left is None or right is None:
                raise RuntimeError(f"Kirpma okunamadi: {easting},{northing}")
            query_chips.append(left)
            map_chips.append(right)
    return query_chips, map_chips


# Benchmark'in dort grubu: yon (g2b = Google sorgu / Bing harita, b2g = tersi)
# x sorgu varyanti. Varsayilan tek grup (g2b_clean) eski davranistir; resmi
# global skor dort grubun ortalamasi oldugu icin --bidirectional ve
# --variants clean,hard_v1 ile ayni ortalama kapida da alinabilir.
QUERY_VARIANTS = ("clean", "hard_v1")


def gate_groups(
    variants: Sequence[str],
    bidirectional: bool,
) -> list[tuple[str, str, str]]:
    """(grup adi, yon, varyant) listesi."""
    directions = ("g2b", "b2g") if bidirectional else ("g2b",)
    return [(f"{d}_{v}", d, v) for d in directions for v in variants]


def group_chip_keys(direction: str, variant: str) -> tuple[str, str]:
    """Grubun (sorgu, harita) kirpma kumesi anahtarlari.

    Benchmark'taki gibi bozulma yalniz sorguya uygulanir; harita temizdir.
    """
    query_key, map_key = ("G", "B") if direction == "g2b" else ("B", "G")
    if variant == "hard_v1":
        query_key += "_hard_v1"
    return query_key, map_key


def hard_v1_chips(
    chips: Sequence[np.ndarray],
    *,
    seed: int,
    prefix: str,
) -> list[np.ndarray]:
    """Benchmark'in hard_v1 bozulmasi, merkez basina deterministik seed ile."""
    from geospatial_model_benchmark import augment_hard_v1, hard_v1_seed

    return [
        augment_hard_v1(chip, seed=hard_v1_seed(seed, f"{prefix}{index:05d}"))[0]
        for index, chip in enumerate(chips)
    ]


def build_chip_sets(
    query_chips: Sequence[np.ndarray],
    map_chips: Sequence[np.ndarray],
    groups: Sequence[tuple[str, str, str]],
    *,
    seed: int,
) -> dict[str, list[np.ndarray]]:
    sets: dict[str, list[np.ndarray]] = {"G": list(query_chips), "B": list(map_chips)}
    for _, direction, variant in groups:
        query_key, _ = group_chip_keys(direction, variant)
        if query_key not in sets:
            source = query_key.split("_", 1)[0]
            sets[query_key] = hard_v1_chips(sets[source], seed=seed, prefix=f"gate_{source}_")
    return sets


def combine_groups(group_scores: dict[str, dict[str, object]]) -> dict[str, object]:
    """Tek grupta eski sonuc bicimi; cok grupta sayisal alanlarin ortalamasi."""
    first = next(iter(group_scores.values()))
    if len(group_scores) == 1:
        return dict(first)
    numeric = [
        key
        for key, value in first.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    combined: dict[str, object] = {
        key: float(np.mean([float(score[key]) for score in group_scores.values()]))
        for key in numeric
    }
    combined["samples"] = int(first["samples"])
    combined["groups"] = {name: dict(score) for name, score in group_scores.items()}
    return combined


def raw_group_scores(
    chip_sets: dict[str, list[np.ndarray]],
    groups: Sequence[tuple[str, str, str]],
) -> dict[str, object]:
    gray = {
        key: [crop_border(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)) for c in chips]
        for key, chips in chip_sets.items()
    }
    scores = {}
    for name, direction, variant in groups:
        query_key, map_key = group_chip_keys(direction, variant)
        scores[name] = score_pairs(gray[query_key], gray[map_key])
    return combine_groups(scores)


def model_group_scores(
    model_path: Path,
    chip_sets: dict[str, list[np.ndarray]],
    groups: Sequence[tuple[str, str, str]],
    *,
    normalization: str,
    batch_size: int,
    require_gpu: bool,
    threads: int = 0,
) -> dict[str, object]:
    """Benchmark'in kendi runtime'i ile olc; cikarim yolu birebir ayni olsun."""
    import goruntu_islemleri as gi

    if threads > 0:
        # CPU kapisi paralel benchmark isciliriyle yarismasin (TF varsayilani tum cekirdekler).
        import tensorflow as tf

        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            # inter_op=1, 12 dalli bir TTA toplulugunda Keras predict'i kilitledi
            # (2026-09-25: 28 dk'da 9 sn CPU); en az 2.
            tf.config.threading.set_inter_op_parallelism_threads(max(2, threads // 3))
        except RuntimeError:
            pass  # TF calisma zamani ilk modelde kuruldu; ayar zaten gecerli
    runtime = gi.LoadedModelRuntime.load(
        str(model_path),
        normalization=normalization,
        enhancement="none",
        output_value_mode="auto",
        require_gpu=require_gpu,
    )
    try:
        needed = sorted({key for _, d, v in groups for key in group_chip_keys(d, v)})
        outputs = {
            key: crop_all(predict_batched(runtime, chip_sets[key], batch_size=batch_size))
            for key in needed
        }
        scores = {}
        for name, direction, variant in groups:
            query_key, map_key = group_chip_keys(direction, variant)
            scores[name] = score_pairs(outputs[query_key], outputs[map_key])
        result = combine_groups(scores)
        result["output_value_mode"] = runtime.output_value_mode
        result["params"] = int(runtime.model.count_params())
        result["layers"] = len(runtime.model.layers)
        return result
    finally:
        import tensorflow as tf

        del runtime
        tf.keras.backend.clear_session()


def raw_score(
    query_chips: Sequence[np.ndarray],
    map_chips: Sequence[np.ndarray],
) -> dict[str, object]:
    groups = gate_groups(("clean",), False)
    return raw_group_scores({"G": list(query_chips), "B": list(map_chips)}, groups)


def model_score(
    model_path: Path,
    query_chips: Sequence[np.ndarray],
    map_chips: Sequence[np.ndarray],
    *,
    normalization: str,
    batch_size: int,
    require_gpu: bool,
) -> dict[str, object]:
    groups = gate_groups(("clean",), False)
    return model_group_scores(
        model_path,
        {"G": list(query_chips), "B": list(map_chips)},
        groups,
        normalization=normalization,
        batch_size=batch_size,
        require_gpu=require_gpu,
    )


def predict_batched(
    runtime,
    chips: Sequence[np.ndarray],
    *,
    batch_size: int,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for start in range(0, len(chips), batch_size):
        outputs.extend(
            runtime.predict_images(
                list(chips[start : start + batch_size]),
                image_size=(TILE_SIZE, TILE_SIZE),
                source_color="rgb",
            )
        )
    return outputs


def crop_all(images: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [crop_border(image) for image in images]


def collect_models(args: argparse.Namespace) -> list[Path]:
    models: list[Path] = [Path(p) for p in args.model]
    for directory in args.model_dir:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Model dizini bulunamadi: {root}")
        models.extend(
            sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
            )
        )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in models:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kaynaklar-arasi eslesme marjini olcerek adaylari benchmark'tan "
            "once eler. RAW her zaman olculur ve gecilmesi gereken baraj odur."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Tek model dosyasi; birden fazla kez verilebilir.",
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Model klasoru; alt klasorler dahil taranir.",
    )
    parser.add_argument("--raw-only", action="store_true", help="Yalniz RAW baraji.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aoi",
        choices=("all", "train", "test"),
        default="all",
        help="Merkezlerin secilecegi bolge; egitim degerlendirmesinde 'train' kullanin.",
    )
    parser.add_argument("--normalization", default="minus1_1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="TF CPU is parcacigi siniri (0 = TF varsayilani, tum cekirdekler).",
    )
    parser.add_argument(
        "--variants",
        default="clean",
        help="Sorgu varyantlari (virgulle): clean, hard_v1. Oran gruplarin ortalamasidir.",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Bing->Google yonunu de olc (benchmark'in dort grubu icin hard_v1 ile birlikte).",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    args.variants = tuple(v.strip() for v in str(args.variants).split(",") if v.strip())
    unknown = [v for v in args.variants if v not in QUERY_VARIANTS]
    if unknown or not args.variants:
        parser.error(f"Gecersiz varyant: {unknown or args.variants}; gecerli: {QUERY_VARIANTS}")
    return args


TABLE_HEADER = (
    f"{'aday':50} {'ORAN':>6} {'z':>5} {'gercek':>7} {'yanlis':>7} "
    f"{'marj':>8} {'kazanc':>7} {'red':>7} {'std q/m':>13}"
)


def format_row(name: str, score: dict[str, object]) -> str:
    ratio = float(score["false_peak_ratio"])
    ratio_text = "  inf" if ratio == float("inf") else f"{ratio:6.2f}"
    return (
        f"{name[:50]:50} "
        f"{ratio_text} "
        f"{score['separation_z']:5.2f} "
        f"{score['truth_ncc_mean']:+7.4f} "
        f"{score['wrong_ncc_mean']:+7.4f} "
        f"{score['margin_mean']:+8.4f} "
        f"{score['wins']:>3}/{score['samples']:<3} "
        f"{score['rejected_templates']:>3}/{score['rejected_total']:<3} "
        f"{score['query_std_mean']:6.1f}/{score['map_std_mean']:<6.1f}"
    )


def file_sha256(path: Path) -> str:
    """Pano (sonuc_panosu.py) kapi olcumunu resmi sonuclarla bu kimlikle eslestirir."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def print_groups(score: dict[str, object]) -> None:
    groups = score.get("groups")
    if isinstance(groups, dict):
        parts = [f"{name}={float(g['false_peak_ratio']):.3f}" for name, g in groups.items()]
        print(f"{'':50} gruplar: {'  '.join(parts)}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.require_gpu:
        # "" bu ortamda GPU'yu GIZLEMEZ (TF 2.10 yine gorur ve bellek ayirir,
        # paralel benchmark'la cakisir); "-1" gizler (2026-09-25 olculdu).
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    for raster in (QUERY_RASTER, MAP_RASTER):
        if not raster.is_file():
            raise SystemExit(f"Raster bulunamadi: {raster}")

    models = [] if args.raw_only else collect_models(args)

    groups = gate_groups(args.variants, args.bidirectional)
    print(
        f"Marj kapisi | ornek={args.samples} seed={args.seed} aoi={args.aoi} "
        f"normalizasyon={args.normalization} gruplar={','.join(g[0] for g in groups)}"
    )
    centers = sample_centers(args.samples, seed=args.seed, aoi=args.aoi)
    query_chips, map_chips = load_source_chips(centers)
    chip_sets = build_chip_sets(query_chips, map_chips, groups, seed=args.seed)

    print()
    print(TABLE_HEADER)
    print("-" * len(TABLE_HEADER))

    results: dict[str, dict[str, object]] = {}
    baseline = raw_group_scores(chip_sets, groups)
    results["RAW"] = baseline
    print(format_row("RAW (baraj)", baseline))
    print_groups(baseline)

    # Birincil olcut: yanlis tepe / gercek tepe orani. Kucuk olan iyi.
    bar = float(baseline["false_peak_ratio"])
    passed: list[str] = []
    failed: list[str] = []

    def write_json(complete: bool) -> None:
        """Her modelden sonra yazilir: surec cokerse (bellek, surucu sifirlama)
        olculmus modeller kaybolmaz; 'complete' yalniz sonda True olur."""
        if not args.json_out:
            return
        payload = {
            "samples": args.samples,
            "seed": args.seed,
            "aoi": args.aoi,
            "normalization": args.normalization,
            "groups": [name for name, _, _ in groups],
            "primary_metric": "false_peak_ratio",
            "raw_false_peak_ratio": bar,
            "raw_margin": float(baseline["margin_mean"]),
            "results": results,
            "passed": passed,
            "failed": failed,
            "complete": complete,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.json_out.with_name(args.json_out.name + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, args.json_out)

    for model_path in models:
        try:
            score = model_group_scores(
                model_path,
                chip_sets,
                groups,
                normalization=args.normalization,
                batch_size=args.batch_size,
                require_gpu=args.require_gpu,
                threads=args.threads,
            )
        except Exception as exc:  # noqa: BLE001 - aday elenir, kosu surer
            print(f"{model_path.name[:52]:52} HATA: {str(exc).splitlines()[0][:60]}")
            results[model_path.name] = {"error": str(exc)}
            failed.append(model_path.name)
            continue
        score["model_path"] = str(model_path.resolve())
        score["model_sha256"] = file_sha256(model_path)
        results[model_path.name] = score
        print(format_row(model_path.name, score), flush=True)
        print_groups(score)
        if float(score["false_peak_ratio"]) < bar:
            passed.append(model_path.name)
        else:
            failed.append(model_path.name)
        write_json(complete=False)

    print()
    print(
        f"RAW baraji: ORAN < {bar:.2f}  "
        f"(yanlis tepe / gercek tepe; kucuk olan daha ayirt edici)"
    )
    if models:
        ranked = sorted(
            (name for name in passed + failed if "error" not in results[name]),
            key=lambda name: float(results[name]["false_peak_ratio"]),
        )
        print(f"Gecen: {len(passed)}/{len(models)}")
        for name in ranked:
            mark = "GECTI " if name in passed else "ELENDI"
            print(f"  {mark} oran={float(results[name]['false_peak_ratio']):5.2f}  {name}")
        if failed:
            print("Elenen adaylar benchmark'a sokulmamali.")

    if args.json_out:
        write_json(complete=True)
        print(f"JSON yazildi: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
