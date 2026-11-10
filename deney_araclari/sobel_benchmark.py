#!/usr/bin/env python
"""Gradient-magnitude baseline on the wider Urgup district benchmark.

The flight replay uses a classical structural representation as a baseline (flight_replay/cae_infer.py,
SobelTransform): gradient magnitude of the Gaussian-smoothed (sigma 1.5) grayscale tile, 3 x 3 Sobel,
magnitude x 0.5, clipped to 255. The district benchmark of Table 6 did not include it. This tool wraps the
same operation as a fixed-weight Keras model, so that the benchmark processes it exactly like a network
(same tiling, normalization, pyramid search, queries and bootstrap), and checks it against OpenCV.

Usage (visual_navigation_cuda environment, from the benchmark folder):
  python deney_araclari/sobel_benchmark.py build      -> models_sobel/sobel_g15.h5 (+ check against OpenCV)
  python run_urgup_cevresi_models.py --model-dir models_sobel --output-dir outputs/sobel_baseline_4000 ^
         --max-queries 4000 --search-modes global      (same queries as Table 6; RAW is recomputed as a check)
  python deney_araclari/sobel_benchmark.py summary    -> outputs/sobel_baseline_4000/sobel_summary.json: global success
         of RAW, checkpoint 1850 (read from the Table 6 run) and Sobel on the identical queries, paired differences
         with 1 km block-bootstrap intervals, the two halves of the spatial split and query-texture tertiles

Result (2026-11-09): gradient magnitude 78.6% (clean) / 77.4% (hard_v1) against 69.0% / 62.7% for checkpoint 1850 and
49.1% / 46.2% for grayscale NCC (mean of both directions). The RAW run reproduced Table 6 query by query.
"""
import os, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models_sobel'
MODEL = MODEL_DIR / 'sobel_g15.h5'
OUT = ROOT / 'outputs' / 'sobel_baseline_4000'
T = 544


def opencv_reference(u8):
    import cv2
    g = cv2.GaussianBlur(u8, (0, 0), 1.5)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.clip(np.sqrt(sx * sx + sy * sy) * 0.5, 0, 255)


def build():
    # tf must be a module-level name: Keras serializes the Lambda functions, and a closure over a local module
    # cannot be copied ("cannot pickle 'module' object")
    global tf
    import cv2
    import tensorflow as tf
    L = tf.keras.layers
    k1 = cv2.getGaussianKernel(11, 1.5).astype(np.float32)          # OpenCV picks 11 taps for sigma 1.5 on 8-bit input
    gk = (k1 @ k1.T)[:, :, None, None]
    sob = np.zeros((3, 3, 1, 2), np.float32)
    sob[:, :, 0, 0] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sob[:, :, 0, 1] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    x_in = L.Input((T, T, 1))
    x = L.Rescaling(127.5, offset=127.5, name='to_gray_levels')(x_in)                 # [-1, 1] -> [0, 255]
    x = L.Lambda(lambda t: tf.pad(t, [[0, 0], [5, 5], [5, 5], [0, 0]], mode='REFLECT'), name='pad5')(x)
    x = L.Conv2D(1, 11, padding='valid', use_bias=False, trainable=False, name='gauss_s15')(x)
    x = L.Lambda(lambda t: tf.round(tf.clip_by_value(t, 0.0, 255.0)), name='to_u8')(x)  # OpenCV blurs in uint8
    x = L.Lambda(lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT'), name='pad1')(x)
    x = L.Conv2D(2, 3, padding='valid', use_bias=False, trainable=False, name='sobel3')(x)
    x = L.Lambda(lambda t: tf.clip_by_value(tf.sqrt(tf.reduce_sum(t * t, axis=-1, keepdims=True) + 1e-12) * 0.5,
                                            0.0, 255.0) / 255.0, name='magnitude')(x)
    m = tf.keras.Model(x_in, x, name='gradient_magnitude_s15')
    m.get_layer('gauss_s15').set_weights([gk])
    m.get_layer('sobel3').set_weights([sob])
    MODEL_DIR.mkdir(exist_ok=True)
    m.save(str(MODEL))
    m2 = tf.keras.models.load_model(str(MODEL), compile=False)
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in range(4):
        u8 = cv2.GaussianBlur(rng.integers(0, 256, (T, T), dtype=np.uint8), (0, 0), 2 + i)  # textured test tiles
        cv2.rectangle(u8, (100, 120), (300, 260), 255, 3); cv2.circle(u8, (400, 400), 60, 30, -1)
        ref = opencv_reference(u8)
        y = m2.predict((u8.astype(np.float32) / 127.5 - 1.0)[None, :, :, None], verbose=0)[0, :, :, 0] * 255.0
        d = np.abs(y - ref)[16:-16, 16:-16]
        worst = max(worst, float(d.max()))
        print(f'tile {i}: max |keras - opencv| = {d.max():.3f}, mean = {d.mean():.4f}')
    meta = {'input_normalization': 'minus1_1', 'enhancement': 'none', 'color_mode': 'grayscale', 'image_size': T,
            'output_activation': 'linear [0, 1]', 'note': 'gradient magnitude baseline, fixed weights, identical to '
            'flight_replay SobelTransform (Gaussian sigma 1.5, 3x3 Sobel, x0.5, clip 255)', 'max_abs_diff_opencv': worst}
    MODEL.with_suffix('.pre.json').write_text(json.dumps(meta, indent=2))
    print('saved', MODEL, '| worst difference to OpenCV (gray levels):', round(worst, 3))
    if worst > 4.0:  # OpenCV blurs 8-bit tiles in fixed-point arithmetic; expected <= 3 gray levels
        print('WARNING: Keras model differs from OpenCV by more than 4 gray levels'); sys.exit(1)


TABLE6 = ROOT / 'outputs' / 'urgup_cevresi_models_4000'          # the Table 6 run (same 4000 queries per direction)
CKPT_1850 = 'cb62bfb86ad0'                                          # f48-k4-s1 checkpoint 1850 (benchmark-selected, Table 6)
KEEP = ['direction', 'query_variant', 'search_mode', 'model_id', 'query_id', 'block_id', 'query_center_northing_m',
        'query_entropy', 'error_m', 'success_25m', 'status', 'reason']
BOOT_ITER, BOOT_SEED = 1000, 42                                     # as in the benchmark runs (run_config.json)
BUFFER_ROW = 13                                                     # spatial split of Section 4.4: rows 0-12 | 13 | 14-25


def _stream(path, want):
    """Global-search records of the models in `want` from a results.jsonl (streams, keeps the KEEP fields)."""
    import pandas as pd
    rows = []
    with open(path, 'rb') as f:
        for line in f:
            if b'"search_mode": "global"' not in line and b'"search_mode":"global"' not in line: continue
            if not any(w in line for w in want): continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get('search_mode') != 'global': continue
            rows.append({k: r.get(k) for k in KEEP})
    return pd.DataFrame(rows)


def _load_table6():
    """RAW and checkpoint 1850 global records of the Table 6 run, cached as a small CSV next to the Sobel outputs."""
    import pandas as pd
    cache = OUT / 'table6_global_raw_1850.csv'
    if cache.exists(): return pd.read_csv(cache)
    src = TABLE6 / 'results.jsonl'
    print('streaming', src, '(once; cached afterwards)')
    d = _stream(src, [b'RAW_BASELINE', CKPT_1850.encode()])
    d = d[(d.model_id == 'RAW_BASELINE') | d.model_id.str.endswith(CKPT_1850)]
    n0 = len(d); d = d.drop_duplicates(['direction', 'query_variant', 'model_id', 'query_id'], keep='first')
    print('table 6 records:', n0, 'duplicates dropped:', n0 - len(d))
    d.to_csv(cache, index=False)
    return d


def _block_boot(x, a, b, rng):
    """95% block-bootstrap interval of mean(x[a]) - mean(x[b]) in percentage points (1 km blocks resampled)."""
    g = x.groupby('block_id')[[a, b]].agg(['sum', 'count'])
    sa, sb, cnt = g[(a, 'sum')].values, g[(b, 'sum')].values, g[(a, 'count')].values
    n = len(cnt); diffs = np.empty(BOOT_ITER)
    for i in range(BOOT_ITER):
        j = rng.integers(0, n, n); diffs[i] = 100 * (sa[j].sum() - sb[j].sum()) / cnt[j].sum()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def summary():
    """Global success of gradient magnitude against grayscale NCC and checkpoint 1850 on the identical queries.

    Writes sobel_summary.json. Rejected queries (no usable template) count as failures, as everywhere in the
    benchmark. Differences are paired (same query, same variant, same direction) with a 1 km block bootstrap."""
    import pandas as pd
    new = _stream(OUT / 'results.jsonl', [b'RAW_BASELINE', b'sobel'])
    new['m'] = np.where(new.model_id == 'RAW_BASELINE', 'raw', 'sobel')
    old = _load_table6()
    old['m'] = np.where(old.model_id == 'RAW_BASELINE', 'raw_t6', 'cae')
    for d_ in (new, old):
        d_['dir'] = np.where(d_.direction.str.startswith('urgup_cevresi_gmap'), 'gb', 'bg')   # gb: Google query on Bing map
        d_['ok'] = d_.success_25m.fillna(0).astype(int)                                        # rejected -> failure
    key = ['dir', 'query_variant', 'query_id']
    s = pd.concat([new, old]).pivot_table(index=key, columns='m', values='ok', aggfunc='first')
    e = pd.concat([new, old]).pivot_table(index=key, columns='m', values='error_m', aggfunc='first')
    meta = new[new.m == 'raw'].set_index(key)[['block_id', 'query_center_northing_m', 'query_entropy']]
    s = s.join(meta)
    assert s[['raw', 'sobel', 'raw_t6', 'cae']].notna().all().all(), 'query sets differ between the two runs'
    out = {'n_per_direction_variant': int(len(s) / 4), 'bootstrap': {'iterations': BOOT_ITER, 'seed': BOOT_SEED, 'block': '1 km'}}
    # 1) the grayscale run reproduces Table 6 query by query (same queries, same matcher)
    out['raw_differs_from_table6'] = int((s.raw != s.raw_t6).sum())
    out['raw_max_abs_error_diff_m'] = float((e.raw - e.raw_t6).abs().max())
    out['sobel_rejected'] = int((new[new.m == 'sobel'].status != 'ok').sum())
    out['sobel_rejected_reasons'] = new[(new.m == 'sobel') & (new.status != 'ok')].reason.value_counts().to_dict()
    # 2) success by direction and variant, and the mean of both directions (the convention of Table 6)
    by = s.groupby(['dir', 'query_variant'])[['raw', 'cae', 'sobel']].mean().mul(100)
    out['by_direction'] = {f'{d}|{v}': {m: round(float(by.loc[(d, v), m]), 2) for m in ('raw', 'cae', 'sobel')} for d, v in by.index}
    mean = by.groupby('query_variant').mean()
    out['mean'] = {v: {m: round(float(mean.loc[v, m]), 2) for m in ('raw', 'cae', 'sobel')} for v in mean.index}
    # 3) paired differences with block bootstrap (both directions pooled, per variant)
    rng = np.random.default_rng(BOOT_SEED)
    out['paired'] = {}
    for v in ('clean', 'hard_v1'):
        x = s.xs(v, level='query_variant')
        r = {'sobel_only': int(((x.sobel == 1) & (x.cae == 0)).sum()), 'cae_only': int(((x.sobel == 0) & (x.cae == 1)).sum()),
             'both': int(((x.sobel == 1) & (x.cae == 1)).sum()), 'neither': int(((x.sobel == 0) & (x.cae == 0)).sum())}
        for a, b in (('sobel', 'cae'), ('sobel', 'raw'), ('cae', 'raw')):
            lo, hi = _block_boot(x, a, b, rng)
            r[f'{a}_minus_{b}'] = {'pp': round(100 * float(x[a].mean() - x[b].mean()), 2), 'ci95': [round(lo, 2), round(hi, 2)]}
        out['paired'][v] = r
    # 4) the two halves of the spatial split (Section 4.4) and query-texture tertiles
    row = s.block_id.str.slice(1, 3).astype(int)
    north = row < BUFFER_ROW; south = row > BUFFER_ROW
    nb = s[north].block_id.nunique(); sb = s[south].block_id.nunique()
    if (nb, sb) != (259, 239): print(f'WARNING: half sizes {nb}/{sb} blocks differ from Section 4.4 (259/239)')
    out['halves'] = {}
    for name, sel in (('north', north), ('south', south)):
        h = s[sel].groupby('query_variant')[['raw', 'cae', 'sobel']].mean().mul(100)
        out['halves'][name] = {'blocks': int(nb if name == 'north' else sb),
                               **{f'{v}|{m}': round(float(h.loc[v, m]), 2) for v in h.index for m in ('raw', 'cae', 'sobel')}}
    s['tert'] = pd.qcut(s.query_entropy, 3, labels=['low', 'mid', 'high'])
    t = s.groupby(['tert', 'query_variant'], observed=True)[['raw', 'cae', 'sobel']].mean().mul(100)
    out['entropy_tertiles'] = {f'{a}|{v}': {m: round(float(t.loc[(a, v), m]), 2) for m in ('raw', 'cae', 'sobel')} for a, v in t.index}
    out['median_error_of_successes_m'] = {m: round(float(e[m][s[m] == 1].median()), 2) for m in ('raw', 'cae', 'sobel')}
    (OUT / 'sobel_summary.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('raw_differs_from_table6', 'sobel_rejected', 'mean', 'paired', 'halves')}, indent=1))
    print('saved', OUT / 'sobel_summary.json')


if __name__ == '__main__':
    {'build': build, 'summary': summary}[sys.argv[1] if len(sys.argv) > 1 else 'build']()
