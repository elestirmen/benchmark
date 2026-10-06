import os, sys, glob
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import h5py
import numpy as np, cv2, rasterio
from rasterio.windows import Window
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
ROOT = r"C:\d_surucusu\visual_navigation\benchmark"
Q = ROOT + r"\urgup_cevresi_harita\urgup_cevresi_gmap_utm_30cm.tif"
M = ROOT + r"\urgup_cevresi_harita\urgup_cevresi_bingmap_utm_30cm.tif"
SP = sys.argv[1]
models = sys.argv[2:]
T = 544
def read(ds, e, n):
    r, c = ds.index(e, n)
    px = ds.read(indexes=[1,2,3], window=Window(c - T//2, r - T//2, T, T))
    return cv2.cvtColor(np.transpose(px, (1,2,0)), cv2.COLOR_RGB2GRAY)
centers = [(665000, 4270000), (670500, 4266500), (675000, 4275000), (668000, 4258000)]
rows = []
with rasterio.open(Q) as q, rasterio.open(M) as m:
    print(q.bounds, q.res, q.count)
    tiles = [(read(q, e, n), read(m, e, n)) for e, n in centers]
outs = {}
for mp in models:
    mdl = tf.keras.models.load_model(mp, compile=False)
    batch = np.stack([t for pair in tiles for t in pair]).astype(np.float32)[..., None]
    batch = (batch - 127.5) / 127.5
    pred = mdl.predict(batch, verbose=0)[..., 0]
    u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    outs[mp] = u8
    stats = [(float(p.min()), float(p.max()), float(p.std()), len(np.unique(u))) for p, u in zip(pred, u8)]
    print(os.path.basename(mp)[:60], ["min%.3f max%.3f std%.3f lv%d" % s for s in stats[:4]])
    del mdl
    tf.keras.backend.clear_session()
# montage: for each center a row: G, B, then per model G_out, B_out
S = 256
def small(x): return cv2.resize(x, (S, S), interpolation=cv2.INTER_AREA)
rows = []
for i, (g, b) in enumerate(tiles):
    row = [small(g), small(b)]
    for mp in models:
        row += [small(outs[mp][2*i]), small(outs[mp][2*i+1])]
    rows.append(np.concatenate([np.pad(x, 2, constant_values=128) for x in row], axis=1))
img = np.concatenate(rows, axis=0)
cv2.imwrite(os.path.join(SP, "viz_outputs.png"), img)
print("saved", img.shape)
