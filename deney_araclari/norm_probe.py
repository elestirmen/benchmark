import os, sys, random, re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import h5py
import numpy as np, cv2
from PIL import Image, ImageOps
import tensorflow as tf
D = r"E:\temp\tez__ayna\python_calismalar\larger-google-sat2maps-dataset\islenmis\sonuclar_19"
names = sorted(n for n in os.listdir(D) if n.endswith(".png") and not re.search(r"( \(\d+\)| copy)\.png$", n))
random.seed(1)
names = random.sample(names, 48)
X, Y = [], []
for n in names:
    im = np.array(Image.open(os.path.join(D, n)).convert("L"))
    h, w = im.shape
    x, y = im[:, : w // 2], im[:, w // 2 :]
    X.append(cv2.resize(x, (544, 544), interpolation=cv2.INTER_NEAREST))
    Y.append(cv2.resize(y, (544, 544), interpolation=cv2.INTER_NEAREST))
X = np.stack(X); Y = np.stack(Y).astype(np.float32) / 255.0
Xeq = np.stack([np.array(ImageOps.equalize(Image.fromarray(x))) for x in X])
print("label black frac", float((Y < 0.5).mean()))
def bce(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
def iou_black(p, y):
    a = p < 0.5; b = y < 0.5
    return float((a & b).sum() / max(1, (a | b).sum()))
for mp in sys.argv[1:]:
    m = tf.keras.models.load_model(mp, compile=False)
    res = []
    for tag, xs in (("raw", X), ("eq", Xeq)):
        for norm in ("zero_1", "minus1_1"):
            xin = xs.astype(np.float32)[..., None]
            xin = xin / 255.0 if norm == "zero_1" else (xin - 127.5) / 127.5
            p = m.predict(xin, batch_size=8, verbose=0)[..., 0]
            res.append(f"{tag}/{norm}: bce={bce(p, Y):.3f} iou={iou_black(p, Y):.3f} mean={p.mean():.2f}")
    print(os.path.basename(mp)[:55], " | ".join(res))
    tf.keras.backend.clear_session()
