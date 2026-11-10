"""Kontrollu karsilastirma: harita hedefi yerine girdinin kendisini yeniden kuran (reconstruction) model.

Amac: kartografik denetimin katkisini,
ayni mimari / veri / tarif / secim (marj kapisi) / benchmark ile egitilmis bir girdi-yeniden-kurma
modeline karsi olcmek.

end_to_end_pipeline.py DEGISTIRILMEZ; bu betik onu modul olarak yukler ve calisma aninda:
  * CONFIG_KEYS'e TARGET_MODE ekler (egitim scriptinde Config.TARGET_MODE, varsayilan "map"),
  * mevcut tum islere TARGET_MODE="map" atar (davranis aynen korunur),
  * "z1_f32_k3_b1_recon" isini ekler: taban tarif (z1_f32_k3_b1) + TARGET_MODE="reconstruction".
Varsayilan olarak yeni bir pipeline kosusunda iki is calisir: recon isi ve harita tarifinin bir
tekrari (z1_f32_k3_b1_tekrar2), boylece iki hedef ayni oturumda, ayni GPU'da egitilir.
Kullanim (benchmark klasorunden):  python deney_araclari/recon_kontrol.py [--yalniz-recon]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
os.chdir(BENCH)
import end_to_end_pipeline as E

if "TARGET_MODE" not in E.CONFIG_KEYS:
    E.CONFIG_KEYS = tuple(E.CONFIG_KEYS) + ("TARGET_MODE",)
E.BASELINE_JOB["TARGET_MODE"] = "map"
for j in E.TRAINING_JOBS:
    j.setdefault("TARGET_MODE", "map")
if not any(j["name"] == "z1_f32_k3_b1_recon" for j in E.TRAINING_JOBS):
    E.TRAINING_JOBS.append(E.job("z1_f32_k3_b1_recon", TARGET_MODE="reconstruction"))

argv = ["--new-run", "--only", "z1_f32_k3_b1_recon"]
if "--yalniz-recon" not in sys.argv:
    argv += ["--only", "z1_f32_k3_b1_tekrar2"]
if "--run-id" in sys.argv:  # devam: python recon_kontrol.py --run-id YYYYMMDD_HHMMSS
    rid = sys.argv[sys.argv.index("--run-id") + 1]
    argv = ["--run-id", rid] + argv[1:]
print("recon_kontrol | pipeline argumanlari:", argv, flush=True)
E.main(argv)
