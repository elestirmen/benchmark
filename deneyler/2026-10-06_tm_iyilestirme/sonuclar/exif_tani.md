# EXIF tanisi: yaw kaynagi ve irtifa tutarliligi

Yaw farki = gimbal (kamera) yawi - ucus (govde) yawi; TM bugun ucus yawini kullaniyor.
Irtifa sacilimi = rota icinde std(GPSAltitude - RelativeAltitude); olcek std % = bunun yerden yukseklige orani.

| Rota | kare | kamera | yaw farki ort | medyan | std | abs p90 | >5 derece | RelAlt medyan (m) | irtifa std (m) | aralik (m) | olcek std % |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_tezde_ucus_5 | 237 | L1D-20c | -5.7 | -5.9 | 1.0 | 7.0 | 67% | 398 | 2.3 | 12.8 | 0.6 |
| 2_tezde_ucus_6 | 147 | L1D-20c | -4.3 | -4.1 | 1.9 | 6.2 | 36% | 454 | 2.8 | 12.1 | 0.6 |
| 3_tezde_ucus_7 | 174 | L1D-20c | -6.7 | -6.4 | 1.7 | 8.7 | 81% | 311 | 2.4 | 12.3 | 0.8 |
| 4_tezde_ucus_8 | 115 | L1D-20c | -8.3 | -8.3 | 0.5 | 8.7 | 100% | 499 | 4.0 | 11.3 | 0.8 |
| 5_tezde_ucus_9 | 115 | L1D-20c | 2.1 | 2.1 | 1.2 | 3.0 | 2% | 300 | 5.9 | 16.9 | 2.0 |
| 6_tezde_ucus_4 | 183 | L1D-20c | -4.8 | -4.8 | 0.4 | 5.3 | 23% | 499 | 2.0 | 9.6 | 0.4 |
| guzergah_1 | 81 | FC2204 | -1.5 | -1.5 | 0.1 | 1.6 | 0% | 500 | 1.9 | 6.1 | 0.4 |
| guzergah_1_donus | 54 | FC2204 | -2.1 | -2.1 | 0.4 | 2.2 | 2% | 500 | 0.3 | 1.2 | 0.1 |
| guzergah_2_tezde_ucus_1 | 145 | FC2204 | -5.6 | -5.6 | 0.7 | 6.3 | 81% | 500 | 1.4 | 11.6 | 0.3 |
| guzergah_3_tezde_ucus_2 | 200 | FC2204 | -10.6 | -10.6 | 0.7 | 11.3 | 100% | 500 | 1.2 | 6.5 | 0.2 |
| guzergah_4_tezde_ucus_3 | 210 | FC2204 | -8.9 | -9.0 | 0.4 | 9.3 | 100% | 500 | 0.7 | 5.2 | 0.1 |
| guzergah_5_to_karain | 198 | L1D-20c | -7.7 | -7.7 | 0.4 | 8.1 | 100% | 500 | 4.7 | 15.5 | 0.9 |
| guzergah_6_to_tagar | 211 | L1D-20c | -2.5 | -2.6 | 0.7 | 3.2 | 0% | 500 | 0.8 | 4.2 | 0.2 |
| guzergah_7_to_gurlevek | 181 | L1D-20c | -11.5 | -11.6 | 1.1 | 12.2 | 99% | 500 | 1.2 | 6.8 | 0.2 |
| guzergah_8_koy_vadi | 435 | FC2204 | -13.0 | -13.1 | 1.0 | 13.5 | 99% | 180 | 3.4 | 12.1 | 1.9 |
| guzergah_9_koy_dag | 265 | FC2204 | -10.6 | -11.4 | 4.6 | 13.0 | 98% | 360 | 2.0 | 9.8 | 0.6 |
