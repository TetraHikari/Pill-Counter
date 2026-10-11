# Other medicines: counting evaluation

Thresholds (tuned on our val, one photo per call as the app does): thai_v2 conf 0.55 IoU 0.6; thai_v2_appfile conf 0.5 IoU 0.5; thai_v3_m1024 conf 0.25 IoU 0.5; thai_v4m_m1024 conf 0.6 IoU 0.5; thai_mix_s1024 conf 0.55 IoU 0.6; imgsz 1024; max_det 500. Error = predicted − true. bias > 0 = over-counts. Tags in brackets = the model saw that set while training or choosing its checkpoint (seen-ish: v4 valid was the v4-stage validation set of thai_v2 / thai_v4m_m1024; thai_mix_*: trained on the Pill-Counter v2 train part, validated on its val part and v4 valid). **pc2_heldout_test is the fair comparison for every model.**

## Overall

| dataset | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| pc2_heldout_test | thai_v2 | 187 | 642 | 22.5 | 1.8 | -0.7 | 16.6 | 61.0 | 59.5 | 24 |
| pc2_heldout_test | thai_v2_appfile | 187 | 642 | 25.1 | 1.9 | -0.3 | 20.9 | 54.0 | 63.1 | 30 |
| pc2_heldout_test | thai_v3_m1024 | 187 | 642 | 16.6 | 3.6 | 0.9 | 29.4 | 54.0 | 140.6 | 115 |
| pc2_heldout_test | thai_v4m_m1024 | 187 | 642 | 36.9 | 1.6 | 0.1 | 21.4 | 41.7 | 55.8 | 42 |
| pc2_heldout_test | thai_mix_s1024 | 187 | 642 | 57.8 | 0.7 | -0.5 | 4.3 | 38.0 | 21.5 | 5 |
| pillcounter_v2 | thai_v2 | 1255 | 4733 | 34.3 | 1.6 | -0.6 | 13.8 | 51.9 | 52.9 | 49 |
| pillcounter_v2 | thai_v2_appfile | 1255 | 4733 | 35.8 | 1.6 | -0.3 | 16.5 | 47.7 | 56.3 | 73 |
| pillcounter_v2 | thai_v3_m1024 | 1255 | 4733 | 13.9 | 4.4 | 0.8 | 25.6 | 60.5 | 177.1 | 499 |
| pillcounter_v2 | thai_v4m_m1024 | 1255 | 4733 | 48.4 | 1.5 | 0.4 | 18.6 | 33.0 | 56.0 | 78 |
| pillcounter_v2 | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 1255 | 4733 | 61.8 | 0.6 | -0.4 | 5.7 | 32.5 | 19.4 | 9 |
| v4_valid | thai_v2 (seen-ish) | 182 | 10356 | 59.3 | 1.8 | -1.4 | 10.4 | 30.2 | 3.3 | 37 |
| v4_valid | thai_v2_appfile (seen-ish) | 182 | 10356 | 65.9 | 1.3 | -1.1 | 8.8 | 25.3 | 2.7 | 28 |
| v4_valid | thai_v3_m1024 | 182 | 10356 | 23.6 | 14.0 | 13.5 | 70.9 | 5.5 | 29.7 | 118 |
| v4_valid | thai_v4m_m1024 (seen-ish) | 182 | 10356 | 74.2 | 0.4 | -0.0 | 14.8 | 11.0 | 0.9 | 5 |
| v4_valid | thai_mix_s1024 (seen: validation) | 182 | 10356 | 95.1 | 0.1 | 0.0 | 2.7 | 2.2 | 0.1 | 4 |
| v1_trays | thai_v2 | 112 | 8838 | 0.0 | 41.9 | -40.7 | 0.9 | 99.1 | 59.2 | 129 |
| v1_trays | thai_v2_appfile | 112 | 8838 | 0.0 | 40.8 | -39.7 | 0.9 | 99.1 | 57.9 | 124 |
| v1_trays | thai_v3_m1024 | 112 | 8838 | 4.5 | 16.4 | -15.2 | 2.7 | 92.9 | 26.7 | 66 |
| v1_trays | thai_v4m_m1024 | 112 | 8838 | 0.0 | 31.3 | -30.2 | 0.9 | 99.1 | 45.5 | 102 |
| v1_trays | thai_mix_s1024 | 112 | 8838 | 0.0 | 47.8 | -46.8 | 0.9 | 99.1 | 66.1 | 132 |

## Pill-Counter v2, held-out test part (no model trained or chose on it)

### By pile size

| pile | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–20 | thai_v2 | 187 | 642 | 22.5 | 1.8 | -0.7 | 16.6 | 61.0 | 59.5 | 24 |
| 1–20 | thai_v2_appfile | 187 | 642 | 25.1 | 1.9 | -0.3 | 20.9 | 54.0 | 63.1 | 30 |
| 1–20 | thai_v3_m1024 | 187 | 642 | 16.6 | 3.6 | 0.9 | 29.4 | 54.0 | 140.6 | 115 |
| 1–20 | thai_v4m_m1024 | 187 | 642 | 36.9 | 1.6 | 0.1 | 21.4 | 41.7 | 55.8 | 42 |
| 1–20 | thai_mix_s1024 | 187 | 642 | 57.8 | 0.7 | -0.5 | 4.3 | 38.0 | 21.5 | 5 |

### By class group (an image's group = its majority class; a tie with numbered ids goes to the named class, a tie between named classes is "mixed (no majority)")

| group | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| brown_capsule | thai_v2 | 8 | 14 | 37.5 | 0.8 | 0.5 | 50.0 | 12.5 | 50.0 | 2 |
| brown_capsule | thai_v2_appfile | 8 | 14 | 25.0 | 1.1 | 0.9 | 62.5 | 12.5 | 70.8 | 3 |
| brown_capsule | thai_v3_m1024 | 8 | 14 | 50.0 | 1.1 | 1.1 | 50.0 | 0.0 | 54.2 | 3 |
| brown_capsule | thai_v4m_m1024 | 8 | 14 | 62.5 | 0.4 | -0.1 | 12.5 | 25.0 | 29.2 | 1 |
| brown_capsule | thai_mix_s1024 | 8 | 14 | 50.0 | 0.6 | -0.4 | 12.5 | 37.5 | 45.8 | 2 |
| numbered drug ids | thai_v2 | 179 | 628 | 21.8 | 1.8 | -0.7 | 15.1 | 63.1 | 59.9 | 24 |
| numbered drug ids | thai_v2_appfile | 179 | 628 | 25.1 | 1.9 | -0.4 | 19.0 | 55.9 | 62.7 | 30 |
| numbered drug ids | thai_v3_m1024 | 179 | 628 | 15.1 | 3.7 | 0.9 | 28.5 | 56.4 | 144.4 | 115 |
| numbered drug ids | thai_v4m_m1024 | 179 | 628 | 35.8 | 1.7 | 0.1 | 21.8 | 42.5 | 57.0 | 42 |
| numbered drug ids | thai_mix_s1024 | 179 | 628 | 58.1 | 0.7 | -0.5 | 3.9 | 38.0 | 20.4 | 5 |

0 of 187 images hold more than one group.

### Any image containing the class (overlapping groups)

| contains | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| brown_capsule | thai_v2 | 8 | 14 | 37.5 | 0.8 | 0.5 | 50.0 | 12.5 | 50.0 | 2 |
| brown_capsule | thai_v2_appfile | 8 | 14 | 25.0 | 1.1 | 0.9 | 62.5 | 12.5 | 70.8 | 3 |
| brown_capsule | thai_v3_m1024 | 8 | 14 | 50.0 | 1.1 | 1.1 | 50.0 | 0.0 | 54.2 | 3 |
| brown_capsule | thai_v4m_m1024 | 8 | 14 | 62.5 | 0.4 | -0.1 | 12.5 | 25.0 | 29.2 | 1 |
| brown_capsule | thai_mix_s1024 | 8 | 14 | 50.0 | 0.6 | -0.4 | 12.5 | 37.5 | 45.8 | 2 |
| numbered drug ids | thai_v2 | 179 | 628 | 21.8 | 1.8 | -0.7 | 15.1 | 63.1 | 59.9 | 24 |
| numbered drug ids | thai_v2_appfile | 179 | 628 | 25.1 | 1.9 | -0.4 | 19.0 | 55.9 | 62.7 | 30 |
| numbered drug ids | thai_v3_m1024 | 179 | 628 | 15.1 | 3.7 | 0.9 | 28.5 | 56.4 | 144.4 | 115 |
| numbered drug ids | thai_v4m_m1024 | 179 | 628 | 35.8 | 1.7 | 0.1 | 21.8 | 42.5 | 57.0 | 42 |
| numbered drug ids | thai_mix_s1024 | 179 | 628 | 58.1 | 0.7 | -0.5 | 3.9 | 38.0 | 20.4 | 5 |

## Pill-Counter v2 (all splits)

### By pile size

| pile | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–20 | thai_v2 | 1255 | 4733 | 34.3 | 1.6 | -0.6 | 13.8 | 51.9 | 52.9 | 49 |
| 1–20 | thai_v2_appfile | 1255 | 4733 | 35.8 | 1.6 | -0.3 | 16.5 | 47.7 | 56.3 | 73 |
| 1–20 | thai_v3_m1024 | 1255 | 4733 | 13.9 | 4.4 | 0.8 | 25.6 | 60.5 | 177.1 | 499 |
| 1–20 | thai_v4m_m1024 | 1255 | 4733 | 48.4 | 1.5 | 0.4 | 18.6 | 33.0 | 56.0 | 78 |
| 1–20 | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 1255 | 4733 | 61.8 | 0.6 | -0.4 | 5.7 | 32.5 | 19.4 | 9 |

### By class group (an image's group = its majority class; a tie with numbered ids goes to the named class, a tie between named classes is "mixed (no majority)")

| group | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| Witaminy | thai_v2 | 1 | 15 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| Witaminy | thai_v2_appfile | 1 | 15 | 0.0 | 1.0 | 1.0 | 100.0 | 0.0 | 6.7 | 1 |
| Witaminy | thai_v3_m1024 | 1 | 15 | 0.0 | 1.0 | 1.0 | 100.0 | 0.0 | 6.7 | 1 |
| Witaminy | thai_v4m_m1024 | 1 | 15 | 0.0 | 1.0 | -1.0 | 0.0 | 100.0 | 6.7 | 1 |
| Witaminy | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 1 | 15 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| brown_capsule | thai_v2 | 18 | 38 | 50.0 | 0.6 | 0.3 | 38.9 | 11.1 | 29.6 | 2 |
| brown_capsule | thai_v2_appfile | 18 | 38 | 38.9 | 0.8 | 0.7 | 55.6 | 5.6 | 40.7 | 3 |
| brown_capsule | thai_v3_m1024 | 18 | 38 | 27.8 | 1.2 | 0.8 | 50.0 | 22.2 | 59.3 | 3 |
| brown_capsule | thai_v4m_m1024 | 18 | 38 | 72.2 | 0.3 | 0.1 | 16.7 | 11.1 | 18.5 | 2 |
| brown_capsule | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 18 | 38 | 55.6 | 0.6 | -0.1 | 16.7 | 27.8 | 29.6 | 2 |
| mixed (no majority) | thai_v2 | 268 | 1389 | 82.1 | 0.2 | -0.1 | 6.7 | 11.2 | 5.9 | 6 |
| mixed (no majority) | thai_v2_appfile | 268 | 1389 | 82.5 | 0.2 | -0.0 | 7.8 | 9.7 | 5.9 | 8 |
| mixed (no majority) | thai_v3_m1024 | 268 | 1389 | 4.9 | 3.8 | -3.7 | 2.6 | 92.5 | 77.6 | 8 |
| mixed (no majority) | thai_v4m_m1024 | 268 | 1389 | 82.5 | 0.2 | 0.0 | 10.1 | 7.5 | 4.9 | 4 |
| mixed (no majority) | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 268 | 1389 | 88.1 | 0.2 | -0.2 | 1.1 | 10.8 | 4.2 | 3 |
| white_tablet | thai_v2 | 17 | 37 | 41.2 | 0.8 | 0.2 | 35.3 | 23.5 | 47.1 | 2 |
| white_tablet | thai_v2_appfile | 17 | 37 | 47.1 | 0.8 | 0.4 | 35.3 | 17.6 | 45.1 | 2 |
| white_tablet | thai_v3_m1024 | 17 | 37 | 35.3 | 0.9 | 0.5 | 47.1 | 17.6 | 66.7 | 3 |
| white_tablet | thai_v4m_m1024 | 17 | 37 | 70.6 | 0.4 | -0.1 | 17.6 | 11.8 | 25.5 | 3 |
| white_tablet | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 17 | 37 | 41.2 | 0.7 | -0.7 | 0.0 | 58.8 | 39.2 | 2 |
| yellow_tablet | thai_v2 | 17 | 37 | 58.8 | 0.4 | 0.2 | 29.4 | 11.8 | 29.4 | 1 |
| yellow_tablet | thai_v2_appfile | 17 | 37 | 58.8 | 0.5 | 0.4 | 35.3 | 5.9 | 35.3 | 2 |
| yellow_tablet | thai_v3_m1024 | 17 | 37 | 76.5 | 0.3 | 0.2 | 17.6 | 5.9 | 25.5 | 2 |
| yellow_tablet | thai_v4m_m1024 | 17 | 37 | 76.5 | 0.2 | 0.0 | 11.8 | 11.8 | 15.7 | 1 |
| yellow_tablet | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 17 | 37 | 76.5 | 0.3 | -0.3 | 0.0 | 23.5 | 9.8 | 2 |
| numbered drug ids | thai_v2 | 934 | 3217 | 19.7 | 2.0 | -0.8 | 14.7 | 65.6 | 67.4 | 49 |
| numbered drug ids | thai_v2_appfile | 934 | 3217 | 21.7 | 2.1 | -0.4 | 17.5 | 60.8 | 71.7 | 73 |
| numbered drug ids | thai_v3_m1024 | 934 | 3217 | 14.8 | 4.7 | 2.1 | 31.4 | 53.9 | 212.9 | 499 |
| numbered drug ids | thai_v4m_m1024 | 934 | 3217 | 37.4 | 2.0 | 0.5 | 21.2 | 41.4 | 72.7 | 78 |
| numbered drug ids | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 934 | 3217 | 54.5 | 0.7 | -0.5 | 7.0 | 38.5 | 23.3 | 9 |

269 of 1255 images hold more than one group.

### Any image containing the class (overlapping groups)

| contains | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| Potas | thai_v2 | 9 | 145 | 33.3 | 1.2 | -1.2 | 0.0 | 66.7 | 7.4 | 3 |
| Potas | thai_v2_appfile | 9 | 145 | 22.2 | 1.1 | -0.7 | 22.2 | 55.6 | 6.8 | 3 |
| Potas | thai_v3_m1024 | 9 | 145 | 55.6 | 0.7 | -0.2 | 22.2 | 22.2 | 4.1 | 2 |
| Potas | thai_v4m_m1024 | 9 | 145 | 11.1 | 1.6 | -1.6 | 0.0 | 88.9 | 9.5 | 4 |
| Potas | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 9 | 145 | 11.1 | 2.0 | -2.0 | 0.0 | 88.9 | 12.3 | 3 |
| Witaminy | thai_v2 | 9 | 145 | 33.3 | 1.2 | -1.2 | 0.0 | 66.7 | 7.4 | 3 |
| Witaminy | thai_v2_appfile | 9 | 145 | 22.2 | 1.1 | -0.7 | 22.2 | 55.6 | 6.8 | 3 |
| Witaminy | thai_v3_m1024 | 9 | 145 | 55.6 | 0.7 | -0.2 | 22.2 | 22.2 | 4.1 | 2 |
| Witaminy | thai_v4m_m1024 | 9 | 145 | 11.1 | 1.6 | -1.6 | 0.0 | 88.9 | 9.5 | 4 |
| Witaminy | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 9 | 145 | 11.1 | 2.0 | -2.0 | 0.0 | 88.9 | 12.3 | 3 |
| advil | thai_v2 | 222 | 1110 | 91.4 | 0.1 | 0.0 | 5.9 | 2.7 | 1.7 | 1 |
| advil | thai_v2_appfile | 222 | 1110 | 92.3 | 0.1 | 0.0 | 5.9 | 1.8 | 1.5 | 1 |
| advil | thai_v3_m1024 | 222 | 1110 | 0.0 | 4.3 | -4.3 | 0.0 | 100.0 | 85.9 | 5 |
| advil | thai_v4m_m1024 | 222 | 1110 | 90.1 | 0.1 | 0.1 | 9.0 | 0.9 | 2.1 | 2 |
| advil | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 222 | 1110 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| aspirin | thai_v2 | 222 | 1110 | 91.4 | 0.1 | 0.0 | 5.9 | 2.7 | 1.7 | 1 |
| aspirin | thai_v2_appfile | 222 | 1110 | 92.3 | 0.1 | 0.0 | 5.9 | 1.8 | 1.5 | 1 |
| aspirin | thai_v3_m1024 | 222 | 1110 | 0.0 | 4.3 | -4.3 | 0.0 | 100.0 | 85.9 | 5 |
| aspirin | thai_v4m_m1024 | 222 | 1110 | 90.1 | 0.1 | 0.1 | 9.0 | 0.9 | 2.1 | 2 |
| aspirin | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 222 | 1110 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| brown_capsule | thai_v2 | 56 | 187 | 42.9 | 0.8 | -0.1 | 21.4 | 35.7 | 29.7 | 6 |
| brown_capsule | thai_v2_appfile | 56 | 187 | 37.5 | 0.9 | 0.1 | 30.4 | 32.1 | 34.4 | 8 |
| brown_capsule | thai_v3_m1024 | 56 | 187 | 23.2 | 1.5 | -0.3 | 26.8 | 50.0 | 49.1 | 8 |
| brown_capsule | thai_v4m_m1024 | 56 | 187 | 58.9 | 0.5 | -0.1 | 17.9 | 23.2 | 19.9 | 3 |
| brown_capsule | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 56 | 187 | 42.9 | 0.8 | -0.6 | 10.7 | 46.4 | 27.6 | 3 |
| diarrhea | thai_v2 | 222 | 1110 | 91.4 | 0.1 | 0.0 | 5.9 | 2.7 | 1.7 | 1 |
| diarrhea | thai_v2_appfile | 222 | 1110 | 92.3 | 0.1 | 0.0 | 5.9 | 1.8 | 1.5 | 1 |
| diarrhea | thai_v3_m1024 | 222 | 1110 | 0.0 | 4.3 | -4.3 | 0.0 | 100.0 | 85.9 | 5 |
| diarrhea | thai_v4m_m1024 | 222 | 1110 | 90.1 | 0.1 | 0.1 | 9.0 | 0.9 | 2.1 | 2 |
| diarrhea | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 222 | 1110 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| motion | thai_v2 | 222 | 1110 | 91.4 | 0.1 | 0.0 | 5.9 | 2.7 | 1.7 | 1 |
| motion | thai_v2_appfile | 222 | 1110 | 92.3 | 0.1 | 0.0 | 5.9 | 1.8 | 1.5 | 1 |
| motion | thai_v3_m1024 | 222 | 1110 | 0.0 | 4.3 | -4.3 | 0.0 | 100.0 | 85.9 | 5 |
| motion | thai_v4m_m1024 | 222 | 1110 | 90.1 | 0.1 | 0.1 | 9.0 | 0.9 | 2.1 | 2 |
| motion | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 222 | 1110 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| tylenol | thai_v2 | 222 | 1110 | 91.4 | 0.1 | 0.0 | 5.9 | 2.7 | 1.7 | 1 |
| tylenol | thai_v2_appfile | 222 | 1110 | 92.3 | 0.1 | 0.0 | 5.9 | 1.8 | 1.5 | 1 |
| tylenol | thai_v3_m1024 | 222 | 1110 | 0.0 | 4.3 | -4.3 | 0.0 | 100.0 | 85.9 | 5 |
| tylenol | thai_v4m_m1024 | 222 | 1110 | 90.1 | 0.1 | 0.1 | 9.0 | 0.9 | 2.1 | 2 |
| tylenol | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 222 | 1110 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| white_tablet | thai_v2 | 54 | 184 | 40.7 | 0.9 | -0.2 | 18.5 | 40.7 | 34.8 | 6 |
| white_tablet | thai_v2_appfile | 54 | 184 | 40.7 | 0.9 | -0.0 | 22.2 | 37.0 | 35.4 | 8 |
| white_tablet | thai_v3_m1024 | 54 | 184 | 25.9 | 1.4 | -0.4 | 25.9 | 48.1 | 51.2 | 8 |
| white_tablet | thai_v4m_m1024 | 54 | 184 | 59.3 | 0.5 | -0.1 | 16.7 | 24.1 | 21.6 | 3 |
| white_tablet | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 54 | 184 | 37.0 | 0.9 | -0.8 | 5.6 | 57.4 | 31.1 | 3 |
| yellow_tablet | thai_v2 | 55 | 186 | 45.5 | 0.7 | -0.2 | 18.2 | 36.4 | 29.6 | 6 |
| yellow_tablet | thai_v2_appfile | 55 | 186 | 43.6 | 0.8 | -0.0 | 23.6 | 32.7 | 32.6 | 8 |
| yellow_tablet | thai_v3_m1024 | 55 | 186 | 38.2 | 1.2 | -0.5 | 16.4 | 45.5 | 38.5 | 8 |
| yellow_tablet | thai_v4m_m1024 | 55 | 186 | 60.0 | 0.5 | -0.1 | 16.4 | 23.6 | 19.1 | 3 |
| yellow_tablet | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 55 | 186 | 49.1 | 0.8 | -0.7 | 5.5 | 45.5 | 21.4 | 3 |
| numbered drug ids | thai_v2 | 934 | 3217 | 19.7 | 2.0 | -0.8 | 14.7 | 65.6 | 67.4 | 49 |
| numbered drug ids | thai_v2_appfile | 934 | 3217 | 21.7 | 2.1 | -0.4 | 17.5 | 60.8 | 71.7 | 73 |
| numbered drug ids | thai_v3_m1024 | 934 | 3217 | 14.8 | 4.7 | 2.1 | 31.4 | 53.9 | 212.9 | 499 |
| numbered drug ids | thai_v4m_m1024 | 934 | 3217 | 37.4 | 2.0 | 0.5 | 21.2 | 41.4 | 72.7 | 78 |
| numbered drug ids | thai_mix_s1024 (SEEN: 70 % trained on, 15 % validation) | 934 | 3217 | 54.5 | 0.7 | -0.5 | 7.0 | 38.5 | 23.3 | 9 |

## Pill Counting v4, valid

### By pile size

| pile | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–20 | thai_v2 (seen-ish) | 8 | 98 | 37.5 | 2.0 | -1.5 | 25.0 | 37.5 | 15.2 | 12 |
| 1–20 | thai_v2_appfile (seen-ish) | 8 | 98 | 62.5 | 1.8 | -1.0 | 25.0 | 12.5 | 15.4 | 11 |
| 1–20 | thai_v3_m1024 | 8 | 98 | 12.5 | 6.4 | 6.4 | 87.5 | 0.0 | 47.9 | 17 |
| 1–20 | thai_v4m_m1024 (seen-ish) | 8 | 98 | 87.5 | 0.1 | 0.1 | 12.5 | 0.0 | 3.1 | 1 |
| 1–20 | thai_mix_s1024 (seen: validation) | 8 | 98 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 21–50 | thai_v2 (seen-ish) | 141 | 6786 | 59.6 | 1.5 | -1.0 | 9.9 | 30.5 | 2.9 | 20 |
| 21–50 | thai_v2_appfile (seen-ish) | 141 | 6786 | 65.2 | 1.1 | -0.9 | 7.8 | 27.0 | 2.2 | 18 |
| 21–50 | thai_v3_m1024 | 141 | 6786 | 24.8 | 17.2 | 16.6 | 68.1 | 7.1 | 35.2 | 118 |
| 21–50 | thai_v4m_m1024 (seen-ish) | 141 | 6786 | 71.6 | 0.4 | -0.1 | 14.9 | 13.5 | 0.9 | 5 |
| 21–50 | thai_mix_s1024 (seen: validation) | 141 | 6786 | 96.5 | 0.0 | 0.0 | 2.1 | 1.4 | 0.1 | 1 |
| 51–100 | thai_v2 (seen-ish) | 19 | 1183 | 68.4 | 0.6 | -0.2 | 15.8 | 15.8 | 1.2 | 4 |
| 51–100 | thai_v2_appfile (seen-ish) | 19 | 1183 | 78.9 | 0.4 | 0.0 | 15.8 | 5.3 | 0.7 | 4 |
| 51–100 | thai_v3_m1024 | 19 | 1183 | 26.3 | 1.4 | 1.4 | 73.7 | 0.0 | 2.3 | 4 |
| 51–100 | thai_v4m_m1024 (seen-ish) | 19 | 1183 | 73.7 | 0.3 | 0.3 | 26.3 | 0.0 | 0.5 | 1 |
| 51–100 | thai_mix_s1024 (seen: validation) | 19 | 1183 | 94.7 | 0.1 | -0.1 | 0.0 | 5.3 | 0.1 | 1 |
| 100+ | thai_v2 (seen-ish) | 14 | 2289 | 57.1 | 6.4 | -6.4 | 0.0 | 42.9 | 3.7 | 37 |
| 100+ | thai_v2_appfile (seen-ish) | 14 | 2289 | 57.1 | 4.1 | -4.1 | 0.0 | 42.9 | 2.4 | 28 |
| 100+ | thai_v3_m1024 | 14 | 2289 | 14.3 | 2.7 | 2.7 | 85.7 | 0.0 | 1.6 | 10 |
| 100+ | thai_v4m_m1024 (seen-ish) | 14 | 2289 | 92.9 | 0.2 | -0.2 | 0.0 | 7.1 | 0.1 | 3 |
| 100+ | thai_mix_s1024 (seen: validation) | 14 | 2289 | 78.6 | 0.6 | 0.1 | 14.3 | 7.1 | 0.3 | 4 |

## pill count v1, tray video frames

### By pile size

| pile | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–20 | thai_v2 | 1 | 10 | 0.0 | 64.0 | 64.0 | 100.0 | 0.0 | 640.0 | 64 |
| 1–20 | thai_v2_appfile | 1 | 10 | 0.0 | 64.0 | 64.0 | 100.0 | 0.0 | 640.0 | 64 |
| 1–20 | thai_v3_m1024 | 1 | 10 | 0.0 | 66.0 | 66.0 | 100.0 | 0.0 | 660.0 | 66 |
| 1–20 | thai_v4m_m1024 | 1 | 10 | 0.0 | 64.0 | 64.0 | 100.0 | 0.0 | 640.0 | 64 |
| 1–20 | thai_mix_s1024 | 1 | 10 | 0.0 | 55.0 | 55.0 | 100.0 | 0.0 | 550.0 | 55 |
| 21–50 | thai_v2 | 26 | 920 | 0.0 | 26.4 | -26.4 | 0.0 | 100.0 | 77.2 | 36 |
| 21–50 | thai_v2_appfile | 26 | 920 | 0.0 | 26.0 | -26.0 | 0.0 | 100.0 | 76.1 | 35 |
| 21–50 | thai_v3_m1024 | 26 | 920 | 0.0 | 12.5 | -12.5 | 0.0 | 100.0 | 35.6 | 23 |
| 21–50 | thai_v4m_m1024 | 26 | 920 | 0.0 | 21.0 | -21.0 | 0.0 | 100.0 | 61.3 | 29 |
| 21–50 | thai_mix_s1024 | 26 | 920 | 0.0 | 29.3 | -29.3 | 0.0 | 100.0 | 84.2 | 42 |
| 51–100 | thai_v2 | 52 | 3947 | 0.0 | 30.0 | -30.0 | 0.0 | 100.0 | 39.4 | 55 |
| 51–100 | thai_v2_appfile | 52 | 3947 | 0.0 | 29.0 | -29.0 | 0.0 | 100.0 | 38.1 | 54 |
| 51–100 | thai_v3_m1024 | 52 | 3947 | 9.6 | 9.7 | -9.6 | 3.8 | 86.5 | 12.5 | 28 |
| 51–100 | thai_v4m_m1024 | 52 | 3947 | 0.0 | 20.9 | -20.9 | 0.0 | 100.0 | 27.0 | 45 |
| 51–100 | thai_mix_s1024 | 52 | 3947 | 0.0 | 37.5 | -37.5 | 0.0 | 100.0 | 49.4 | 64 |
| 100+ | thai_v2 | 33 | 3961 | 0.0 | 72.2 | -72.2 | 0.0 | 100.0 | 58.6 | 129 |
| 100+ | thai_v2_appfile | 33 | 3961 | 0.0 | 70.4 | -70.4 | 0.0 | 100.0 | 57.1 | 124 |
| 100+ | thai_v3_m1024 | 33 | 3961 | 0.0 | 28.6 | -28.6 | 0.0 | 100.0 | 22.8 | 57 |
| 100+ | thai_v4m_m1024 | 33 | 3961 | 0.0 | 55.0 | -55.0 | 0.0 | 100.0 | 44.2 | 102 |
| 100+ | thai_mix_s1024 | 33 | 3961 | 0.0 | 78.3 | -78.3 | 0.0 | 100.0 | 63.4 | 132 |

### Per video

| video | model | images | pills | exact % | MAE | bias | over % | under % | rel err % | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| WIN_20230504_09_20_21_Pro | thai_v2 | 47 | 2516 | 0.0 | 22.6 | -19.9 | 2.1 | 97.9 | 62.3 | 64 |
| WIN_20230504_09_20_21_Pro | thai_v2_appfile | 47 | 2516 | 0.0 | 21.9 | -19.2 | 2.1 | 97.9 | 61.1 | 64 |
| WIN_20230504_09_20_21_Pro | thai_v3_m1024 | 47 | 2516 | 10.6 | 8.3 | -5.4 | 4.3 | 85.1 | 31.5 | 66 |
| WIN_20230504_09_20_21_Pro | thai_v4m_m1024 | 47 | 2516 | 0.0 | 16.0 | -13.2 | 2.1 | 97.9 | 48.8 | 64 |
| WIN_20230504_09_20_21_Pro | thai_mix_s1024 | 47 | 2516 | 0.0 | 28.0 | -25.7 | 2.1 | 97.9 | 69.5 | 55 |
| vid_1 | thai_v2 | 8 | 814 | 0.0 | 44.8 | -44.8 | 0.0 | 100.0 | 44.0 | 48 |
| vid_1 | thai_v2_appfile | 8 | 814 | 0.0 | 43.4 | -43.4 | 0.0 | 100.0 | 42.7 | 48 |
| vid_1 | thai_v3_m1024 | 8 | 814 | 0.0 | 19.5 | -19.5 | 0.0 | 100.0 | 19.1 | 23 |
| vid_1 | thai_v4m_m1024 | 8 | 814 | 0.0 | 34.8 | -34.8 | 0.0 | 100.0 | 34.3 | 43 |
| vid_1 | thai_mix_s1024 | 8 | 814 | 0.0 | 45.8 | -45.8 | 0.0 | 100.0 | 45.0 | 50 |
| vid_2 | thai_v2 | 7 | 686 | 0.0 | 51.4 | -51.4 | 0.0 | 100.0 | 52.4 | 61 |
| vid_2 | thai_v2_appfile | 7 | 686 | 0.0 | 50.4 | -50.4 | 0.0 | 100.0 | 51.4 | 60 |
| vid_2 | thai_v3_m1024 | 7 | 686 | 0.0 | 23.9 | -23.9 | 0.0 | 100.0 | 24.3 | 33 |
| vid_2 | thai_v4m_m1024 | 7 | 686 | 0.0 | 41.9 | -41.9 | 0.0 | 100.0 | 42.7 | 49 |
| vid_2 | thai_mix_s1024 | 7 | 686 | 0.0 | 60.3 | -60.3 | 0.0 | 100.0 | 61.5 | 69 |
| vid_4 | thai_v2 | 7 | 748 | 0.0 | 53.3 | -53.3 | 0.0 | 100.0 | 49.9 | 61 |
| vid_4 | thai_v2_appfile | 7 | 748 | 0.0 | 51.7 | -51.7 | 0.0 | 100.0 | 48.4 | 60 |
| vid_4 | thai_v3_m1024 | 7 | 748 | 0.0 | 19.3 | -19.3 | 0.0 | 100.0 | 18.0 | 24 |
| vid_4 | thai_v4m_m1024 | 7 | 748 | 0.0 | 39.1 | -39.1 | 0.0 | 100.0 | 36.6 | 44 |
| vid_4 | thai_mix_s1024 | 7 | 748 | 0.0 | 62.0 | -62.0 | 0.0 | 100.0 | 58.0 | 70 |
| vid_5 | thai_v2 | 8 | 341 | 0.0 | 31.6 | -31.6 | 0.0 | 100.0 | 74.1 | 36 |
| vid_5 | thai_v2_appfile | 8 | 341 | 0.0 | 31.0 | -31.0 | 0.0 | 100.0 | 72.7 | 35 |
| vid_5 | thai_v3_m1024 | 8 | 341 | 0.0 | 17.0 | -17.0 | 0.0 | 100.0 | 39.6 | 23 |
| vid_5 | thai_v4m_m1024 | 8 | 341 | 0.0 | 24.8 | -24.8 | 0.0 | 100.0 | 58.0 | 29 |
| vid_5 | thai_mix_s1024 | 8 | 341 | 0.0 | 34.8 | -34.8 | 0.0 | 100.0 | 81.3 | 42 |
| vid_6 | thai_v2 | 8 | 624 | 0.0 | 40.4 | -40.4 | 0.0 | 100.0 | 51.7 | 54 |
| vid_6 | thai_v2_appfile | 8 | 624 | 0.0 | 39.2 | -39.2 | 0.0 | 100.0 | 50.3 | 52 |
| vid_6 | thai_v3_m1024 | 8 | 624 | 0.0 | 20.2 | -20.2 | 0.0 | 100.0 | 26.0 | 22 |
| vid_6 | thai_v4m_m1024 | 8 | 624 | 0.0 | 30.4 | -30.4 | 0.0 | 100.0 | 39.0 | 38 |
| vid_6 | thai_mix_s1024 | 8 | 624 | 0.0 | 43.2 | -43.2 | 0.0 | 100.0 | 55.4 | 54 |
| vid_7 | thai_v2 | 9 | 717 | 0.0 | 35.8 | -35.8 | 0.0 | 100.0 | 45.0 | 40 |
| vid_7 | thai_v2_appfile | 9 | 717 | 0.0 | 34.7 | -34.7 | 0.0 | 100.0 | 43.6 | 38 |
| vid_7 | thai_v3_m1024 | 9 | 717 | 0.0 | 6.4 | -6.2 | 11.1 | 88.9 | 8.2 | 13 |
| vid_7 | thai_v4m_m1024 | 9 | 717 | 0.0 | 24.3 | -24.3 | 0.0 | 100.0 | 30.6 | 29 |
| vid_7 | thai_mix_s1024 | 9 | 717 | 0.0 | 47.7 | -47.7 | 0.0 | 100.0 | 59.9 | 57 |
| vid_8 | thai_v2 | 9 | 952 | 0.0 | 68.4 | -68.4 | 0.0 | 100.0 | 64.8 | 76 |
| vid_8 | thai_v2_appfile | 9 | 952 | 0.0 | 67.2 | -67.2 | 0.0 | 100.0 | 63.6 | 75 |
| vid_8 | thai_v3_m1024 | 9 | 952 | 0.0 | 20.4 | -20.4 | 0.0 | 100.0 | 19.3 | 24 |
| vid_8 | thai_v4m_m1024 | 9 | 952 | 0.0 | 48.8 | -48.8 | 0.0 | 100.0 | 46.1 | 51 |
| vid_8 | thai_mix_s1024 | 9 | 952 | 0.0 | 72.0 | -72.0 | 0.0 | 100.0 | 68.1 | 78 |
| vid_9 | thai_v2 | 9 | 1440 | 0.0 | 113.6 | -113.6 | 0.0 | 100.0 | 71.0 | 129 |
| vid_9 | thai_v2_appfile | 9 | 1440 | 0.0 | 110.9 | -110.9 | 0.0 | 100.0 | 69.4 | 124 |
| vid_9 | thai_v3_m1024 | 9 | 1440 | 0.0 | 50.6 | -50.6 | 0.0 | 100.0 | 31.6 | 57 |
| vid_9 | thai_v4m_m1024 | 9 | 1440 | 0.0 | 90.8 | -90.8 | 0.0 | 100.0 | 56.8 | 102 |
| vid_9 | thai_mix_s1024 | 9 | 1440 | 0.0 | 123.4 | -123.4 | 0.0 | 100.0 | 77.2 | 132 |

## MAE at other confidence cut-offs (app IoU; for later, not re-tuned)

| set | model | conf 0.1 | conf 0.2 | conf 0.3 | conf 0.4 | conf 0.5 | conf 0.6 | conf 0.7 |
|---|---|---|---|---|---|---|---|---|
| pc2_heldout_test | thai_v2 | 4.46 | 3.01 | 2.40 | 2.10 | 1.90 | 1.91 | 2.24 |
| pc2_heldout_test | thai_v2_appfile | 4.25 | 2.92 | 2.35 | 2.06 | 1.89 | 1.91 | 2.24 |
| pc2_heldout_test | thai_v3_m1024 | 5.08 | 3.90 | 3.42 | 3.29 | 3.19 | 3.20 | 3.39 |
| pc2_heldout_test | thai_v4m_m1024 | 5.29 | 3.42 | 2.58 | 2.17 | 1.83 | 1.62 | 2.04 |
| pc2_heldout_test | thai_mix_s1024 | 1.46 | 0.78 | 0.56 | 0.48 | 0.57 | 0.85 | 1.61 |
| pillcounter_v2 | thai_v2 | 4.03 | 2.70 | 2.12 | 1.81 | 1.63 | 1.59 | 2.01 |
| pillcounter_v2 | thai_v2_appfile | 3.85 | 2.64 | 2.10 | 1.80 | 1.63 | 1.59 | 2.01 |
| pillcounter_v2 | thai_v3_m1024 | 5.77 | 4.65 | 4.15 | 3.73 | 3.40 | 3.31 | 3.43 |
| pillcounter_v2 | thai_v4m_m1024 | 5.72 | 3.57 | 2.67 | 2.16 | 1.78 | 1.52 | 1.79 |
| pillcounter_v2 | thai_mix_s1024 | 1.30 | 0.76 | 0.56 | 0.47 | 0.53 | 0.75 | 1.47 |
| v4_valid | thai_v2 | 2.25 | 1.43 | 1.15 | 1.03 | 1.34 | 2.86 | 39.18 |
| v4_valid | thai_v2_appfile | 1.49 | 1.07 | 0.91 | 0.91 | 1.31 | 2.93 | 39.19 |
| v4_valid | thai_v3_m1024 | 17.47 | 14.84 | 13.25 | 11.98 | 11.00 | 10.05 | 9.53 |
| v4_valid | thai_v4m_m1024 | 2.63 | 1.30 | 0.91 | 0.66 | 0.49 | 0.37 | 0.68 |
| v4_valid | thai_mix_s1024 | 0.79 | 0.34 | 0.19 | 0.13 | 0.08 | 0.07 | 0.20 |
| v1_trays | thai_v2 | 30.02 | 33.49 | 35.78 | 38.35 | 40.71 | 43.47 | 67.62 |
| v1_trays | thai_v2_appfile | 30.89 | 34.00 | 36.12 | 38.57 | 40.79 | 43.50 | 67.62 |
| v1_trays | thai_v3_m1024 | 11.12 | 14.98 | 17.91 | 20.58 | 23.59 | 27.75 | 33.90 |
| v1_trays | thai_v4m_m1024 | 15.01 | 19.21 | 21.97 | 24.61 | 27.58 | 31.35 | 38.44 |
| v1_trays | thai_mix_s1024 | 24.61 | 29.26 | 32.27 | 36.27 | 42.86 | 53.80 | 66.68 |

## Data notes

| set | polygon_line | odd_line | unreadable_line | box_over_10pct | zero_labels | missing_label_file |
|---|---|---|---|---|---|---|
| pillcounter_v2 | 0 | 0 | 0 | 15 | 0 | 0 |
| v1_trays | 6 | 0 | 0 | 0 | 0 | 0 |
| v4_valid | 0 | 0 | 0 | 0 | 0 | 0 |

Images with label boxes > 10 % of the image: VAIPE_P_185_37 (pillcounter_v2, 1), VAIPE_P_185_55 (pillcounter_v2, 1), VAIPE_P_267_15 (pillcounter_v2, 1), VAIPE_P_412_3 (pillcounter_v2, 1), VAIPE_P_412_7 (pillcounter_v2, 1), VAIPE_P_463_6 (pillcounter_v2, 1), VAIPE_P_537_7 (pillcounter_v2, 1), VAIPE_P_658_7 (pillcounter_v2, 1), VAIPE_P_76_10 (pillcounter_v2, 1), VAIPE_P_76_43 (pillcounter_v2, 1), VAIPE_P_76_47 (pillcounter_v2, 1), VAIPE_P_89_2 (pillcounter_v2, 2), VAIPE_P_99_7 (pillcounter_v2, 1), VAIPE_P_167_20 (pillcounter_v2, 1)

32 images share a source name with another image in the same set (Roboflow re-uploads): VAIPE_P_429_1, VAIPE_P_433_3, VAIPE_P_434_0, VAIPE_P_434_11, VAIPE_P_434_4, VAIPE_P_435_10, VAIPE_P_435_2, VAIPE_P_435_6, VAIPE_P_438_2, VAIPE_P_438_6, VAIPE_P_439_0, VAIPE_P_43_1, VAIPE_P_43_5, VAIPE_P_441_8, VAIPE_P_449_7, VAIPE_P_44_6

Pile sizes: pillcounter_v2 1–18 (median 4); v4_valid 4–180 (median 50); v1_trays 10–175 (median 77)
