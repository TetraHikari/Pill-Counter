"""Writes notebooks/train_counter_mix_colab.ipynb (data from tools/make_mixed_zip.py)."""
import json
CELLS = []
md = lambda s: CELLS.append(('markdown', s.strip('\n')))
code = lambda s: CELLS.append(('code', s.strip('\n')))

md(r'''
# PillTally: mixed-data counter (our photos + public pills) on A100

`thai_v3_m1024` (our 3 medicines only) counts other medicines badly; `thai_v2` (via 71k public pills) does better but is
weaker on our crowded trays. This run trains one counter on **both**, so it should be good on our trays and on other medicines.

| Data | Train | Validation (during training) |
|---|---|---|
| Our photos (v3 split, from `pilltally_real_v3_for_colab.zip`) | 151 photos, **each listed 6×** (≈ 32 % of an epoch); crowded trays s25, s27, s28 included | 60 (s06, s13, s18, s23, s26) |
| Pill Counting v4 | 1,054 | 182 (`valid`) |
| Pill-Counter v2 (110 drugs), re-split by source group | 879 | 189 |
| **Held out, never used here** | our test set (45) and the Pill-Counter v2 test part (187) | |

Every public label is class 0 `pill` (polygons → boxes); public images are already shrunk to 1280 px. Validation = one combined list (431 images).

Recipe = the two-stage recipe of `thai_v3` / `thai_v4m`: stage 1 frozen layers 0–9, SGD 0.005, 30 epochs → stage 2 unfrozen,
SGD 0.001, up to 60 epochs, patience 25; imgsz 1024, max_det 500, degrees 180, flips, scale 0.3, hsv_v 0.4, hsv_s 0.5, translate 0.1.

| Variant | Starts from | Saves |
|---|---|---|
| `mix_s1024` (YOLO11s) | `v4_pill_model.pt` | `MyDrive/PillTally/thai_mix_s1024.pt` |
| `mix_m1024` (YOLO11m) | `runs/v4m_pill_model/weights/best.pt` (the v4m run) | `MyDrive/PillTally/thai_mix_m1024.pt` |

**Before running**, check Google Drive → `MyDrive/PillTally/` has:
- `pilltally_mixed_for_colab.zip` (**new, ~470 MB**: upload it)
- `pilltally_real_v3_for_colab.zip` (there from the v3 run)
- `v4_pill_model.pt` and `runs/v4m_pill_model/weights/best.pt` (there from earlier runs)

Runtime → **A100 GPU** → Run all. Expected on an A100: about **2–3 h for `mix_s1024`** and **3.5–5 h for `mix_m1024`**
(≈ 2,840 images per epoch, up to 90 epochs each; stage 2 often stops early). To split the work over two sessions,
set `RUN` in the training cell to one variant per session. Training writes to Drive, so a disconnect doesn't lose the run
(resume cells at the bottom).

Dataset credits: *Pill Counting* (Roboflow Universe, CC BY 4.0); *Pill-Counter* (dubfai, Roboflow Universe, public domain).
''')

code(r'''
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/PillTally'
!nvidia-smi --query-gpu=name,memory.total --format=csv
''')

code(r'''
!pip -q install ultralytics
import os
if not os.path.exists('/content/mixed/train.txt'):
    !unzip -q -o "{DRIVE}/pilltally_mixed_for_colab.zip" -d /content
if not os.path.exists('/content/real/train.txt'):
    !unzip -q -o "{DRIVE}/pilltally_real_v3_for_colab.zip" -d /content
# every listed image and its label must be there (our photos come from the v3 zip, the rest from the mixed zip)
for name in ('train', 'val', 'pc2_test'):
    lst = [l.strip() for l in open(f'/content/mixed/{name}.txt') if l.strip()]
    bad = [p for p in lst if not os.path.exists(p) or not os.path.exists(p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')]
    print(f'{name}: {len(lst)} entries ({len(set(lst))} images, {sum("/real/" in p for p in lst)} entries ours), {len(bad)} missing')
    assert not bad, bad[:5]
print(open('/content/mixed/mixed.yaml').read())
''')

md(r'''
## Train: two stages per variant
`pc2_test.txt` (the held-out Pill-Counter v2 part) is not used here; it is for `notebooks/eval_other_pills_colab.ipynb`.
''')

code(r'''
import shutil
from ultralytics import YOLO

VARIANTS = {  # name: (start weights, batch)
    'mix_s1024': (f'{DRIVE}/v4_pill_model.pt', 16),
    'mix_m1024': (f'{DRIVE}/runs/v4m_pill_model/weights/best.pt', 8),
}

def common(name):
    return dict(data='/content/mixed/mixed.yaml', imgsz=1024, batch=VARIANTS[name][1], cache=False, workers=8,
                project=f'{DRIVE}/runs', exist_ok=True, max_det=500, optimizer='SGD', patience=25,
                degrees=180, flipud=0.5, fliplr=0.5, scale=0.3, hsv_v=0.4, hsv_s=0.5, translate=0.1)

def stage1(name):
    YOLO(VARIANTS[name][0]).train(epochs=30, freeze=10, lr0=0.005, name=f'{name}_stage1', **common(name))

def stage2(name):
    YOLO(f'{DRIVE}/runs/{name}_stage1/weights/best.pt').train(epochs=60, lr0=0.001, warmup_epochs=0,
                                                               name=f'{name}_stage2', **common(name))

def save(name):
    shutil.copy(f'{DRIVE}/runs/{name}_stage2/weights/best.pt', f'{DRIVE}/thai_{name}.pt')
    print('Saved', f'{DRIVE}/thai_{name}.pt')
''')

code(r'''
RUN = ['mix_s1024']   # small model first; next session: ['mix_m1024']
for name in RUN:
    stage1(name)
    stage2(name)
    save(name)
''')

md(r'''
## Done
Download **`MyDrive/PillTally/thai_mix_s1024.pt`** and **`thai_mix_m1024.pt`** into `results/models/`.
They are compared afterwards: thresholds tuned on our val photos (`tools/compare_counters.py`), then
`notebooks/eval_other_pills_colab.ipynb` (held-out Pill-Counter v2 test part).
''')

md(r'''
### If Colab disconnects
Run the first 2 cells (mount, install + unzip) and the cell with `VARIANTS` / `stage1` / `stage2` / `save` (definitions only,
it doesn't train), set `NAME` below to the variant that was running, then run **one** of the two cells.
''')

code(r'''
NAME = 'mix_s1024'
# stage 1 was interrupted:
YOLO(f'{DRIVE}/runs/{NAME}_stage1/weights/last.pt').train(resume=True)
stage2(NAME)
save(NAME)
''')

code(r'''
# stage 2 was interrupted:
YOLO(f'{DRIVE}/runs/{NAME}_stage2/weights/last.pt').train(resume=True)
save(NAME)
''')

nb = {'cells': [], 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []},
                                'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
for kind, src in CELLS:
    c = {'cell_type': kind, 'metadata': {}, 'source': src.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    nb['cells'].append(c)
import os
json.dump(nb, open(os.path.join(os.path.dirname(__file__), 'train_counter_mix_colab.ipynb'), 'w'), indent=1, ensure_ascii=False)
