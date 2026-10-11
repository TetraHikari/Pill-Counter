"""Build pilltally_mixed_for_colab.zip for notebooks/train_counter_mix_colab.ipynb.

1. Re-splits Pill-Counter v2 (all 1,255 images) by source group into train / val / test (70 / 15 / 15 % of images,
   stratified by single-drug / multi-drug / capsule) and writes data/open/pillcounter_v2_split.csv.
   Groups (Roboflow's own splits mix photos of one scene, so they are ignored):
   - VAIPE_P_<id>_<n>  -> VAIPE_P_<id> (one prescription; this also keeps the 16 duplicate source names together)
   - DSC00947-DSC01189 -> one group: one session of the same 5 pills, nudged a little between frames (no clean
     break anywhere), so it can't be split without near-duplicates crossing parts. Pinned to train.
   - IMG_3979-4041 (640 px) and IMG_4044-4076 (3024x4032) are two sessions of the same 3 pills; within each, a run of
     consecutive numbers with the same pills on the tray is one setup = one group (needed so every part gets capsules).
   - potas-i-witaminy_mp4-* -> one group (frames of one video).
2. Copies Pill-Counter v2 and Pill Counting v4 into a temp folder: images shrunk to 1280 px (long side),
   every label line -> class 0, polygons -> boxes.
3. Writes the lists used on Colab (absolute /content paths): train.txt (v4 train + PC2 train + our train photos
   repeated REPEAT times), val.txt (our val + v4 valid + PC2 val), pc2_test.txt (held out), mixed.yaml.
   Our photos come from pilltally_real_v3_for_colab.zip (already in Drive); only its lists are read here.

data/open/ is only read, except for writing the split CSV.

    .venv/bin/python tools/make_mixed_zip.py            # split + zip
    .venv/bin/python tools/make_mixed_zip.py --split    # split CSV + counts only
"""
import collections, csv, glob, os, random, re, shutil, sys, tempfile, zipfile
from concurrent.futures import ThreadPoolExecutor
import yaml
from PIL import Image, ImageOps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OPEN = os.path.join(ROOT, 'data/open')
PC2 = os.path.join(OPEN, 'Pill-Counter.v2i.yolov11')
V4 = os.path.join(OPEN, 'Pill Counting.v4i.yolov11')
REAL_ZIP = os.path.join(ROOT, 'pilltally_real_v3_for_colab.zip')
SPLIT_CSV = os.path.join(OPEN, 'pillcounter_v2_split.csv')
OUT = os.path.join(ROOT, 'pilltally_mixed_for_colab.zip')
COLAB = '/content'               # mixed zip unpacks to /content/mixed, ours to /content/real
FRACS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEED = 0
CROWDED = ('s25', 's27', 's28')


def base_name(img):  # Roboflow name without the .rf.<hash> suffix
    return re.sub(r'_(jpg|jpeg|png|bmp)\.rf\.[0-9a-f]+\.\w+$', '', os.path.basename(img), flags=re.I)


def label_of(img):
    d, f = os.path.split(img)
    return os.path.join(os.path.dirname(d), 'labels', os.path.splitext(f)[0] + '.txt')


def pc2_images():
    rows = []
    names = yaml.safe_load(open(os.path.join(PC2, 'data.yaml')))['names']
    for img in sorted(glob.glob(f'{PC2}/*/images/*')):
        cls = [names[int(l.split()[0])] for l in open(label_of(img)) if l.strip()]
        rows.append(dict(image=os.path.relpath(img, PC2), name=base_name(img), classes=cls,
                         size=Image.open(img).size))
    return rows


def group_of(rows):
    """Adds 'group' to each row (see module docstring)."""
    img_rows = sorted((r for r in rows if r['name'].startswith('IMG_')), key=lambda r: int(r['name'][4:]))
    prev, n = None, 0
    for r in img_rows:  # runs of consecutive numbers with the same pills, within one session (= image size)
        key = (r['size'], tuple(sorted(collections.Counter(r['classes']).items())))
        if key != prev:
            n, prev = n + 1, key
            sess = '640' if r['size'] == (640, 640) else 'large'
        r['group'] = f'IMG_{sess}_setup{n:02d}'
    for r in rows:
        nm = r['name']
        if nm.startswith('VAIPE_P_'):
            r['group'] = re.match(r'(VAIPE_P_\d+)_\d+$', nm).group(1)
        elif nm.startswith('DSC'):
            r['group'] = 'DSC_session'
        elif nm.startswith('potas-i-witaminy'):
            r['group'] = 'potas-i-witaminy_video'
        elif not nm.startswith('IMG_'):
            raise ValueError('unknown source name ' + nm)


def stratum_of(rs):
    if any('brown_capsule' in r['classes'] for r in rs):
        return 'capsule'
    return 'multi-drug' if any(len(set(r['classes'])) > 1 for r in rs) else 'single-drug'


def make_split():
    rows = pc2_images()
    group_of(rows)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['group']].append(r)
    strata = collections.defaultdict(list)
    for g, rs in groups.items():
        strata[stratum_of(rs)].append(g)
    split_of = {'DSC_session': 'train'}
    rng = random.Random(SEED)
    for st, gs in strata.items():
        total = sum(len(groups[g]) for g in gs)
        have = collections.Counter({p: 0 for p in FRACS})
        for g in gs:
            if g in split_of:
                have[split_of[g]] += len(groups[g])
        free = [g for g in gs if g not in split_of]
        rng.shuffle(free)
        free.sort(key=lambda g: -len(groups[g]))      # big groups first, then fill the gaps
        for g in free:
            p = max(FRACS, key=lambda p: FRACS[p] * total - have[p])
            split_of[g] = p
            have[p] += len(groups[g])
    with open(SPLIT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'group', 'split', 'stratum'])
        for r in rows:
            w.writerow([r['image'], r['group'], split_of[r['group']], stratum_of(groups[r['group']])])
    # report
    print(f'{len(rows)} images, {len(groups)} groups -> {os.path.relpath(SPLIT_CSV, ROOT)}')
    dup = [n for n, c in collections.Counter(r['name'] for r in rows).items() if c > 1]
    assert all(len({split_of[r['group']] for r in rows if r['name'] == n}) == 1 for n in dup)
    print(f'{len(dup)} duplicate source names, each kept in one part')
    print(f"{'stratum':<13}" + ''.join(f'{p:>18}' for p in FRACS))
    for st in sorted(strata) + ['all']:
        gs = strata[st] if st != 'all' else list(groups)
        cells = []
        for p in FRACS:
            pg = [g for g in gs if split_of[g] == p]
            cells.append(f'{len(pg)} gr / {sum(len(groups[g]) for g in pg)} img')
        print(f'{st:<13}' + ''.join(f'{c:>18}' for c in cells))
    return {r['image']: split_of[r['group']] for r in rows}


def shrink_copy(src, dst, long_side=1280):
    im = Image.open(src)
    if max(im.size) > long_side or im.getexif().get(274, 1) != 1:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((long_side, long_side))
        im.convert('RGB').save(dst, quality=92)
    else:
        shutil.copy(src, dst)


def to_pill_boxes(src, dst):
    """Every line -> class 0; polygons -> their bounding box (clipped to the image)."""
    out = []
    for line in open(src):
        v = [float(x) for x in line.split()[1:]]
        if not v:
            continue
        if len(v) == 4:
            x, y, w, h = v
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        elif len(v) >= 6 and len(v) % 2 == 0:
            x1, x2, y1, y2 = min(v[0::2]), max(v[0::2]), min(v[1::2]), max(v[1::2])
        else:
            raise ValueError(f'odd label line in {src}: {line!r}')
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, 1), min(y2, 1)
        if x2 > x1 and y2 > y1:
            out.append(f'0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}')
    open(dst, 'w').write('\n'.join(out) + ('\n' if out else ''))


def real_lists():
    with zipfile.ZipFile(REAL_ZIP) as z:
        lst = {s: [f'{COLAB}/real/' + l.strip()[2:] for l in z.read(f'real/{s}.txt').decode().split('\n') if l.strip()]
               for s in ('train', 'val')}
    for c in CROWDED:
        assert any(os.path.basename(p).startswith(c + '_') for p in lst['train']), c + ' missing from our train'
    return lst


def main():
    split = make_split()
    if '--split' in sys.argv:
        return
    tmp = tempfile.mkdtemp(prefix='mixed_')
    jobs, lists = [], collections.defaultdict(list)
    for img_rel, part in split.items():   # Pill-Counter v2 -> pc2/<new split>/
        jobs.append((os.path.join(PC2, img_rel), f'pc2/{part}', f'pc2_{part}'))
    for src_split, part in (('train', 'train'), ('valid', 'valid')):
        for img in sorted(glob.glob(f'{V4}/{src_split}/images/*')):
            jobs.append((img, f'v4/{part}', f'v4_{part}'))

    def do(job):
        img, folder, _ = job
        f = os.path.basename(img)
        shrink_copy(img, f'{tmp}/{folder}/images/{f}')
        to_pill_boxes(label_of(img), f'{tmp}/{folder}/labels/{os.path.splitext(f)[0]}.txt')
    for _, folder, _ in jobs:
        for sub in ('images', 'labels'):
            os.makedirs(f'{tmp}/{folder}/{sub}', exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(do, jobs))
    for img, folder, key in jobs:
        lists[key].append(f'{COLAB}/mixed/{folder}/images/{os.path.basename(img)}')

    real = real_lists()
    public_train = len(lists['v4_train']) + len(lists['pc2_train'])
    repeat = max(1, round(public_train / (2 * len(real['train']))))   # ours = about 1/3 of each epoch
    train = lists['pc2_train'] + lists['v4_train'] + real['train'] * repeat
    val = lists['pc2_val'] + real['val'] + lists['v4_valid']
    for name, lst in (('train', train), ('val', val), ('pc2_test', lists['pc2_test'])):
        open(f'{tmp}/{name}.txt', 'w').write('\n'.join(lst) + '\n')
    yaml.safe_dump({'path': f'{COLAB}/mixed', 'train': 'train.txt', 'val': 'val.txt', 'names': {0: 'pill'}},
                   open(f'{tmp}/mixed.yaml', 'w'), sort_keys=False)
    shutil.copy(SPLIT_CSV, tmp)
    shutil.copy(os.path.join(PC2, 'data.yaml'), f'{tmp}/pillcounter_v2_data.yaml')
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_STORED) as z:  # JPEGs don't compress further
        for f in sorted(glob.glob(f'{tmp}/**/*', recursive=True)):
            if os.path.isfile(f):
                z.write(f, os.path.join('mixed', os.path.relpath(f, tmp)))
    shutil.rmtree(tmp)
    n_ours = len(real['train']) * repeat
    print(f'train list: {len(lists["v4_train"])} v4 + {len(lists["pc2_train"])} PC2 + {len(real["train"])} ours x{repeat} '
          f'= {len(train)} entries (ours {100 * n_ours / len(train):.0f} % of an epoch)')
    print(f'val list: {len(real["val"])} ours + {len(lists["v4_valid"])} v4 + {len(lists["pc2_val"])} PC2 = {len(val)}')
    print(f'held-out PC2 test: {len(lists["pc2_test"])}')
    print('->', OUT, f'{os.path.getsize(OUT) / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
