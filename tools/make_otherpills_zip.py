"""Build pilltally_otherpills_for_colab.zip for notebooks/eval_other_pills_colab.ipynb.

Copies Pill-Counter v2 (all splits) and the pill count v1 tray-video frames (image_vid*, WIN_*) into a temp
folder, shrinks images to 1280 px on the long side (labels are normalized, so unchanged), and zips them.
data/open/ is only read.

    .venv/bin/python tools/make_otherpills_zip.py
"""
import glob, os, shutil, tempfile, zipfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps

ROOT = os.path.join(os.path.dirname(__file__), '..')
OPEN = os.path.join(ROOT, 'data/open')
OUT = os.path.join(ROOT, 'pilltally_otherpills_for_colab.zip')
SETS = {  # name in zip: (source folder, keep this image?)
    'pillcounter_v2': ('Pill-Counter.v2i.yolov11', lambda f: True),
    'v1_trays': ('pill count.v1i.yolov11', lambda f: 'image_vid' in f or 'WIN_' in f),
}


def shrink_copy(src, dst, long_side=1280):
    im = Image.open(src)
    if max(im.size) > long_side or im.getexif().get(274, 1) != 1:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((long_side, long_side))
        im.convert('RGB').save(dst, quality=92)
    else:
        shutil.copy(src, dst)


def main():
    tmp = tempfile.mkdtemp(prefix='otherpills_')
    jobs = []
    for name, (folder, keep) in SETS.items():
        src_root = os.path.join(OPEN, folder)
        shutil.copy(os.path.join(src_root, 'data.yaml'), os.path.join(tmp, name + '_data.yaml'))
        for split in ('train', 'valid', 'test'):
            for img in sorted(glob.glob(f'{src_root}/{split}/images/*')):
                if not keep(os.path.basename(img)):
                    continue
                lab = f'{src_root}/{split}/labels/' + os.path.splitext(os.path.basename(img))[0] + '.txt'
                for sub in ('images', 'labels'):
                    os.makedirs(f'{tmp}/{name}/{split}/{sub}', exist_ok=True)
                shutil.copy(lab, f'{tmp}/{name}/{split}/labels/')
                jobs.append((img, f'{tmp}/{name}/{split}/images/' + os.path.basename(img)))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: shrink_copy(*j), jobs))
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_STORED) as z:  # JPEGs don't compress further
        for f in sorted(glob.glob(f'{tmp}/**/*', recursive=True)):
            if os.path.isfile(f):
                z.write(f, os.path.join('otherpills', os.path.relpath(f, tmp)))
    shutil.rmtree(tmp)
    print(len(jobs), 'images ->', OUT, f'{os.path.getsize(OUT) / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
