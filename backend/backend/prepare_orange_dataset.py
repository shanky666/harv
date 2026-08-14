import os, sys, shutil, argparse, zipfile
from pathlib import Path

CLASS_MAP = {
    'healthy': 'Better',
    'scab': 'Good',
    'blackspot': 'Reject',
    'black spot': 'Reject',
    'canker': 'Reject',
    'greening': 'Reject',
    'melanose': 'Reject',
}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def find_class(folder_name):
    name_lower = folder_name.lower().strip()
    for key, grade in CLASS_MAP.items():
        if name_lower == key or name_lower.startswith(key):
            return grade
    return None

def walk_images(source_root):
    for dirpath, _, filenames in os.walk(source_root):
        grade = find_class(Path(dirpath).name)
        if grade is None:
            continue
        for fname in filenames:
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                yield Path(dirpath) / fname, grade

def prepare(source_root, out_dir, copy=True):
    out_path = Path(out_dir)
    for grade in ['Better', 'Good', 'Reject']:
        (out_path / grade).mkdir(parents=True, exist_ok=True)
    counts = {'Better': 0, 'Good': 0, 'Reject': 0}
    skipped = 0
    print('Scanning: ' + str(source_root))
    for img_path, grade in walk_images(source_root):
        dest_name = img_path.stem + '_' + img_path.parent.name + img_path.suffix
        dest = out_path / grade / dest_name
        idx = 1
        while dest.exists():
            dest = out_path / grade / (img_path.stem + '_' + img_path.parent.name + '_' + str(idx) + img_path.suffix)
            idx += 1
        try:
            if copy:
                shutil.copy2(img_path, dest)
            else:
                shutil.move(str(img_path), dest)
            counts[grade] += 1
        except Exception as e:
            print('  WARN: ' + str(img_path) + ': ' + str(e))
            skipped += 1
    print('--- Summary ---')
    for g, n in counts.items():
        print('  ' + g + ': ' + str(n))
    print('  Total: ' + str(sum(counts.values())))
    if skipped:
        print('  Skipped: ' + str(skipped))
    if all(v == 0 for v in counts.values()):
        print('ERROR: No images mapped.')
        sys.exit(1)
    return counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', default=None)
    parser.add_argument('--src', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--move', action='store_true')
    args = parser.parse_args()
    workspace = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    out_dir = args.out or os.path.join(workspace, 'dataset-orange', 'dataset-orange')
    out_dir = os.path.abspath(out_dir)
    if args.zip is None and args.src is None:
        print('ERROR: Provide --zip path or --src folder')
        print('Dataset: https://data.mendeley.com/datasets/3f83gxmv57/2')
        sys.exit(1)
    if args.zip:
        extract_root = os.path.join(os.path.dirname(os.path.abspath(args.zip)), 'citrus_extracted')
        print('Extracting to ' + extract_root)
        os.makedirs(extract_root, exist_ok=True)
        with zipfile.ZipFile(args.zip, 'r') as zf:
            zf.extractall(extract_root)
        source_root = extract_root
    else:
        source_root = args.src
    prepare(source_root, out_dir, copy=not args.move)
    print('Output: ' + out_dir)
    print('Next: .\\venv\\Scripts\\python.exe training\\train.py --fruit orange --epochs 25 --batch-size 16')

if __name__ == '__main__':
    main()
