"""Build HarvestLenz Core Four demo videos from real grading results.

Renders 1920x1080 @24fps frames with PIL, encodes H.264 with the ffmpeg
binary bundled by imageio-ffmpeg, and concatenates a full demo reel.
"""
import json
import os
import subprocess
import tempfile
import shutil
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = r"C:\Users\madha\OneDrive\Desktop\HarvestLenz"
DEMO_IMAGES = os.path.join(ROOT, "demo_images")
OUT_DIR = os.path.join(ROOT, "demo_videos")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

W, H, FPS = 1920, 1080, 24

BG       = (12, 18, 14)
PANEL    = (18, 27, 21)
GREEN    = (58, 205, 106)
GREEN_D  = (47, 107, 63)
ORANGE   = (255, 184, 77)
RED      = (255, 77, 77)
WHITE    = (240, 244, 238)
GRAY     = (150, 160, 150)
FAINT    = (110, 122, 112)

F_REG = r"C:\Windows\Fonts\segoeui.ttf"
F_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
F_LIGHT = r"C:\Windows\Fonts\segoeuil.ttf"
F_SEMI = r"C:\Windows\Fonts\seguisb.ttf"

LATIN = {
    "mango": "Mangifera indica",
    "pineapple": "Ananas comosus",
    "grapes": "Vitis vinifera",
    "pomegranate": "Punica granatum",
}

DEFECT_LABELS = {
    "severe_rot_or_bruising": "Severe rot / bruising",
    "shape_deformation": "Shape deformation",
    "minor_spots": "Minor surface spots",
    "minor_bruising": "Minor bruising",
    "shrivel": "Shrivel / drying",
}

GRADE_COLOR = {"Good": GREEN, "Better": ORANGE, "Reject": RED}


def load_results():
    with open(os.path.join(ROOT, "demo_results.json"), encoding="utf-8") as f:
        return json.load(f)


def font(path, size):
    return ImageFont.truetype(path, size)


def crop_cover(img, box_w, box_h):
    iw, ih = img.size
    scale = max(box_w / iw, box_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    x = (nw - box_w) // 2
    y = (nh - box_h) // 2
    return img.crop((x, y, x + box_w, y + box_h))


def draw_image(photo, canvas, box, pad=10, zoom=1.0):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    frame_w, frame_h = bw - 2 * pad, bh - 2 * pad
    fw, fh = int(frame_w * zoom), int(frame_h * zoom)
    img = crop_cover(photo, fw, fh)
    ox = (x0 + pad) + (frame_w - fw) // 2
    oy = (y0 + pad) + (frame_h - fh) // 2
    mask = Image.new("L", (fw, fh), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, fw - 1, fh - 1), 18, fill=255)
    canvas.paste(img, (ox, oy), mask)


def rounded(draw, box, r, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def center_text(draw, cx, cy, text, fnt, fill):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((cx - w / 2 - bbox[0], cy - h / 2 - bbox[1]), text, font=fnt, fill=fill)


def row_height(draw, text, fnt):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[3] - bbox[1]


def result_rows(data):
    defects = data.get("defects") or []
    labels = ", ".join(DEFECT_LABELS.get(d, d.replace("_", " ")) for d in defects) if defects else "None detected"
    return [
        ("Confidence", f"{data['conf'] * 100:.1f}%"),
        ("Defect score", f"{data['defect']:.2f}"),
        ("Defects", labels),
        ("Shelf life", f"{data['shelf_life_days']} day{'s' if data['shelf_life_days'] != 1 else ''}"),
        ("Market call", f"{data['market']}  \u00b7  {data['price']}"),
    ]


def draw_scene(base, data, t, duration, phase_meta=None):
    """Draw one frame. t in seconds. phase_meta (dict or None) draws a tag."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # subtle top/bottom bars
    d.rectangle((0, 0, W, 6), fill=GREEN_D)
    d.rectangle((0, H - 6, W, H), fill=GREEN_D)

    # top strip
    d.text((64, 26), "HARVESTLENZ", font=font(F_BOLD, 30), fill=GREEN)
    right_txt = "CORE FOUR  \u00b7  LIVE GRADING DEMO"
    rb = d.textbbox((0, 0), right_txt, font=font(F_SEMI, 22))
    d.text((W - 64 - (rb[2] - rb[0]), 34), right_txt, font=font(F_SEMI, 22), fill=FAINT)

    # left image panel
    zoom = 1.0 + 0.04 * min(1.0, t / 0.9)
    img_box = (80, 120, 1180, 1000)
    rounded(d, img_box, 24, fill=PANEL)
    draw_image(data["_img"], img, img_box, pad=12, zoom=zoom)
    rounded(d, img_box, 24, outline=GREEN_D, width=3)

    # right panel
    rx = 1230
    ry = 150
    fruit = data["fruit"]
    d.text((rx, ry), data["name"], font=font(F_BOLD, 48), fill=WHITE)
    d.text((rx + 4, ry + 66), LATIN.get(fruit, ""), font=font(F_LIGHT, 24), fill=FAINT)
    d.rectangle((rx, ry + 108, rx + 300, ry + 112), fill=GREEN)

    top_y = ry + 150
    if t < 1.0:
        dots = "." * (1 + int(t * 4) % 3)
        d.text((rx, top_y), f"ANALYZING{dots}", font=font(F_SEMI, 34), fill=GREEN)
        return img

    # result rows slide in sequentially
    rows = result_rows(data)
    delays = [1.0 + 0.14 * i for i in range(len(rows))]
    phase = data.get("phase", "")
    y = top_y + 10
    for i, (label, value) in enumerate(rows):
        tt = t - delays[i]
        if tt < 0:
            y += 86
            continue
        alpha = min(1.0, tt / 0.25)
        off = int((1 - alpha) * 30)
        if alpha <= 0:
            y += 86
            continue
        d.text((rx, y - off + 4), label.upper(), font=font(F_SEMI, 20), fill=FAINT)
        d.text((rx, y - off + 30), value, font=font(F_SEMI, 30), fill=WHITE)
        if label == "Defect score":
            bx, by = rx, y - off + 74
            d.rounded_rectangle((bx, by, bx + 320, by + 10), 5, fill=(52, 62, 55))
            frac = min(1.0, data["defect"])
            col = GREEN if frac < 0.3 else (ORANGE if frac < 0.6 else RED)
            d.rounded_rectangle((bx, by, bx + int(320 * frac), by + 10), 5, fill=col)
        y += 86

    # grade badge bottom of panel
    gy = 690
    gc = GRADE_COLOR.get(data["grade"], GREEN)
    pull = 0 if t >= 1.8 else (1.8 - t) * 200
    pill = (rx, int(gy + pull), rx + 300, int(gy + pull + 64))
    rounded(d, pill, 32, fill=gc)
    center_text(d, (pill[0] + pill[2]) / 2, (pill[1] + pill[3]) / 2, f"GRADE  {data['grade'].upper()}", font(F_BOLD, 26), (10, 14, 10))

    # frame counter / bottom timeline
    ty = 1032
    frac = min(1.0, t / duration)
    d.rounded_rectangle((80, ty, W - 80, ty + 10), 5, fill=(30, 38, 32))
    d.rounded_rectangle((80, ty, 80 + int((W - 160) * frac), ty + 10), 5, fill=GREEN)
    d.text((W - 300, 1002), f"{int(t):02d} / {int(duration):02d}s", font=font(F_SEMI, 20), fill=FAINT)

    if phase_meta:
        d.rounded_rectangle((rx, 872, rx + 620, 936), 16, fill=(26, 34, 28), outline=GREEN_D, width=2)
        d.text((rx + 24, 892), phase_meta, font=font(F_SEMI, 24), fill=GREEN)

    return img


def render_clip(frames_dir, out_mp4, fps=FPS):
    try:
        subprocess.run(
            [FFMPEG, "-y", "-framerate", str(fps), "-i",
             os.path.join(frames_dir, "frame_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
             "-preset", "medium", "-movflags", "+faststart", out_mp4],
            check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("FFMPEG ERROR:", e.stderr.decode(errors="replace"))
        raise
    print("  ->", os.path.basename(out_mp4), f"{os.path.getsize(out_mp4)/1024/1024:.1f} MB")


def make_fruit_clip(data, seconds, out_name, phases=None):
    """phases: optional list of (seconds, data, tag) for multi-scene clips."""
    tmp = tempfile.mkdtemp(prefix="hl_")
    n = int(seconds * FPS)
    phases = phases or [(seconds, data, None)]
    timeline = []
    for (dur, dd, tag) in phases:
        timeline.extend([(dd, tag)] * int(dur * FPS))
    for i in range(n):
        t = i / FPS
        if i < len(timeline):
            dd, tag = timeline[i]
        else:
            dd, tag = timeline[-1]
        frame = draw_scene(Image.new("RGB", (1, 1), BG), dd, t, seconds, tag)
        frame.save(os.path.join(tmp, f"frame_{i + 1:04d}.png"))
    out = os.path.join(OUT_DIR, out_name)
    render_clip(tmp, out)
    shutil.rmtree(tmp, ignore_errors=True)


def make_intro(duration=2.0, out="intro.mp4"):
    tmp = tempfile.mkdtemp(prefix="hl_intro_")
    n = int(duration * FPS)
    for i in range(n):
        t = i / FPS
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, W, 6), fill=GREEN_D)
        d.rectangle((0, H - 6, W, H), fill=GREEN_D)
        a = min(1.0, t / 0.5)
        center_text(d, W / 2, 430, "HARVESTLENZ", font(F_BOLD, 110), tuple(int(c * a + (1 - a) * BG[c]) for c in range(3)))
        center_text(d, W / 2, 560, "Fruit Quality Grading  \u00b7  Core Four Demo", font(F_SEMI, 40), GRAY)
        d.rectangle((W // 2 - 220, 640, W // 2 + 220, 644), fill=GREEN)
        d.rectangle((W // 2 - 160, 656, W // 2 + 160, 660), fill=ORANGE)
        center_text(d, W / 2, 720, "One photo in \u2014 grade, shelf life, market call out.", font(F_LIGHT, 28), FAINT)
        img.save(os.path.join(tmp, f"frame_{i + 1:04d}.png"))
    out_p = os.path.join(OUT_DIR, out)
    render_clip(tmp, out_p)
    shutil.rmtree(tmp, ignore_errors=True)


def make_outro(duration=2.0, out="outro.mp4"):
    tmp = tempfile.mkdtemp(prefix="hl_outro_")
    n = int(duration * FPS)
    for i in range(n):
        t = i / FPS
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, W, 6), fill=GREEN_D)
        d.rectangle((0, H - 6, W, H), fill=GREEN_D)
        a = min(1.0, t / 0.4)
        center_text(d, W / 2, 470, "Thank you", font(F_BOLD, 90), tuple(int(c * a + (1 - a) * BG[c]) for c in range(3)))
        center_text(d, W / 2, 600, "HarvestLenz \u00b7 The Core Four", font(F_SEMI, 38), GRAY)
        d.rectangle((W // 2 - 160, 670, W // 2 + 160, 674), fill=GREEN)
        center_text(d, W / 2, 730, "www.harvestlenz.app (demo)", font(F_LIGHT, 26), FAINT)
        img.save(os.path.join(tmp, f"frame_{i + 1:04d}.png"))
    out_p = os.path.join(OUT_DIR, out)
    render_clip(tmp, out_p)
    shutil.rmtree(tmp, ignore_errors=True)


def concat(clips, out_name):
    tmp = os.path.join(tempfile.mkdtemp(prefix="hl_cat_"), "list.txt")
    with open(tmp, "w") as f:
        for c in clips:
            f.write(f"file '{os.path.join(OUT_DIR, c)}'\n")
    out = os.path.join(OUT_DIR, out_name)
    try:
        subprocess.run(
            [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", tmp,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
             "-preset", "medium", "-movflags", "+faststart", out],
            check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("FFMPEG CONCAT ERROR:", e.stderr.decode(errors="replace"))
        raise
    print("  ->", out_name, f"{os.path.getsize(out)/1024/1024:.1f} MB")
    shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = load_results()
    images = {}
    for root, _dirs, files in os.walk(DEMO_IMAGES):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                images.setdefault(fn, os.path.join(root, fn))
    for r in res:
        r["_img"] = Image.open(images[r["image"]])

    names = {
        "mango3.jpg": "Mango", "pine3.jpg": "Pineapple", "grape2.jpg": "Grapes",
        "pom_plain.jpg": "Pomegranate", "pom_rot_spotty.jpg": "Pomegranate",
    }
    for r in res:
        r["name"] = names.get(r["image"], r["fruit"].capitalize())

    # per-fruit clips (5s each)
    fruit_files = {}
    for r in res:
        if r["image"] in ("pom_plain.jpg", "pom_rot_spotty.jpg"):
            continue
        key = r["fruit"]
        make_fruit_clip(r, 5.0, f"{key}_demo.mp4")
        fruit_files[key] = f"{key}_demo.mp4"

    # pomegranate clip: clean -> rotten
    pom_plain = next(r for r in res if r["image"] == "pom_plain.jpg")
    pom_rot = next(r for r in res if r["image"] == "pom_rot_spotty.jpg")
    make_fruit_clip(pom_plain, 8.0, "pomegranate_demo.mp4",
                    phases=[(3.5, pom_plain, None),
                            (0.5, pom_rot, None),
                            (4.0, pom_rot, "Same camera \u00b7 fresh photo \u2014 now rejected")])
    fruit_files["pomegranate"] = "pomegranate_demo.mp4"

    make_intro()
    make_outro()
    concat(["intro.mp4", fruit_files["mango"], fruit_files["pineapple"],
            fruit_files["grapes"], fruit_files["pomegranate"], "outro.mp4"],
           "harvestlenz_full_demo.mp4")
    print("DONE")


if __name__ == "__main__":
    main()
