"""Checks label files against the photos they describe: boxes that leave the photo, boxes with
no area, boxes drawn twice, files with no photo, and label sizes that contradict the photo.

The size check turns the EXIF rotation warning from "worth checking by hand" into a finding:
when the labels record the rotated size and the file stores the unrotated one, every box is
wrong for any tool that ignores the rotation tag.

Reads YOLO (.txt, one file per photo), Pascal VOC (.xml, one per photo) and COCO (.json, one for
all). Every check reports (subject file, kind, detail); nothing here changes or removes a file.
"""
import json
import os
import xml.etree.ElementTree as ET

from PIL import Image

from exif_conflict import swaps_dimensions

MAX_LABEL_BYTES = 2 * 1024 * 1024
MAX_BOXES_PER_PHOTO = 500
MAX_MISSING_SHOWN = 10
OUTSIDE_TOLERANCE = 0.01     # label tools round; a box 1% past the edge is noise
DUPLICATE_IOU = 0.95
# .txt files that sit beside YOLO labels without being labels.
NOT_LABELS = {"classes.txt", "labels.txt", "obj.names", "readme.txt"}


def _name(path):
    return os.path.basename(path).split("__")[-1]


def _stem(name):
    return os.path.splitext(os.path.basename(name))[0].lower()


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _size_problem(declared, actual, orientation):
    if declared == actual:
        return None
    (dw, dh), (aw, ah) = declared, actual
    if declared == (ah, aw):
        if swaps_dimensions(orientation):
            return (f"the labels are for the rotated photo ({dw}x{dh}) but the file stores it as {aw}x{ah} with "
                    f"an EXIF rotation tag ({orientation}); any tool that ignores the tag puts every box in the wrong place")
        return f"the labels say {dw}x{dh} but the photo is {aw}x{ah}: width and height look swapped"
    return f"the labels say {dw}x{dh} but the photo is {aw}x{ah}: was it resized after labeling?"


def _box_problems(boxes, frame):
    """boxes: (class, x1, y1, x2, y2) as fractions of the photo; frame: its (width, height) in pixels."""
    fw, fh = frame
    outside = empty = repeated = 0
    kept = []
    for cls, x1, y1, x2, y2 in boxes[:MAX_BOXES_PER_PHOTO]:
        if x2 - x1 < 1e-9 or y2 - y1 < 1e-9 or (x2 - x1) * fw < 1 or (y2 - y1) * fh < 1:
            empty += 1
            continue
        if (min(x1, y1) < -OUTSIDE_TOLERANCE or max(x2, y2) > 1 + OUTSIDE_TOLERANCE):
            outside += 1
        if any(c == cls and _iou(b, (x1, y1, x2, y2)) >= DUPLICATE_IOU for c, b in kept):
            repeated += 1
        kept.append((cls, (x1, y1, x2, y2)))
    problems = []
    total = min(len(boxes), MAX_BOXES_PER_PHOTO)
    if outside:
        problems.append(("outside", f"{outside} of {total} boxes extend outside the photo"))
    if empty:
        problems.append(("empty", f"{empty} of {total} boxes have no area"))
    if repeated:
        problems.append(("repeated", f"{repeated} of {total} boxes repeat another box (same class, same place)"))
    return problems


def _parse_yolo(text):
    """(boxes as fractions of the photo, first bad line number or None, bad line count)."""
    boxes, bad, first_bad = [], 0, None
    for number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls, *values = parts
            values = [float(v) for v in values]
            cls = int(float(cls))
            if cls < 0 or any(v != v for v in values):
                raise ValueError
            if len(values) in (4, 5):
                cx, cy, w, h = values[:4]
                boxes.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            elif len(values) >= 6 and len(values) % 2 == 0:
                xs, ys = values[0::2], values[1::2]
                boxes.append((cls, min(xs), min(ys), max(xs), max(ys)))
            else:
                raise ValueError
        except ValueError:
            bad += 1
            first_bad = first_bad or number
    return boxes, first_bad, bad


def _parse_voc(data):
    if b"<!doctype" in data.lower() or b"<!entity" in data.lower():
        raise ValueError("XML with a DOCTYPE isn't read")
    root = ET.fromstring(data)
    size = root.find("size")
    declared = (int(float(size.findtext("width"))), int(float(size.findtext("height")))) if size is not None else None
    boxes = []
    for obj in root.iter("object"):
        box = obj.find("bndbox")
        if box is not None:
            boxes.append((obj.findtext("name", ""), *(float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return root.findtext("filename"), declared, boxes


def check_labels(image_paths, label_paths):
    """[(subject file, kind, detail)] for the label files and the photos they should describe."""
    photos = {}
    for path in image_paths:
        photos.setdefault(_stem(_name(path)), []).append(path)
    problems, labeled = [], set()

    def report(subject, kind, detail):
        problems.append((subject, kind, detail))

    def photo_for(name):
        matches = photos.get(_stem(name), [])
        return matches[0] if len(matches) == 1 else None

    def check_photo(photo, boxes, declared=None, fractional=False):
        """boxes: (class, x1, y1, x2, y2), as fractions of the photo when `fractional`, else in
        pixels of the frame the labels were drawn on (`declared`, or the photo's own size)."""
        labeled.add(photo)
        with Image.open(photo) as image:
            actual, orientation = image.size, image.getexif().get(274, 1)
        if declared and min(declared) <= 0:
            declared = None     # some tools write 0x0 when they didn't record a size
        if declared:
            problem = _size_problem(declared, actual, orientation)
            if problem:
                report(photo, "size", problem)
        frame = declared or actual
        if not fractional:
            boxes = [(c, x1 / frame[0], y1 / frame[1], x2 / frame[0], y2 / frame[1]) for c, x1, y1, x2, y2 in boxes]
        for kind, detail in _box_problems(boxes, frame):
            report(photo, kind, detail)

    for path in label_paths:
        name = _name(path)
        extension = os.path.splitext(name)[1].lower()
        if name.lower() in NOT_LABELS:
            continue
        try:
            if os.path.getsize(path) > MAX_LABEL_BYTES:
                raise ValueError("file is too large to check")
            with open(path, "rb") as f:
                data = f.read()
            if extension == ".txt":
                boxes, first_bad, bad = _parse_yolo(data.decode("utf-8", "replace"))
                photo = photo_for(name)
                if photo is None:
                    report(path, "orphan", "no photo has this label file's name")
                    continue
                if bad:
                    report(photo, "malformed", f"{bad} lines aren't 'class x y w h' (first: line {first_bad})")
                check_photo(photo, boxes, fractional=True)
            elif extension == ".xml":
                filename, declared, boxes = _parse_voc(data)
                photo = photo_for(name) or (photo_for(filename) if filename else None)
                if photo is None:
                    report(path, "orphan", "no photo has this label file's name")
                    continue
                check_photo(photo, boxes, declared)
            elif extension == ".json":
                _check_coco(json.loads(data), photos, check_photo, report, path)
        except Exception as e:
            # Label files are visitors' input in three formats; whatever is wrong with one, the rest still get checked.
            report(path, "unreadable", f"couldn't be read as a label file ({type(e).__name__}: {e})")

    if labeled:
        missing = [p for p in image_paths if p not in labeled]
        for photo in missing[:MAX_MISSING_SHOWN]:
            report(photo, "missing", "no label for this photo (fine if it has nothing to label)")
    return problems


def _check_coco(data, photos, check_photo, report, path):
    if not isinstance(data, dict) or "images" not in data:
        raise ValueError("no 'images' list")
    boxes_by_image = {}
    for annotation in data.get("annotations", []):
        x, y, w, h = (float(v) for v in annotation["bbox"])
        boxes_by_image.setdefault(annotation["image_id"], []).append((annotation.get("category_id"), x, y, x + w, y + h))
    unmatched = 0
    for entry in data["images"]:
        matches = photos.get(_stem(entry["file_name"]), [])
        if len(matches) != 1:
            unmatched += 1
            continue
        declared = (int(entry["width"]), int(entry["height"])) if "width" in entry and "height" in entry else None
        check_photo(matches[0], boxes_by_image.get(entry["id"], []), declared)
    if unmatched:
        report(path, "orphan", f"{unmatched} of the {len(data['images'])} images it lists aren't among the photos")
