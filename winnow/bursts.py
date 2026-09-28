"""Photos taken seconds apart on one camera: the same moment, sampled more than once.

Pixels can't see this when the subject moved between frames, but the capture time can. A group
that spans the train and test sets is a leak that no image comparison would find.
"""
import datetime
from itertools import combinations

from PIL import Image

# A judgement call, not a measured threshold: no public photo set with capture times and
# same-moment labels was available to calibrate it.
BURST_GAP_SECONDS = 3.0
MAX_GROUPS = 20

_EXIF_IFD = 0x8769
_MAKE, _MODEL = 271, 272
_TAKEN, _TAKEN_FRACTION, _BODY_SERIAL = 36867, 37521, 42033


def _text(value):
    return str(value).strip("\x00 ") if value is not None else ""


def capture_info(path):
    """(camera, capture time in seconds), or None when the file doesn't record both."""
    try:
        exif = Image.open(path).getexif()
        ifd = exif.get_ifd(_EXIF_IFD)
        taken = datetime.datetime.strptime(_text(ifd.get(_TAKEN)), "%Y:%m:%d %H:%M:%S")
    except Exception:
        # Corrupt EXIF can raise almost anything; one bad file must not hide every other burst.
        return None
    camera = " ".join(t for t in (_text(exif.get(_MAKE)), _text(exif.get(_MODEL)), _text(ifd.get(_BODY_SERIAL))) if t)
    if not camera:
        return None
    fraction = _text(ifd.get(_TAKEN_FRACTION))
    seconds = taken.replace(tzinfo=datetime.timezone.utc).timestamp()
    return camera, seconds + (float(f"0.{fraction}") if fraction.isdigit() else 0.0)


def find_bursts(paths, test_paths=(), found_pairs=()):
    """Groups of two or more photos from one camera, each within BURST_GAP_SECONDS of the next.

    found_pairs: file pairs an image comparison already reported. A group they fully account for
    is left out, so the same moment isn't listed twice.
    """
    shots = []
    for path in [*paths, *test_paths]:
        info = capture_info(path)
        if info:
            shots.append((info[0], info[1], path))
    shots.sort()

    runs, run = [], []
    for shot in shots:
        if run and shot[0] == run[-1][0] and shot[1] - run[-1][1] <= BURST_GAP_SECONDS:
            run.append(shot)
        else:
            runs.append(run)
            run = [shot]
    runs.append(run)

    known = {frozenset(pair[:2]) for pair in found_pairs}
    test = set(test_paths)
    groups = []
    for run in runs:
        files = [s[2] for s in run]
        if len(files) < 2 or all(frozenset(pair) in known for pair in combinations(files, 2)):
            continue
        groups.append({"camera": run[0][0], "files": files,
                       "test_files": [f for f in files if f in test],
                       "span": round(run[-1][1] - run[0][1], 1)})
    return sorted(groups, key=lambda g: -len(g["files"]))[:MAX_GROUPS]
