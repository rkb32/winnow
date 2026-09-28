"""What a photo's own file says about where it came from. Read from the file, never inferred from
pixels: a photo that declares it was generated is flagged, and one that declares nothing proves
nothing (social media strips this metadata), so a clean result is not a guarantee.
"""
import re

MAX_READ = 16 * 1024 * 1024

# Lowercase byte strings, matched anywhere in the file. Metadata sits uncompressed in JPEG EXIF/XMP
# segments and PNG text chunks; these strings are long enough that compressed pixel data does not
# contain them by chance (5 MB of random bytes holds a given 10-byte string about once in 10^17 files).
AI_MARKERS = (
    # IPTC's "digital source type" (also written into C2PA content credentials by generators):
    # trainedAlgorithmicMedia and compositeWithTrainedAlgorithmicMedia.
    (b"rainedalgorithmicmedia", "marked AI-generated in its metadata (IPTC digital source type)"),
    (b"midjourney", "Midjourney"),
    (b"stable diffusion", "Stable Diffusion"),
    (b"stablediffusion", "Stable Diffusion"),
    (b"dall-e", "DALL-E"),
    (b"dall\xc2\xb7e", "DALL-E"),
    (b"adobe firefly", "Adobe Firefly"),
    (b"novelai", "NovelAI"),
    (b"comfyui", "ComfyUI"),
    (b"invokeai", "InvokeAI"),
)
# Automatic1111 and its forks write "Steps: 20, Sampler: Euler a, ..." into PNG text or JPEG EXIF
# (UTF-16 there, so the zero bytes are dropped before matching).
GENERATION_SETTINGS = re.compile(rb"steps: \d+, sampler: ")
# ComfyUI writes its whole node graph into a PNG text chunk.
COMFY_GRAPH = (b'"class_type"', b'"inputs"')

STOCK_MARKERS = (
    (b"getty images", "Getty Images"),
    (b"gettyimages", "Getty Images"),
    (b"shutterstock", "Shutterstock"),
    (b"istockphoto", "iStock"),
    (b"istock by getty", "iStock"),
    (b"adobe stock", "Adobe Stock"),
    (b"stock.adobe.com", "Adobe Stock"),
    (b"alamy", "Alamy"),
    (b"depositphotos", "Depositphotos"),
    (b"dreamstime", "Dreamstime"),
    (b"123rf", "123RF"),
    (b"pond5", "Pond5"),
    (b"stocksy", "Stocksy"),
)


def _first(markers, data):
    for needle, label in markers:
        if needle in data:
            return label
    return None


def read_markers(path):
    """(AI-generation marker or None, stock-agency marker or None) declared in the file."""
    with open(path, "rb") as f:
        data = f.read(MAX_READ).lower()
    flat = data.replace(b"\x00", b"")
    ai = _first(AI_MARKERS, data)
    if ai is None and GENERATION_SETTINGS.search(flat):
        ai = "Stable Diffusion settings (Steps, Sampler)"
    if ai is None and all(part in data for part in COMFY_GRAPH):
        ai = "ComfyUI workflow"
    return ai, _first(STOCK_MARKERS, data)


def find_provenance_flags(paths):
    """([(path, marker)] declared AI-generated, [(path, agency)] credited to a stock agency)."""
    ai, stock = [], []
    for path in paths:
        try:
            ai_marker, stock_marker = read_markers(path)
        except OSError:
            continue
        if ai_marker:
            ai.append((path, ai_marker))
        if stock_marker:
            stock.append((path, stock_marker))
    return ai, stock
