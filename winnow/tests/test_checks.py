import os

import numpy as np

from bursts import find_bursts
from conftest import GETTY, IPTC_AI, jpeg_with_capture, photo_with_metadata
from provenance import find_provenance_flags, read_markers
from shift import check_shift

# --- provenance ---

def test_iptc_ai_source_type_in_xmp_is_reported(tmp_path):
    path = photo_with_metadata(str(tmp_path / "gen.jpg"), xmp=IPTC_AI)
    assert read_markers(path)[0].startswith("marked AI-generated")


def test_stable_diffusion_png_settings_are_reported(tmp_path):
    text = ("parameters", "a cat\nNegative prompt: blur\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1")
    assert "Stable Diffusion" in read_markers(photo_with_metadata(str(tmp_path / "sd.png"), text=text))[0]


def test_comfyui_graph_in_png_is_reported(tmp_path):
    text = ("prompt", '{"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}')
    assert read_markers(photo_with_metadata(str(tmp_path / "comfy.png"), text=text))[0] == "ComfyUI workflow"


def test_generation_settings_stored_as_utf16_are_reported(tmp_path):
    path = tmp_path / "exif_comment.jpg"
    path.write_bytes(b"\xff\xd8" + "UNICODE Steps: 20, Sampler: Euler a".encode("utf-16-be"))
    assert read_markers(str(path))[0].startswith("Stable Diffusion")


def test_stock_agency_credit_is_reported_separately_from_ai(tmp_path):
    path = photo_with_metadata(str(tmp_path / "stock.jpg"), xmp=GETTY)
    assert read_markers(path) == (None, "Getty Images")


def test_photo_without_markers_is_clean_and_batch_lists_only_flagged(tmp_path):
    clean = photo_with_metadata(str(tmp_path / "clean.jpg"), seed=1)
    ai = photo_with_metadata(str(tmp_path / "gen.jpg"), seed=2, xmp=IPTC_AI)
    assert read_markers(clean) == (None, None)
    flagged_ai, flagged_stock = find_provenance_flags([clean, ai])
    assert [os.path.basename(p) for p, _ in flagged_ai] == ["gen.jpg"] and flagged_stock == []


# --- bursts ---

def shots(tmp_path, *specs, folder="train"):
    (tmp_path / folder).mkdir(exist_ok=True)
    return [jpeg_with_capture(str(tmp_path / folder / name), camera, taken, fraction)
            for name, camera, taken, fraction in specs]


def test_photos_seconds_apart_on_one_camera_form_a_burst(tmp_path):
    files = shots(tmp_path,
                  ("a.jpg", "Canon R5", "2024:05:01 10:00:00", ""),
                  ("b.jpg", "Canon R5", "2024:05:01 10:00:01", ""),
                  ("c.jpg", "Canon R5", "2024:05:01 10:00:03", ""),
                  ("later.jpg", "Canon R5", "2024:05:01 10:05:00", ""))
    [group] = find_bursts(files)
    assert [os.path.basename(f) for f in group["files"]] == ["a.jpg", "b.jpg", "c.jpg"]
    assert group["span"] == 3.0 and group["camera"] == "Canon R5" and group["test_files"] == []


def test_fractions_of_a_second_decide_the_order_and_the_gap(tmp_path):
    files = shots(tmp_path,
                  ("a.jpg", "Sony A7", "2024:05:01 10:00:00", "95"),
                  ("b.jpg", "Sony A7", "2024:05:01 10:00:01", "05"))
    assert find_bursts(files)[0]["span"] == 0.1


def test_different_cameras_at_the_same_moment_are_not_a_burst(tmp_path):
    files = shots(tmp_path,
                  ("a.jpg", "Canon R5", "2024:05:01 10:00:00", ""),
                  ("b.jpg", "Sony A7", "2024:05:01 10:00:00", ""))
    assert find_bursts(files) == []


def test_burst_across_train_and_test_names_the_test_photo(tmp_path):
    [train] = shots(tmp_path, ("a.jpg", "Canon R5", "2024:05:01 10:00:00", ""))
    [test] = shots(tmp_path, ("b.jpg", "Canon R5", "2024:05:01 10:00:02", ""), folder="test")
    [group] = find_bursts([train], [test])
    assert group["test_files"] == [test]


def test_burst_already_explained_by_an_image_match_is_not_listed_again(tmp_path):
    a, b = shots(tmp_path,
                 ("a.jpg", "Canon R5", "2024:05:01 10:00:00", ""),
                 ("b.jpg", "Canon R5", "2024:05:01 10:00:01", ""))
    assert find_bursts([a, b], found_pairs=[(a, b, 0)]) == []


def test_photos_without_capture_times_are_ignored(tmp_path, save):
    from conftest import scene
    assert find_bursts([save("plain.png", scene(1)), save("plain2.png", scene(2))]) == []


# --- train/test shift ---

def unit(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def embeddings(rng, center, count, prefix, spread=0.5):
    vectors = unit(center + spread * rng.standard_normal((count, 32)))
    return {f"{prefix}{i}": v for i, v in enumerate(vectors)}


def test_test_photos_from_elsewhere_are_flagged_with_the_strangest_first():
    rng = np.random.default_rng(0)
    home, away = rng.standard_normal(32), rng.standard_normal(32)
    result = check_shift(embeddings(rng, home, 15, "train"), embeddings(rng, away, 8, "test"))
    assert result["checked"] and result["flagged"] and result["p_value"] < 0.01
    assert len(result["outliers"]) == 3 and all(p.startswith("test") for p, _ in result["outliers"])
    assert result["outliers"] == sorted(result["outliers"], key=lambda o: o[1])


def test_test_photos_from_the_same_source_are_not_flagged():
    rng = np.random.default_rng(1)
    center = rng.standard_normal(32)
    result = check_shift(embeddings(rng, center, 15, "train"), embeddings(rng, center, 8, "test"))
    assert result["checked"] and not result["flagged"] and result["outliers"] == []


def test_too_few_photos_are_not_compared():
    rng = np.random.default_rng(2)
    center = rng.standard_normal(32)
    result = check_shift(embeddings(rng, center, 15, "train"), embeddings(rng, center, 4, "test"))
    assert result == {"checked": False, "train_count": 15, "test_count": 4}
    assert check_shift(embeddings(rng, center, 15, "train"), embeddings(rng, center, 5, "test"))["checked"]


def test_shift_check_gives_the_same_answer_every_time():
    rng = np.random.default_rng(3)
    train, test = embeddings(rng, rng.standard_normal(32), 12, "train"), embeddings(rng, rng.standard_normal(32), 6, "test")
    assert check_shift(train, test) == check_shift(train, test)
