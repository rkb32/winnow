import json
import os

from conftest import jpeg_with_orientation, scene
from labels import check_labels

# scene() photos are 480 wide and 360 high.


def photo(save, name="a.png", seed=1):
    return save(name, scene(seed), "photos")


def label(tmp_path, name, text):
    folder = tmp_path / "labels"
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text(text if isinstance(text, str) else json.dumps(text))
    return str(path)


def kinds(problems):
    return sorted((os.path.basename(subject), kind) for subject, kind, _ in problems)


def voc(name="a.png", width=480, height=360, boxes=("10 10 100 100",)):
    objects = "".join(
        f"<object><name>cat</name><bndbox><xmin>{b.split()[0]}</xmin><ymin>{b.split()[1]}</ymin>"
        f"<xmax>{b.split()[2]}</xmax><ymax>{b.split()[3]}</ymax></bndbox></object>" for b in boxes)
    return (f"<annotation><filename>{name}</filename><size><width>{width}</width><height>{height}</height>"
            f"<depth>3</depth></size>{objects}</annotation>")


def test_clean_yolo_labels_have_no_problems(save, tmp_path):
    a = photo(save)
    assert check_labels([a], [label(tmp_path, "a.txt", "0 0.5 0.5 0.4 0.4\n1 0.2 0.3 0.1 0.1\n")]) == []


def test_yolo_box_leaving_the_photo_is_reported(save, tmp_path):
    a = photo(save)
    problems = check_labels([a], [label(tmp_path, "a.txt", "0 0.5 0.5 0.4 0.4\n0 0.95 0.5 0.4 0.4\n")])
    assert kinds(problems) == [("a.png", "outside")] and "1 of 2" in problems[0][2]


def test_boxes_with_no_area_and_repeated_boxes_are_reported(save, tmp_path):
    a = photo(save)
    text = "0 0.5 0.5 0 0.3\n1 0.5 0.5 0.3 0.3\n1 0.5 0.5 0.3 0.3\n"
    assert kinds(check_labels([a], [label(tmp_path, "a.txt", text)])) == [("a.png", "empty"), ("a.png", "repeated")]


def test_the_same_box_under_two_classes_is_not_a_repeat(save, tmp_path):
    a = photo(save)
    assert check_labels([a], [label(tmp_path, "a.txt", "1 0.5 0.5 0.3 0.3\n2 0.5 0.5 0.3 0.3\n")]) == []


def test_malformed_yolo_lines_are_counted_with_the_first_line_number(save, tmp_path):
    a = photo(save)
    problems = check_labels([a], [label(tmp_path, "a.txt", "0 0.5 0.5 0.4 0.4\ncat 1 2\n0 0.5\n")])
    assert kinds(problems) == [("a.png", "malformed")] and "2 lines" in problems[0][2] and "line 2" in problems[0][2]


def test_yolo_polygon_lines_are_checked_by_their_extent(save, tmp_path):
    a = photo(save)
    assert kinds(check_labels([a], [label(tmp_path, "a.txt", "0 0.1 0.1 0.9 0.1 0.5 1.4\n")])) == [("a.png", "outside")]


def test_label_file_with_no_photo_and_photo_with_no_label(save, tmp_path):
    a, b = photo(save, "a.png", 1), photo(save, "b.png", 2)
    problems = check_labels([a, b], [label(tmp_path, "a.txt", "0 0.5 0.5 0.4 0.4\n"),
                                     label(tmp_path, "ghost.txt", "0 0.5 0.5 0.4 0.4\n")])
    assert kinds(problems) == [("b.png", "missing"), ("ghost.txt", "orphan")]


def test_class_list_files_are_not_treated_as_labels(save, tmp_path):
    a = photo(save)
    assert check_labels([a], [label(tmp_path, "a.txt", "0 0.5 0.5 0.4 0.4\n"), label(tmp_path, "classes.txt", "cat\ndog\n")]) == []


def test_voc_size_and_boxes_are_checked_against_the_photo(save, tmp_path):
    a = photo(save)
    assert check_labels([a], [label(tmp_path, "a.xml", voc())]) == []
    outside = check_labels([a], [label(tmp_path, "a.xml", voc(boxes=("10 10 100 100", "400 300 520 380")))])
    assert kinds(outside) == [("a.png", "outside")]


def test_voc_width_and_height_swapped_is_reported_as_a_size_problem(save, tmp_path):
    a = photo(save)
    [(subject, kind, detail)] = check_labels([a], [label(tmp_path, "a.xml", voc(width=360, height=480, boxes=("10 10 100 100",)))])
    assert kind == "size" and "swapped" in detail


def test_labels_made_on_the_rotated_photo_are_reported_as_the_exif_conflict(tmp_path):
    rotated = jpeg_with_orientation(str(tmp_path / "rot.jpg"), 6)   # stored 480x360, displayed 360x480
    problems = check_labels([rotated], [label(tmp_path, "rot.xml", voc("rot.jpg", width=360, height=480))])
    assert kinds(problems) == [("rot.jpg", "size")] and "EXIF rotation tag (6)" in problems[0][2]


def test_labels_for_a_resized_photo_are_reported(save, tmp_path):
    a = photo(save)
    [(_, kind, detail)] = check_labels([a], [label(tmp_path, "a.xml", voc(width=960, height=720, boxes=("10 10 100 100",)))])
    assert kind == "size" and "resized" in detail


def test_voc_with_a_doctype_is_refused_not_parsed(save, tmp_path):
    a = photo(save)
    evil = '<!DOCTYPE a [<!ENTITY x "boom">]>' + voc()
    [(subject, kind, _)] = check_labels([a], [label(tmp_path, "a.xml", evil)])
    assert kind == "unreadable" and subject.endswith("a.xml")


def coco(images, annotations):
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "cat"}]}


def test_coco_boxes_and_sizes_are_checked_per_photo(save, tmp_path):
    a, b = photo(save, "a.png", 1), photo(save, "b.png", 2)
    data = coco([{"id": 1, "file_name": "train/a.png", "width": 480, "height": 360},
                 {"id": 2, "file_name": "b.png", "width": 360, "height": 480},
                 {"id": 3, "file_name": "not_uploaded.png", "width": 10, "height": 10}],
                [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100]},
                 {"image_id": 1, "category_id": 1, "bbox": [400, 300, 120, 80]},
                 {"image_id": 2, "category_id": 1, "bbox": [10, 10, 50, 50]}])
    problems = check_labels([a, b], [label(tmp_path, "instances.json", data)])
    assert kinds(problems) == [("a.png", "outside"), ("b.png", "size"), ("instances.json", "orphan")]


def test_unparseable_label_files_are_reported_not_fatal(save, tmp_path):
    a = photo(save)
    problems = check_labels([a], [label(tmp_path, "broken.json", "{not json"), label(tmp_path, "a.txt", "0 0.5 0.5 0.4 0.4\n")])
    assert kinds(problems) == [("broken.json", "unreadable")]


def test_a_voc_size_of_zero_is_ignored_not_a_crash(save, tmp_path):
    a = photo(save)
    assert check_labels([a], [label(tmp_path, "a.xml", voc(width=0, height=0))]) == []


def test_junk_values_in_a_label_file_never_abort_the_check(save, tmp_path):
    a, b = photo(save, "a.png", 1), photo(save, "b.png", 2)
    data = coco([{"id": 1, "file_name": "a.png", "width": 480, "height": 360}],
                [{"image_id": 1, "category_id": 1, "bbox": ["x", None, 3, 4]}])
    problems = check_labels([a, b], [label(tmp_path, "bad.json", data), label(tmp_path, "b.txt", "0 0.5 0.5 0.4 0.4\n")])
    assert kinds(problems) == [("a.png", "missing"), ("bad.json", "unreadable")]
