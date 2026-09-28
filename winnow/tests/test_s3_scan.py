from s3_scan import local_name, public_key, session_layout

SESSION = "1" * 32
KEPT = "saved/" + "a" * 32 + f"/{SESSION}/"


def test_plain_uploads_are_read_where_they_were_put():
    layout = session_layout(SESSION)
    assert layout["images"] == f"uploads/{SESSION}/images/" and layout["labels"] == f"uploads/{SESSION}/labels/"
    assert layout["quarantine"] == f"uploads/{SESSION}/quarantine/" and layout["alias"] is None


def test_kept_photos_are_read_from_the_guests_folder():
    layout = session_layout(SESSION, KEPT)
    assert layout["images"] == KEPT + "images/" and layout["test"] == KEPT + "test/"
    assert layout["alias"] == f"uploads/{SESSION}/"


def test_local_copies_of_kept_photos_are_named_like_plain_uploads():
    key = KEPT + "images/3/photo.png"
    assert local_name(key, KEPT + "images/", f"uploads/{SESSION}/images/") == f"uploads__{SESSION}__images__3__photo.png"
    assert "saved" not in local_name(key, KEPT + "images/", f"uploads/{SESSION}/images/")


def test_local_names_without_an_alias_keep_the_whole_key():
    assert local_name("samples/a.png", "samples/") == "samples__a.png"


def test_quarantine_keys_in_a_public_report_never_show_the_guests_folder():
    moved = KEPT + "quarantine/harbor.png"
    assert public_key(moved, KEPT + "quarantine/", f"uploads/{SESSION}/quarantine/") == f"uploads/{SESSION}/quarantine/harbor.png"
    assert "saved" not in public_key(moved, KEPT + "quarantine/", f"uploads/{SESSION}/quarantine/")


def test_plain_uploads_keep_their_real_quarantine_keys():
    key = f"uploads/{SESSION}/quarantine/harbor.png"
    assert public_key(key, f"uploads/{SESSION}/quarantine/", None) == key
