import copy

import pytest

import agent
from conftest import crop, scene

A = "/tmp/x/uploads__s__images__0__a.png"
B = "/tmp/x/uploads__s__images__1__b.png"
C = "/tmp/x/uploads__s__images__2__c.png"
RESULT = {"duplicate_pairs": [(A, B, 0)], "blurry_images": [(C, 12.5)], "risky_orientation_images": []}


def record(*decisions, pairs=()):
    answer = {"decisions": list(decisions), "pairs": list(pairs)}
    return {"toolUse": {"toolUseId": "d", "name": "record_decisions", "input": answer}}


def verdict(value, reason="", pair="pair0"):
    return {"pair": pair, "verdict": value, "reason": reason}


def zoom(n=0, pair="pair0", image="a", box=(0.0, 0.0, 0.5, 0.5)):
    return {"toolUse": {"toolUseId": f"z{n}", "name": "zoom", "input": {"pair": pair, "image": image, "box": list(box)}}}


class FakeBedrock:
    """Replies with each scripted message content in turn, recording a snapshot of every request."""

    def __init__(self, *replies, stop_reason="tool_use"):
        self.replies = list(replies)
        self.stop_reason = stop_reason
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        content = self.replies.pop(0)
        return {"stopReason": self.stop_reason, "output": {"message": {"role": "assistant", "content": content}}}


def use(monkeypatch, client):
    monkeypatch.setattr(agent.boto3, "client", lambda *args, **kwargs: client)
    return client


def test_no_findings_never_calls_bedrock(monkeypatch):
    monkeypatch.setattr(agent.boto3, "client", lambda *a, **k: pytest.fail("Bedrock was called"))
    assert agent.decide_actions({"duplicate_pairs": [], "blurry_images": [], "risky_orientation_images": []}) == []


def test_short_ids_map_back_to_paths_and_unknown_ids_are_dropped(monkeypatch):
    fake = use(monkeypatch, FakeBedrock([record(
        {"file": "img1", "action": "quarantine", "reason": "Duplicate of a.png"},
        {"file": "img9", "action": "quarantine", "reason": "An id that doesn't exist"},
    )]))
    assert agent.decide_actions(RESULT) == [{"file": B, "action": "quarantine", "reason": "Duplicate of a.png"}]

    call = fake.calls[0]
    prompt = call["messages"][0]["content"][0]["text"]
    assert "/tmp/x" not in prompt and '"img1": "b.png"' in prompt
    assert call["toolConfig"]["toolChoice"] == {"tool": {"name": "record_decisions"}}


def test_cut_off_reply_is_an_error_not_an_empty_answer(monkeypatch):
    use(monkeypatch, FakeBedrock([record()], stop_reason="max_tokens"))
    with pytest.raises(RuntimeError):
        agent.decide_actions(RESULT)


def test_findings_sent_to_claude_are_capped(monkeypatch):
    fake = use(monkeypatch, FakeBedrock([record()]))
    many = dict(RESULT, blurry_images=[(f"/tmp/x/p{i}.png", 1.0) for i in range(50)])
    agent.decide_actions(many)
    prompt = fake.calls[0]["messages"][0]["content"][0]["text"]
    assert prompt.count('"type": "blurry"') == agent.MAX_FINDINGS - 1


def test_every_finding_type_is_summarized():
    result = {
        "leaked_pairs": [(A, B, 0)], "similar_leaks": [(A, C, 0.95, 80)],
        "duplicate_pairs": [(A, B, 0)], "similar_pairs": [(B, C, 0.9, 40)],
        "blurry_images": [(C, 3.0)], "risky_orientation_images": [(A, 6, True)],
    }
    types = [f["type"] for f in agent.build_findings_summary(result)]
    assert types == ["train_test_leak", "train_test_leak_edited_copy", "duplicate", "edited_copy", "blurry", "risky_exif"]


def borderline_pair(save):
    image = scene(20)
    a, b = save("a.png", image), save("b.png", crop(image))
    result = {"duplicate_pairs": [], "similar_pairs": [(a, b, 0.9, 30)],
              "blurry_images": [], "risky_orientation_images": []}
    return a, b, result


def review_everything(monkeypatch):
    # These tests cover the review loop itself, whatever select_for_review picks.
    monkeypatch.setattr(agent, "select_for_review", lambda findings: findings)


def test_borderline_pair_is_shown_as_photos_and_can_be_zoomed(monkeypatch, save):
    review_everything(monkeypatch)
    a, b, result = borderline_pair(save)
    fake = use(monkeypatch, FakeBedrock(
        [zoom(image="b", box=(0.6, 0.6, 0.2, 0.2))],
        [record(pairs=[verdict("same", "Cropped copy of a.png")])],
    ))
    decisions = agent.decide_actions(result)

    review = {"verdict": "same", "zooms": [{"file": b, "box": [0.2, 0.2, 0.6, 0.6], "size": [480, 360]}]}
    assert decisions == [
        {"file": a, "action": "keep", "reason": "Cropped copy of a.png", "visual_review": review},
        {"file": b, "action": "quarantine", "reason": "Cropped copy of a.png", "visual_review": review},
    ]
    first, second = fake.calls
    assert ["image" in block for block in first["messages"][0]["content"]] == [False, False, True, True]
    assert first["toolConfig"]["toolChoice"] == {"any": {}}
    assert [t["toolSpec"]["name"] for t in first["toolConfig"]["tools"]] == ["record_decisions", "zoom"]
    reply = second["messages"][-1]["content"][0]["toolResult"]
    assert reply["toolUseId"] == "z0" and "image" in reply["content"][0]


def test_zooms_run_out_then_a_decision_is_forced(monkeypatch, save):
    review_everything(monkeypatch)
    _, b, result = borderline_pair(save)
    fake = use(monkeypatch, FakeBedrock(
        [zoom(n) for n in range(agent.ZOOMS_PER_PAIR + 1)],
        [record(pairs=[verdict("unsure", "Too close to call")])],
    ))
    decision = {d["file"]: d for d in agent.decide_actions(result)}[b]

    replies = [r["toolResult"] for r in fake.calls[1]["messages"][-1]["content"]]
    assert [r.get("status") for r in replies] == [None] * agent.ZOOMS_PER_PAIR + ["error"]
    assert fake.calls[1]["toolConfig"]["toolChoice"] == {"tool": {"name": "record_decisions"}}
    assert decision["action"] == "quarantine" and decision["disputed"]
    assert len(decision["visual_review"]["zooms"]) == agent.ZOOMS_PER_PAIR


def test_claude_can_dispute_a_leak_but_not_clear_it_or_pick_which_photo_goes(monkeypatch, save):
    review_everything(monkeypatch)
    image = scene(26)
    train, test = save("a_train.png", image), save("b_test.png", crop(image))
    result = {"duplicate_pairs": [], "similar_pairs": [], "blurry_images": [], "risky_orientation_images": [],
              "leaked_pairs": [], "similar_leaks": [(train, test, 0.9, 30)]}
    use(monkeypatch, FakeBedrock([record(
        {"file": "img0", "action": "keep", "reason": "Ignored: this pair gets a verdict"},
        {"file": "img1", "action": "quarantine", "reason": "Ignored: this pair gets a verdict"},
        pairs=[verdict("different", "Different subjects")],
    )]))
    decisions = {d["file"]: d for d in agent.decide_actions(result)}
    assert decisions[train]["action"] == "quarantine" and decisions[train]["disputed"]
    assert decisions[test]["action"] == "keep" and "disputed" not in decisions[test]
    assert decisions[train]["reason"] == "Different subjects"


def test_a_pair_left_without_a_verdict_stays_removed_and_disputed(monkeypatch, save):
    review_everything(monkeypatch)
    _, b, result = borderline_pair(save)
    use(monkeypatch, FakeBedrock([record()]))
    decision = {d["file"]: d for d in agent.decide_actions(result)}[b]
    assert decision["action"] == "quarantine" and decision["disputed"]


def test_one_pairs_zooms_cant_starve_the_other(monkeypatch, save):
    # Seen live: Claude asked for every zoom at once, and a shared budget left the second pair with none.
    review_everything(monkeypatch)
    first, second = scene(24), scene(25)
    p = (save("p_a.png", first), save("p_b.png", crop(first)))
    q = (save("q_a.png", second), save("q_b.png", crop(second)))
    result = {"duplicate_pairs": [], "similar_pairs": [(*p, 0.9, 30), (*q, 0.9, 31)],
              "blurry_images": [], "risky_orientation_images": []}
    fake = use(monkeypatch, FakeBedrock(
        [zoom(n) for n in range(agent.ZOOMS_PER_PAIR + 1)] + [zoom(9, pair="pair1")], [record()]))
    agent.decide_actions(result)
    replies = [r["toolResult"] for r in fake.calls[1]["messages"][-1]["content"]]
    assert [r.get("status") for r in replies] == [None] * agent.ZOOMS_PER_PAIR + ["error", None]


def test_a_model_that_never_stops_zooming_is_cut_off(monkeypatch, save):
    review_everything(monkeypatch)
    _, _, result = borderline_pair(save)
    fake = use(monkeypatch, FakeBedrock(*[[zoom(n, pair="pair9")] for n in range(agent.MAX_ZOOM_ROUNDS)], [record()]))
    agent.decide_actions(result)
    assert len(fake.calls) == agent.MAX_ZOOM_ROUNDS + 1
    assert fake.calls[-1]["toolConfig"]["toolChoice"] == {"tool": {"name": "record_decisions"}}


def test_a_bad_zoom_request_goes_back_to_claude_instead_of_failing_the_scan(monkeypatch, save):
    review_everything(monkeypatch)
    _, _, result = borderline_pair(save)
    fake = use(monkeypatch, FakeBedrock([zoom(pair="pair9"), zoom(n=1, box=(0, 0, 1))], [record()]))
    agent.decide_actions(result)
    replies = [r["toolResult"] for r in fake.calls[1]["messages"][-1]["content"]]
    assert [r["status"] for r in replies] == ["error", "error"]


def finding(a, b, inliers, kind="edited_copy"):
    return {"type": kind, "files": [a, b], "similarity": 0.85, "matching_keypoints": inliers}


def test_only_borderline_edited_copies_are_reviewed_leaks_first(save):
    image = scene(21)
    a, b = save("a.png", image), save("b.png", crop(image))
    confident, close, closer = finding(a, b, 200), finding(a, b, 30), finding(a, b, 26)
    leak = finding(a, b, 33, "train_test_leak_edited_copy")
    hashed = {"type": "duplicate", "files": [a, b], "distance": 0}
    blurry = {"type": "blurry", "file": a, "sharpness": 3.0}
    assert agent.select_for_review([confident, close, hashed, leak, closer, blurry]) == [leak, closer, close]


def test_a_match_confined_to_a_small_patch_is_reviewed_despite_many_inliers(save):
    # Two different photos sharing one small region, like the same watermark or brochure layout.
    source, other = scene(22), scene(23)
    other[40:200, 40:200] = source[40:200, 40:200]
    shared_patch = finding(save("a.png", source), save("b.png", other), 80)
    assert agent.select_for_review([shared_patch]) == [shared_patch]


def test_big_scans_skip_the_photos(monkeypatch):
    review_everything(monkeypatch)
    fake = use(monkeypatch, FakeBedrock([record()]))
    blurry = [(f"/tmp/x/p{i}.png", 1.0) for i in range(agent.REVIEW_MAX_FINDINGS)]
    agent.decide_actions({"duplicate_pairs": [], "similar_pairs": [(A, B, 0.9, 30)],
                          "blurry_images": blurry, "risky_orientation_images": []})
    call = fake.calls[0]
    assert len(call["messages"][0]["content"]) == 1
    assert call["toolConfig"]["toolChoice"] == {"tool": {"name": "record_decisions"}}


class FakeS3:
    exceptions = type("Exceptions", (), {"ClientError": Exception})

    def __init__(self):
        self.copied = []

    def copy_object(self, **kwargs):
        self.copied.append(kwargs["Key"])


def test_quarantine_copies_only_known_files_marked_quarantine(monkeypatch):
    fake = use(monkeypatch, FakeS3())
    moved = agent.apply_quarantine(
        "bucket",
        [{"file": "/l/a.png", "action": "quarantine"},
         {"file": "/l/b.png", "action": "keep"},
         {"file": "/l/c.png", "action": "cannot_confirm"},
         {"file": "/l/unknown.png", "action": "quarantine"}],
        {"/l/a.png": "uploads/s/images/0/a.png", "/l/b.png": "uploads/s/images/1/b.png",
         "/l/c.png": "uploads/s/images/2/c.png"},
        "uploads/s/quarantine/",
    )
    assert moved == fake.copied == ["uploads/s/quarantine/a.png"]
