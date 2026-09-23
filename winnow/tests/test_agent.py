import pytest

import agent

A = "/tmp/x/uploads__s__images__0__a.png"
B = "/tmp/x/uploads__s__images__1__b.png"
C = "/tmp/x/uploads__s__images__2__c.png"
RESULT = {"duplicate_pairs": [(A, B, 0)], "blurry_images": [(C, 12.5)], "risky_orientation_images": []}


class FakeBedrock:
    def __init__(self, decisions, stop_reason="tool_use"):
        self.response = {
            "stopReason": stop_reason,
            "output": {"message": {"content": [{"toolUse": {"input": {"decisions": decisions}}}]}},
        }
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def use(monkeypatch, client):
    monkeypatch.setattr(agent.boto3, "client", lambda *args, **kwargs: client)
    return client


def test_no_findings_never_calls_bedrock(monkeypatch):
    monkeypatch.setattr(agent.boto3, "client", lambda *a, **k: pytest.fail("Bedrock was called"))
    assert agent.decide_actions({"duplicate_pairs": [], "blurry_images": [], "risky_orientation_images": []}) == []


def test_short_ids_map_back_to_paths_and_unknown_ids_are_dropped(monkeypatch):
    fake = use(monkeypatch, FakeBedrock([
        {"file": "img1", "action": "quarantine", "reason": "Duplicate of a.png"},
        {"file": "img9", "action": "quarantine", "reason": "An id that doesn't exist"},
    ]))
    assert agent.decide_actions(RESULT) == [{"file": B, "action": "quarantine", "reason": "Duplicate of a.png"}]

    call = fake.calls[0]
    prompt = call["messages"][0]["content"][0]["text"]
    assert "/tmp/x" not in prompt and '"img1": "b.png"' in prompt
    assert call["toolConfig"]["toolChoice"] == {"tool": {"name": "record_decisions"}}


def test_cut_off_reply_is_an_error_not_an_empty_answer(monkeypatch):
    use(monkeypatch, FakeBedrock([], stop_reason="max_tokens"))
    with pytest.raises(RuntimeError):
        agent.decide_actions(RESULT)


def test_findings_sent_to_claude_are_capped(monkeypatch):
    fake = use(monkeypatch, FakeBedrock([]))
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
         {"file": "/l/unknown.png", "action": "quarantine"}],
        {"/l/a.png": "uploads/s/images/0/a.png", "/l/b.png": "uploads/s/images/1/b.png"},
        "uploads/s/quarantine/",
    )
    assert moved == fake.copied == ["uploads/s/quarantine/a.png"]
