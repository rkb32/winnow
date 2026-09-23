"""Are Claude's doubts about the detector's borderline calls well aimed?

Runs the live agent (winnow/agent.py) on every full-audit leak pair it would send to visual review,
one pair per call. The agent can't clear a flagged leak, only dispute it: the training photo is
still removed by default, and the dispute asks a person to look. This scores the disputes against
the AI-verified verdicts in full_audit/verified_pairs.csv: disputing a false match is a catch,
disputing a real leak is a wasted look. Needs AWS credentials with Bedrock access.
"""
import collections
import csv
import sys

sys.path.insert(0, "/app")
import boto3
import agent

ROOT = "/data/imagenette2-160"
PRICE_IN, PRICE_OUT = 1 / 1e6, 5 / 1e6  # Claude Haiku 4.5 on Bedrock, per token

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
tokens = collections.Counter()


class Metered:
    def converse(self, **kwargs):
        response = bedrock.converse(**kwargs)
        tokens["in"] += response["usage"]["inputTokens"]
        tokens["out"] += response["usage"]["outputTokens"]
        tokens["calls"] += 1
        return response


agent.boto3.client = lambda *a, **k: Metered()

rows = [r for r in csv.DictReader(open("/data/full_audit/verified_pairs.csv", encoding="utf-8"))
        if r["method"] == "keypoints"]
outcomes = collections.Counter()
misses = []
for r in rows:
    train, val = f"{ROOT}/{r['train_file']}", f"{ROOT}/{r['val_file']}"
    result = {"duplicate_pairs": [], "similar_pairs": [], "blurry_images": [], "risky_orientation_images": [],
              "leaked_pairs": [], "similar_leaks": [(train, val, float(r["similarity"]), int(r["score"]))]}
    if not agent.select_for_review(agent.build_findings_summary(result)):
        continue
    decision = next((d for d in agent.decide_actions(result) if d["file"] == train), None)
    real = r["verdict"] != "different"
    if decision is None:
        outcome = "no decision"
    elif decision["action"] != "quarantine":
        outcome = "cleared"  # must never happen: the agent isn't allowed to clear a flagged leak
    else:
        outcome = "disputed" if decision.get("disputed") else "confirmed"
    outcomes[real, outcome] += 1
    if outcome != ("confirmed" if real else "disputed"):
        misses.append(f"  #{r['id']} {'real leak' if real else 'false match'} -> {outcome}: "
                      f"{(decision or {}).get('reason', '')}")
    print(f"#{r['id']:>3} {'real ' if real else 'false'} {outcome:11} {(decision or {}).get('reason', '')[:80]}",
          flush=True)

real_total = sum(n for (real, _), n in outcomes.items() if real)
false_total = sum(n for (real, _), n in outcomes.items() if not real)
disputes = outcomes[False, "disputed"] + outcomes[True, "disputed"]
print(f"\nreviewed: {real_total} real leaks, {false_total} false matches")
print(f"false matches disputed (caught):  {outcomes[False, 'disputed']}/{false_total}   (without review: 0)")
print(f"real leaks disputed (wasted look): {outcomes[True, 'disputed']}/{real_total}")
print(f"disputes that were right:         {outcomes[False, 'disputed']}/{disputes}")
print(f"leaks let through by default:     {outcomes[True, 'cleared'] + outcomes[False, 'cleared']}   "
      f"(no decision: {outcomes[True, 'no decision'] + outcomes[False, 'no decision']})")
cost = tokens["in"] * PRICE_IN + tokens["out"] * PRICE_OUT
print(f"\n{tokens['calls']} Bedrock calls, {tokens['in']} in / {tokens['out']} out tokens, ${cost:.3f} total, "
      f"${cost / max(real_total + false_total, 1):.4f} per reviewed pair")
print("\nnot right:")
print("\n".join(misses) or "  none")
