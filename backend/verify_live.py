import urllib.request
import json

tests = [
    "The battery life is extraordinary and the screen is beautiful.",
    "I regret buying this phone, it is super laggy and crashes constantly.",
    "The design looked promising, however the performance was quite disappointing.",
    "Staff was extremely polite, room was clean and spacious.",
    "Disgusting meal, cold soup and rude waitress.",
    "The quick brown fox jumps over the lazy dog." # Neutral sentence
]

print("Verifying live FastAPI sentiment endpoint...")
for t in tests:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/sentiment/classify",
        data=json.dumps({"text": t}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        score = res["running_score"][-1] * 100
        print(f"[{res['final_label']:8s} {score:5.1f}%] \"{t}\"")
