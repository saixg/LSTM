from backend.app.routers.sentiment import classify_sentiment, SentimentRequest

tests = [
    "This software is super fast and incredibly reliable, love the new features!",
    "The app keeps crashing, terrible battery drain and buggy interface.",
    "The pizza was delicious, fresh ingredients and very friendly server.",
    "The appetizers were wonderful, but the main dish was cold and the waiter was rude.",
    "The first half was quite boring, however the ending was absolutely breathtaking and brilliant!",
    "Horrible customer support, complete waste of money and time.",
    "Although the design is neat, the performance is disappointing.",
    "The hotel was luxurious, clean room and stunning ocean view.",
    "Worst vacation ever, dirty bed and unhelpful staff."
]

for t in tests:
    res = classify_sentiment(SentimentRequest(text=t))
    print(f"INPUT: \"{t}\"")
    print(f"RESULT: {res.final_label} ({res.running_score[-1]*100:.1f}%) | Tokens: {len(res.tokens)}")
    print("Trajectory:", [round(s*100, 1) for s in res.running_score])
    print("-" * 65)
