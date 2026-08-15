import os
import json
import math
import random
import urllib.request
import tarfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def create_tech_corpus():
    """Generates a small tech-flavored corpus for next-word prediction."""
    text = (
        "The AI model learns from a large dataset of text to predict the next word. "
        "Deep learning architectures like LSTM and Transformer are powerful tools for natural language processing. "
        "Training a neural network requires significant computational resources, often utilizing GPUs. "
        "Gradient descent is used to optimize the weights by minimizing the loss function. "
        "In modern machine learning, hyperparameter tuning is essential for achieving state-of-the-art accuracy. "
        "The quick brown fox jumps over the lazy dog but the neural network predicts the pattern. "
        "Artificial intelligence will transform how software engineers write code and deploy applications. "
        "Hackathons provide a great environment to build prototypes and experiment with new frameworks. "
        "The model architecture consists of an embedding layer followed by recurrent cells. "
        "Overfitting occurs when the model memorizes the training data but fails to generalize to unseen examples. "
        "Regularization techniques such as dropout help mitigate overfitting in deep neural networks. "
        "Data preprocessing is a crucial step in the machine learning pipeline. "
    ) * 50 # Repeat to artificially inflate corpus size for training stability
    
    path = os.path.join(DATA_DIR, "tech_corpus.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Created {path}")

def create_anomaly_data():
    """Generates synthetic sensor data with and without anomalies."""
    seq_len = 100
    samples = []
    
    # Generate 5 normal baseline samples
    for i in range(5):
        base_freq = random.uniform(0.05, 0.15)
        # Normal sine wave with slight noise
        sequence = [math.sin(t * base_freq) + random.uniform(-0.1, 0.1) for t in range(seq_len)]
        samples.append({"id": f"normal_{i}", "type": "normal", "data": sequence})
    
    # Generate 2 samples with anomalies
    for i in range(2):
        base_freq = random.uniform(0.05, 0.15)
        sequence = [math.sin(t * base_freq) + random.uniform(-0.1, 0.1) for t in range(seq_len)]
        # Inject spike anomaly in the middle
        anomaly_start = random.randint(40, 60)
        for t in range(anomaly_start, anomaly_start + 5):
            sequence[t] += random.uniform(1.5, 2.5) # Huge spike
        samples.append({"id": f"anomaly_{i}", "type": "anomaly", "data": sequence})
        
    path = os.path.join(DATA_DIR, "anomaly_samples.json")
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Created {path}")

def create_sentiment_data():
    """Generates a rich synthetic sentiment dataset with:
    - Pure positive reviews
    - Pure negative reviews
    - Positive-to-negative sentiment pivot sentences (e.g., 'good start but awful ending')
    - Negative-to-positive sentiment pivot sentences (e.g., 'slow start but amazing ending')
    - Negation structures ('not good', 'not bad', 'never boring')
    - Per-token target trajectories for running score supervision
    """
    pos_adjectives = [
        "amazing", "great", "excellent", "fantastic", "wonderful", "masterpiece",
        "brilliant", "outstanding", "superb", "incredible", "loved", "love",
        "best", "perfect", "phenomenal", "flawless", "delightful", "gem",
        "tasty", "delicious", "fresh", "friendly", "clean", "fast", "helpful",
        "recommend", "worth", "enjoyed", "pleasant", "good", "nice", "solid",
        "fun", "entertaining", "compelling", "impressive", "charming", "awesome"
    ]
    
    neg_adjectives = [
        "terrible", "awful", "horrible", "worst", "hate", "hated", "boring",
        "disaster", "garbage", "trash", "pathetic", "stupid", "ugly", "poor",
        "waste", "ruined", "bad", "slow", "annoying", "dull", "unpleasant",
        "rude", "dirty", "stale", "mess", "flawed", "ridiculous", "avoid",
        "disappointed", "disappointing", "failure", "crap", "pointless", "mediocre"
    ]
    
    subjects = [
        "movie", "film", "food", "service", "meal", "acting", "actor", "actress",
        "director", "plot", "story", "script", "dialogue", "scene", "ending",
        "start", "beginning", "climax", "cast", "music", "visuals", "staff",
        "restaurant", "hotel", "room", "experience", "performance", "place"
    ]
    
    pivots = ["but", "however", "although", "though", "yet", "until", "nevertheless", "still"]
    
    samples = []
    
    # 1. Pure Positive Sentences
    for _ in range(400):
        s1 = random.choice(subjects)
        s2 = random.choice(subjects)
        p1 = random.choice(pos_adjectives)
        p2 = random.choice(pos_adjectives)
        templates = [
            f"the {s1} was {p1} and the {s2} was {p2}",
            f"a truly {p1} {s1} with {p2} {s2}",
            f"the {s1} was {p1} and very {p2}",
            f"i loved this {s1} it was {p1} and {p2}",
            f"great {s1} {p1} {s2} and {p2} experience",
            f"highly recommend this {s1} absolutely {p1}",
            f"the {s1} is {p1} with {p2} {s2}"
        ]
        text = random.choice(templates)
        tokens = text.lower().split()
        target = [0.95] * len(tokens)
        samples.append({"text": text, "label": 1, "trajectory": target})
        
    # 2. Pure Negative Sentences
    for _ in range(400):
        s1 = random.choice(subjects)
        s2 = random.choice(subjects)
        n1 = random.choice(neg_adjectives)
        n2 = random.choice(neg_adjectives)
        templates = [
            f"the {s1} was {n1} and the {s2} was {n2}",
            f"a truly {n1} {s1} with {n2} {s2}",
            f"the {s1} was {n1} and very {n2}",
            f"i hated this {s1} it was {n1} and {n2}",
            f"terrible {s1} {n1} {s2} and {n2} experience",
            f"waste of time and money avoid this {s1} {n1}",
            f"the {s1} is {n1} with {n2} {s2}"
        ]
        text = random.choice(templates)
        tokens = text.lower().split()
        target = [0.05] * len(tokens)
        samples.append({"text": text, "label": 0, "trajectory": target})
        
    # 3. Positive -> Negative Pivot Sentences (The "but the service ruined it" pattern)
    for _ in range(500):
        s1 = random.choice(subjects)
        s2 = random.choice(subjects)
        p = random.choice(pos_adjectives)
        n = random.choice(neg_adjectives)
        piv = random.choice(pivots)
        
        part1 = f"the {s1} was {p}"
        part2 = f"{piv} the {s2} was {n}"
        text = f"{part1} {part2}"
        
        toks1 = part1.split()
        toks2 = part2.split()
        
        # Trajectory starts high (~0.9), drops to low (~0.05) after pivot
        target = ([0.92] * len(toks1)) + ([0.08] * len(toks2))
        samples.append({"text": text, "label": 0, "trajectory": target})
        
    # 4. Negative -> Positive Pivot Sentences (The "started slow but ending was fantastic" pattern)
    for _ in range(500):
        s1 = random.choice(subjects)
        s2 = random.choice(subjects)
        n = random.choice(neg_adjectives)
        p = random.choice(pos_adjectives)
        piv = random.choice(pivots)
        
        part1 = f"the {s1} was {n}"
        part2 = f"{piv} the {s2} was {p}"
        text = f"{part1} {part2}"
        
        toks1 = part1.split()
        toks2 = part2.split()
        
        # Trajectory starts low (~0.08), rises to high (~0.92) after pivot
        target = ([0.08] * len(toks1)) + ([0.92] * len(toks2))
        samples.append({"text": text, "label": 1, "trajectory": target})
        
    # 5. Negation Sentences ("not good", "not bad")
    for _ in range(300):
        s = random.choice(subjects)
        p = random.choice(pos_adjectives)
        n = random.choice(neg_adjectives)
        # "not good" -> negative
        text_neg = f"the {s} was not {p} and very disappointing"
        tokens_neg = text_neg.split()
        samples.append({"text": text_neg, "label": 0, "trajectory": [0.08] * len(tokens_neg)})
        
        # "not bad" -> positive/neutral
        text_pos = f"the {s} was not {n} but actually quite {p}"
        tokens_pos = text_pos.split()
        samples.append({"text": text_pos, "label": 1, "trajectory": [0.90] * len(tokens_pos)})
        
    random.shuffle(samples)
    
    path = os.path.join(DATA_DIR, "sentiment_samples.json")
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Created {path} with {len(samples)} diverse sentiment samples.")

if __name__ == "__main__":
    print("Preparing datasets...")
    create_tech_corpus()
    create_anomaly_data()
    create_sentiment_data()
    print("All datasets prepared.")
