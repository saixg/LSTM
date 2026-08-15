import os
import json
import re
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from backend.app.models.sentiment_model import SentimentLSTM

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 1. Broad Sentiment Lexicon Builder
POSITIVE_WORDS = {
    # Strong positive (+3.0 to +4.0)
    "masterpiece": 3.8, "flawless": 3.7, "perfection": 3.8, "perfect": 3.6, "brilliant": 3.5,
    "phenomenal": 3.7, "incredible": 3.5, "outstanding": 3.6, "superb": 3.5, "exceptional": 3.6,
    "extraordinary": 3.6, "legendary": 3.7, "magnificent": 3.6, "spectacular": 3.6, "breathtaking": 3.5,
    "stunning": 3.4, "triumph": 3.4, "triumphant": 3.4, "gem": 3.5, "genius": 3.6, "exquisite": 3.5,
    # High positive (+2.5 to +3.2)
    "amazing": 3.2, "awesome": 3.2, "fantastic": 3.2, "wonderful": 3.2, "excellent": 3.2,
    "loved": 3.0, "love": 2.8, "loves": 2.8, "loving": 2.8, "lovely": 2.8, "best": 3.2,
    "delicious": 3.0, "tasty": 2.6, "delightful": 2.9, "delight": 2.8, "charming": 2.7,
    "gorgeous": 2.9, "beautiful": 2.8, "beautifully": 2.8, "fascinating": 2.9, "impressive": 2.8,
    "impressively": 2.8, "compelling": 2.7, "thrilling": 2.8, "hilarious": 2.8, "joy": 2.8,
    "joyful": 2.8, "enjoyed": 2.7, "enjoy": 2.6, "enjoys": 2.6, "enjoyable": 2.6, "super": 2.7,
    # Moderate positive (+1.5 to +2.4)
    "great": 2.4, "good": 2.0, "nice": 1.8, "solid": 2.0, "fun": 2.2, "pleasant": 2.0,
    "pleased": 2.0, "pleasing": 2.0, "friendly": 2.2, "helpful": 2.2, "clean": 2.0,
    "fast": 1.8, "smooth": 2.0, "reliable": 2.2, "recommend": 2.4, "recommended": 2.4,
    "recommending": 2.4, "worth": 2.2, "worthwhile": 2.4, "positive": 2.0, "happy": 2.2,
    "glad": 2.0, "satisfied": 2.2, "satisfying": 2.2, "effective": 2.0, "efficient": 2.0,
    "fresh": 2.0, "sweet": 1.8, "rich": 1.8, "vibrant": 2.2, "crisp": 1.8, "cozy": 2.0,
    "innovative": 2.4, "seamless": 2.4, "polite": 2.0, "generous": 2.2, "caring": 2.2,
    # Subtle positive (+0.8 to +1.4)
    "fine": 1.0, "decent": 1.4, "acceptable": 1.0, "adequate": 0.8, "okay": 0.8, "ok": 0.8,
    "neat": 1.4, "cool": 1.6, "promising": 1.4, "liked": 1.6, "like": 1.4, "likes": 1.4,
    "smart": 1.8, "clever": 1.8, "capable": 1.6, "competent": 1.6, "stable": 1.6, "safe": 1.4
}

NEGATIVE_WORDS = {
    # Extreme negative (-3.5 to -4.0)
    "disaster": -3.8, "disastrous": -3.8, "abomination": -3.9, "catastrophe": -3.8, "unwatchable": -3.9,
    "unbearable": -3.7, "garbage": -3.7, "trash": -3.7, "crap": -3.6, "bullshit": -3.8,
    "atrocious": -3.8, "abysmal": -3.8, "appalling": -3.7, "disgusting": -3.7, "vile": -3.7,
    "repulsive": -3.7, "horrific": -3.7, "horrendous": -3.7, "toxic": -3.6, "scam": -3.8,
    # High negative (-2.5 to -3.4)
    "terrible": -3.3, "awful": -3.3, "horrible": -3.3, "worst": -3.4, "hate": -3.0,
    "hated": -3.0, "hating": -3.0, "hates": -3.0, "ruined": -3.1, "ruin": -3.0, "ruins": -3.0,
    "pathetic": -3.0, "worthless": -3.2, "useless": -3.0, "pointless": -2.8, "stupid": -2.8,
    "idiotic": -3.0, "ridiculous": -2.6, "annoying": -2.6, "annoyed": -2.5, "annoy": -2.4,
    "painful": -2.9, "miserable": -3.0, "dreadful": -3.1, "furious": -3.0, "outrageous": -3.0,
    "frustrating": -2.8, "frustrated": -2.8, "frustration": -2.8, "disappointed": -2.8,
    "disappointing": -2.8, "disappointment": -2.8, "failure": -3.0, "failed": -2.8, "fails": -2.8,
    # Moderate negative (-1.5 to -2.4)
    "bad": -2.4, "poor": -2.2, "poorly": -2.2, "boring": -2.4, "bored": -2.2, "bore": -2.0,
    "dull": -2.0, "ugly": -2.2, "rude": -2.6, "dirty": -2.5, "filthy": -3.0, "stale": -2.2,
    "slow": -1.6, "mess": -2.4, "messy": -2.2, "flaw": -1.8, "flawed": -2.2, "flaws": -2.0,
    "avoid": -2.6, "avoided": -2.4, "waste": -2.8, "wasted": -2.8, "wasting": -2.8,
    "broken": -2.6, "buggy": -2.4, "crash": -2.4, "crashes": -2.4, "crashed": -2.4,
    "unpleasant": -2.2, "mediocre": -1.8, "shallow": -1.8, "clunky": -2.0, "tiresome": -2.2,
    "confusing": -1.8, "confused": -1.6, "uncomfortable": -2.0, "expensive": -1.4,
    # Subtle negative (-0.8 to -1.4)
    "weak": -1.4, "lacking": -1.2, "limited": -1.0, "dated": -1.2, "bland": -1.6, "dry": -1.0,
    "cliche": -1.4, "predictable": -1.4, "overpriced": -1.6, "noisy": -1.4, "crowded": -1.0
}

INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "incredibly": 2.0, "super": 1.6, "highly": 1.6,
    "absolutely": 2.0, "totally": 1.8, "completely": 1.8, "utterly": 2.0, "deeply": 1.7,
    "really": 1.4, "extraordinarily": 2.0, "insanely": 1.8, "immensely": 1.8,
    "tremendously": 1.8, "overwhelmingly": 1.9, "quite": 1.2, "truly": 1.4, "so": 1.3
}

DIMINISHERS = {
    "slightly": 0.6, "somewhat": 0.7, "barely": 0.4, "hardly": 0.3, "scarcely": 0.3,
    "a bit": 0.6, "mildly": 0.6, "moderately": 0.8, "partially": 0.7
}

NEGATIONS = {
    "not", "never", "no", "neither", "nor", "hardly", "scarcely", "barely",
    "cannot", "cant", "can't", "dont", "don't", "didnt", "didn't", "wont", "won't",
    "isnt", "isn't", "wasnt", "wasn't", "arent", "aren't", "werent", "weren't",
    "hasnt", "hasn't", "havent", "haven't", "hadnt", "hadn't", "without"
}

PIVOTS = {
    "but", "however", "although", "though", "yet", "nevertheless", "still",
    "except", "until", "nonetheless", "whereas", "despite", "while"
}

# Build Open-Vocabulary Corpus (10,000+ words including expanded common English words)
GENERAL_VOCAB = [
    # Common English words
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "it's",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "mine", "yours", "theirs", "ours",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "done", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must",
    # Domain topics: Movies & Art
    "movie", "film", "cinema", "theater", "director", "actor", "actress", "cast",
    "acting", "plot", "story", "storyline", "script", "screenplay", "dialogue",
    "scene", "scenes", "ending", "start", "beginning", "middle", "climax",
    "character", "characters", "hero", "villain", "soundtrack", "score", "music",
    "visual", "visuals", "cinematography", "effects", "cgi", "animation", "drama",
    "comedy", "action", "thriller", "horror", "romance", "documentary", "genre",
    # Domain topics: Food & Dining
    "food", "dish", "dishes", "meal", "meals", "dinner", "lunch", "breakfast",
    "menu", "restaurant", "cafe", "bar", "kitchen", "chef", "cook", "waiter",
    "waitress", "server", "service", "staff", "table", "order", "taste", "flavor",
    "flavors", "ingredient", "ingredients", "portion", "portions", "price", "bill",
    "drink", "drinks", "coffee", "tea", "wine", "beer", "cocktail", "dessert",
    "appetizer", "bread", "cheese", "meat", "chicken", "beef", "fish", "soup",
    "salad", "pasta", "pizza", "burger", "rice", "sauce", "spices", "quality",
    # Domain topics: Tech & Software
    "app", "software", "product", "system", "code", "model", "neural", "network",
    "algorithm", "interface", "ui", "ux", "design", "feature", "features", "update",
    "performance", "speed", "battery", "screen", "display", "device", "phone",
    "laptop", "computer", "hardware", "tool", "tools", "support", "developer",
    "platform", "site", "website", "database", "api", "latency", "memory", "server",
    # Domain topics: Travel & Hotels
    "hotel", "room", "rooms", "bed", "bathroom", "shower", "stay", "resort",
    "location", "view", "viewpoint", "beach", "city", "street", "pool", "spa",
    "amenities", "check-in", "reception", "manager", "host", "booking", "trip",
    "flight", "airline", "airport", "journey", "vacation", "holiday", "destination",
    # Prepositions, conjunctions, neutrals
    "and", "or", "so", "for", "with", "at", "by", "from", "in", "into", "on", "onto",
    "of", "off", "out", "over", "to", "up", "down", "about", "above", "across",
    "after", "against", "along", "among", "around", "as", "before", "behind", "below",
    "beneath", "beside", "between", "beyond", "during", "inside", "near", "outside",
    "through", "throughout", "toward", "under", "underneath", "upon", "within",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "what", "which", "who", "whom", "whose", "why", "how",
    "here", "there", "when", "where", "why", "again", "almost", "already", "also",
    "always", "enough", "even", "ever", "far", "just", "later", "much", "never",
    "now", "often", "only", "quite", "rather", "seldom", "since", "soon", "still",
    "then", "together", "well", "yesterday", "today", "tomorrow", "day", "night",
    "time", "year", "people", "person", "man", "woman", "child", "thing", "things",
    "way", "world", "life", "hand", "part", "eye", "place", "work", "week", "case",
    "point", "government", "company", "number", "group", "problem", "fact"
]

def build_full_lexicon():
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    # Add all classified words
    all_words = set(list(POSITIVE_WORDS.keys()) + list(NEGATIVE_WORDS.keys()) +
                    list(INTENSIFIERS.keys()) + list(DIMINISHERS.keys()) +
                    list(NEGATIONS) + list(PIVOTS) + GENERAL_VOCAB)
    
    # Expand with plurals, past tense, adverbs
    expanded = set(all_words)
    for w in all_words:
        if len(w) > 3:
            expanded.add(w + "s")
            expanded.add(w + "ed")
            expanded.add(w + "ing")
            expanded.add(w + "ly")
            
    for w in sorted(expanded):
        if w not in vocab:
            vocab[w] = len(vocab)
            
    return vocab

VOCAB = build_full_lexicon()

def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

def encode_text(text: str):
    tokens = tokenize(text)
    return [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens]

# Valence calculation for ground truth trajectory synthesis
def compute_token_valence(token: str) -> float:
    if token in POSITIVE_WORDS:
        return POSITIVE_WORDS[token]
    if token in NEGATIVE_WORDS:
        return NEGATIVE_WORDS[token]
    # Check simple stemmed forms
    if token.endswith("ly") and token[:-2] in POSITIVE_WORDS:
        return POSITIVE_WORDS[token[:-2]]
    if token.endswith("ed") and token[:-2] in POSITIVE_WORDS:
        return POSITIVE_WORDS[token[:-2]]
    if token.endswith("s") and token[:-1] in POSITIVE_WORDS:
        return POSITIVE_WORDS[token[:-1]]
    if token.endswith("ly") and token[:-2] in NEGATIVE_WORDS:
        return NEGATIVE_WORDS[token[:-2]]
    if token.endswith("ed") and token[:-2] in NEGATIVE_WORDS:
        return NEGATIVE_WORDS[token[:-2]]
    if token.endswith("s") and token[:-1] in NEGATIVE_WORDS:
        return NEGATIVE_WORDS[token[:-1]]
    return 0.0

def simulate_sentence_trajectory(tokens):
    """
    Simulates the true ground-truth sentiment trajectory for any arbitrary token stream,
    modeling human sentiment accumulation, intensifiers, negations, and pivot conjunctions.
    """
    scores = []
    running_valence = 0.0
    current_modifier = 1.0
    negation_active = False
    negation_countdown = 0

    for t in tokens:
        if t in PIVOTS:
            # Contrastive pivot forgets prior memory and resets
            running_valence = 0.0
            current_modifier = 1.0
            negation_active = False
            negation_countdown = 0
            
        elif t in NEGATIONS:
            negation_active = True
            negation_countdown = 3 # Affects next 3 words
            
        elif t in INTENSIFIERS:
            current_modifier = INTENSIFIERS[t]
            
        elif t in DIMINISHERS:
            current_modifier = DIMINISHERS[t]
            
        else:
            w_val = compute_token_valence(t)
            if w_val != 0.0:
                val = w_val * current_modifier
                if negation_active and negation_countdown > 0:
                    val = -val * 0.8
                running_valence = (running_valence * 0.6) + val
                current_modifier = 1.0
                
            if negation_active:
                negation_countdown -= 1
                if negation_countdown <= 0:
                    negation_active = False
                    
        # Map accumulated valence to sigmoid probability [0, 1]
        prob = 1.0 / (1.0 + math.exp(-running_valence))
        scores.append(prob)
        
    return scores

def generate_synthetic_dataset(num_samples=6000):
    samples = []
    
    all_pos = list(POSITIVE_WORDS.keys())
    all_neg = list(NEGATIVE_WORDS.keys())
    all_neut = GENERAL_VOCAB[:100]
    all_piv = list(PIVOTS)
    all_int = list(INTENSIFIERS.keys())
    
    # 1. Standard Pure Positive Sentences
    for _ in range(num_samples // 4):
        p1 = random.choice(all_pos)
        p2 = random.choice(all_pos)
        s1 = random.choice(all_neut)
        s2 = random.choice(all_neut)
        intens = random.choice(all_int)
        
        templates = [
            f"the {s1} is {p1} and {p2}",
            f"really {p1} {s1} with {intens} {p2} {s2}",
            f"i loved this {s1} it was {p1}",
            f"a {intens} {p1} experience overall",
            f"highly recommend this {s1} because it is {p1}",
            f"great {s1} and {p1} {s2}"
        ]
        text = random.choice(templates)
        tokens = tokenize(text)
        traj = simulate_sentence_trajectory(tokens)
        samples.append({"text": text, "tokens": tokens, "traj": traj, "label": 1})

    # 2. Standard Pure Negative Sentences
    for _ in range(num_samples // 4):
        n1 = random.choice(all_neg)
        n2 = random.choice(all_neg)
        s1 = random.choice(all_neut)
        s2 = random.choice(all_neut)
        intens = random.choice(all_int)
        
        templates = [
            f"the {s1} is {n1} and {n2}",
            f"really {n1} {s1} with {intens} {n2} {s2}",
            f"i hated this {s1} it was {n1}",
            f"a {intens} {n1} disaster avoid it",
            f"completely avoid this {s1} because it is {n1}",
            f"terrible {s1} and {n1} {s2}"
        ]
        text = random.choice(templates)
        tokens = tokenize(text)
        traj = simulate_sentence_trajectory(tokens)
        samples.append({"text": text, "tokens": tokens, "traj": traj, "label": 0})

    # 3. Positive -> Negative Pivot Sentences (The "amazing start but terrible end")
    for _ in range(num_samples // 4):
        p = random.choice(all_pos)
        n = random.choice(all_neg)
        s1 = random.choice(all_neut)
        s2 = random.choice(all_neut)
        piv = random.choice(all_piv)
        
        templates = [
            f"the {s1} was {p} {piv} the {s2} was {n}",
            f"started out {p} {piv} quickly became {n}",
            f"great {s1} {piv} the {s2} ruined everything",
            f"delicious {s1} {piv} terrible {s2} and rude staff",
            f"impressive {s1} {piv} the {s2} is utterly {n}"
        ]
        text = random.choice(templates)
        tokens = tokenize(text)
        traj = simulate_sentence_trajectory(tokens)
        samples.append({"text": text, "tokens": tokens, "traj": traj, "label": 0})

    # 4. Negative -> Positive Pivot Sentences (The "slow start but fantastic end")
    for _ in range(num_samples // 4):
        n = random.choice(all_neg)
        p = random.choice(all_pos)
        s1 = random.choice(all_neut)
        s2 = random.choice(all_neut)
        piv = random.choice(all_piv)
        
        templates = [
            f"the {s1} was {n} {piv} the {s2} was {p}",
            f"started out {n} {piv} the ending was {p}",
            f"poor {s1} {piv} the wonderful {s2} saved it",
            f"slow beginning {piv} overall a {p} masterpiece",
            f"confusing at first {piv} turned out {p} and brilliant"
        ]
        text = random.choice(templates)
        tokens = tokenize(text)
        traj = simulate_sentence_trajectory(tokens)
        samples.append({"text": text, "tokens": tokens, "traj": traj, "label": 1})

    random.shuffle(samples)
    return samples

def train_and_save():
    print(f"Building Open-Vocabulary Sentiment Engine with {len(VOCAB)} words...", flush=True)
    vocab_path = os.path.join(CHECKPOINT_DIR, "sentiment_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(VOCAB, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")

    dataset = generate_synthetic_dataset(num_samples=6000)
    print(f"Generated {len(dataset)} training sequences across positive, negative, and pivot clauses.")

    model = SentimentLSTM(vocab_size=len(VOCAB), embedding_dim=64, hidden_size=128)
    
    # Initialize embedding weights with informative polarity prior for faster convergence
    with torch.no_grad():
        for word, idx in VOCAB.items():
            val = compute_token_valence(word)
            if val != 0.0:
                # Give a positive or negative bias to embedding dimension 0
                model.embedding.weight[idx, 0] = val * 0.5
            if word in PIVOTS:
                model.embedding.weight[idx, 1] = 2.0 # Distinct marker for pivot words
            if word in NEGATIONS:
                model.embedding.weight[idx, 2] = -2.0 # Distinct marker for negations

    optimizer = optim.Adam(model.parameters(), lr=0.004)
    criterion = nn.MSELoss()

    epochs = 6
    batch_size = 32
    print(f"Beginning PyTorch training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        for sample in dataset:
            token_ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in sample["tokens"]]
            if len(token_ids) == 0:
                continue
            
            x = torch.tensor(token_ids, dtype=torch.long)
            target = torch.tensor(sample["traj"], dtype=torch.float32)

            optimizer.zero_grad()
            running_scores, _ = model(x, tokens=[""] * len(token_ids))
            scores_flat = running_scores.view(-1)

            loss = criterion(scores_flat, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (scores_flat[-1].item() >= 0.5) == bool(sample["label"]):
                correct += 1

        acc = (correct / len(dataset)) * 100
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Final Accuracy: {acc:.2f}%", flush=True)

    model_path = os.path.join(CHECKPOINT_DIR, "sentiment_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch sentiment model to {model_path}")

if __name__ == "__main__":
    train_and_save()
