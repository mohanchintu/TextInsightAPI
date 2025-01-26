import pandas as pd
import random


labels = ["Objection", "Pricing Discussion", "Security", "Competition"]
competitors = ["CompetitorX", "CompetitorY", "CompetitorZ"]
features = ["analytics", "AI engine", "data pipeline"]
pricing_keywords = ["discount", "renewal cost", "budget", "pricing model"]


def generate_snippet():
    snippets = [
        f"We love the {random.choice(features)}, but {random.choice(competitors)} has a cheaper subscription.",
        f"Our compliance team is worried about data handling. Are you SOC2 certified?",
        f"What pricing model do you offer compared to {random.choice(competitors)}?",
        f"We are considering a discount, but our {random.choice(features)} needs better integration.",
        f"Security concerns like data breaches are a top priority for us.",
        f"Can you offer a better renewal cost? We also like {random.choice(competitors)}'s {random.choice(features)}.",
        f"We are evaluating the analytics capabilities of your solution vs {random.choice(competitors)}.",
        f"Data pipeline efficiency matters a lot. Can you customize it?"
    ]
    return random.choice(snippets)


def generate_labels():
    num_labels = random.randint(1, 3)
    return ", ".join(random.sample(labels, num_labels))


def create_dataset(num_rows=100):
    dataset = {
        "id": list(range(1, num_rows + 1)),
        "text_snippet": [generate_snippet() for _ in range(num_rows)],
        "labels": [generate_labels() for _ in range(num_rows)]
    }
    return pd.DataFrame(dataset)


dataset = create_dataset(100)  
dataset.to_csv("calls_dataset.csv", index=False)
print("Synthetic dataset saved as calls_dataset.csv!")
