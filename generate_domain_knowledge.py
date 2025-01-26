import json


domain_knowledge = {
    "competitors": ["CompetitorX", "CompetitorY", "CompetitorZ"],
    "features": ["analytics", "AI engine", "data pipeline"],
    "pricing_keywords": ["discount", "renewal cost", "budget", "pricing model"]
}


with open("domain_knowledge.json", "w") as f:
    json.dump(domain_knowledge, f, indent=4)

print("Domain knowledge saved as domain_knowledge.json!")
import json


domain_knowledge = {
    "competitors": ["CompetitorX", "CompetitorY", "CompetitorZ"],
    "features": ["analytics", "AI engine", "data pipeline"],
    "pricing_keywords": ["discount", "renewal cost", "budget", "pricing model"]
}

with open("domain_knowledge.json", "w") as f:
    json.dump(domain_knowledge, f, indent=4)

print("Domain knowledge saved as domain_knowledge.json!")

