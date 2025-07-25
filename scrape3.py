import json
from pathlib import Path
import random

# Define topics and corresponding subtopics with estimated time
topics_data = {
    "numpy": [
        ("Arrays and Ndarray", 2),
        ("Broadcasting and Indexing", 2),
        ("Vectorized Operations", 3),
        ("Linear Algebra with NumPy", 3),
        ("Advanced NumPy Tricks", 2)
    ],
    "pandas": [
        ("Series and DataFrames", 2),
        ("Data Cleaning", 2),
        ("GroupBy and Aggregation", 3),
        ("Merging and Joining", 2),
        ("Time Series Analysis", 2)
    ],
    "c++": [
        ("Syntax and Basics", 2),
        ("OOP Concepts", 3),
        ("STL (Standard Template Library)", 2),
        ("Memory Management", 2),
        ("Advanced Topics", 3)
    ],
    "operating systems": [
        ("Process Management", 2),
        ("Memory Management", 2),
        ("File Systems", 3),
        ("Scheduling Algorithms", 2),
        ("Concurrency and Deadlocks", 3)
    ],
    "dsa": [
        ("Arrays and Linked Lists", 2),
        ("Stacks and Queues", 2),
        ("Trees and Graphs", 3),
        ("Sorting and Searching", 2),
        ("Dynamic Programming", 3)
    ],
    "computer networks": [
        ("OSI Model", 2),
        ("TCP/IP Protocols", 3),
        ("Routing and Switching", 2),
        ("DNS and DHCP", 2),
        ("Security and Firewalls", 2)
    ],
    "nlp": [
        ("Text Preprocessing", 2),
        ("Tokenization and Embeddings", 2),
        ("POS Tagging and NER", 3),
        ("Sentiment Analysis", 2),
        ("Transformers and BERT", 3)
    ],
    "git": [
        ("Git Basics and Init", 1),
        ("Branches and Merging", 2),
        ("Remote Repos and GitHub", 2),
        ("Rebasing and Cherry-pick", 2),
        ("Best Practices", 2)
    ],
    "docker": [
        ("Containers vs VMs", 2),
        ("Docker Images", 2),
        ("Volumes and Networking", 2),
        ("Docker Compose", 2),
        ("Docker Swarm and Orchestration", 2)
    ],
    "blender": [
        ("User Interface", 1),
        ("Basic Modelling", 2),
        ("Materials and Texturing", 2),
        ("Lighting and Rendering", 2),
        ("Animation and Rigging", 3)
    ],
    "machine learning": [
        ("Supervised Learning", 2),
        ("Unsupervised Learning", 2),
        ("Model Evaluation", 2),
        ("Regression and Classification", 2),
        ("Neural Networks", 3)
    ],
    "devops": [
        ("CI/CD Concepts", 2),
        ("Version Control and Git", 2),
        ("Infrastructure as Code", 2),
        ("Monitoring and Logging", 2),
        ("DevOps Tools Overview", 2)
    ]
}

# Generate 1000 samples by mixing topics and rephrasing prompts
def generate_samples(num_samples=1000):
    samples = []
    topic_keys = list(topics_data.keys())
    for _ in range(num_samples):
        topic = random.choice(topic_keys)
        subtopics = topics_data[topic]
        prompt = f"I want to learn {topic}"
        completion_lines = []
        for i, (sub, time) in enumerate(subtopics, start=1):
            completion_lines.append(f"Step {i}: {sub}\nTime: {time} days")
        completion = "\n".join(completion_lines)
        samples.append({"prompt": prompt, "completion": completion})
    return samples

# Generate and write to JSONL file
output_path = Path(r"C:\Users\Hp\Documents\C3I\roadmap_generator\Roadmint\roadmap_dataset_large.jsonl")
samples = generate_samples(1000)

with open(output_path, "w", encoding="utf-8") as f:
    for entry in samples:
        json.dump(entry, f)
        f.write("\n")

output_path.name
