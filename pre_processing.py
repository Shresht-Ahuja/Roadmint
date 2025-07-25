import pandas as pd
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

questions = pd.read_csv("Questions.csv", encoding='latin1')
answers = pd.read_csv("Answers.csv", encoding='latin1')

questions = questions.head(10000)
answers = answers.head(10000)

questions = questions[['Id', 'Title']]
answers = answers[['ParentId', 'Body', 'Score']]

answers_sorted = answers.sort_values(by=['ParentId', 'Score'], ascending=[True, False])
answers_top = answers_sorted.drop_duplicates(subset=['ParentId'], keep='first')

qa_pairs = pd.merge(questions, answers_top, left_on='Id', right_on='ParentId')

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

dataset = []
for _, row in tqdm(qa_pairs.iterrows(), total=len(qa_pairs)):
    prompt = row['Title']
    completion = clean_html(row['Body'])
    
    if len(completion.strip()) > 0 and len(prompt.strip()) > 0:
        dataset.append({
            "prompt": prompt.strip(),
            "completion": completion.strip()
        })

with open("stackoverflow_qa_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n Saved {len(dataset)} prompt-completion pairs to 'stackoverflow_qa_dataset.jsonl'")
