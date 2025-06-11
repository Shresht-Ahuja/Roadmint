from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup


def get_learning_links(skill, num_results=5):
    query = f"{skill} tutorial site:geeksforgeeks.org OR site:youtube.com OR site:https://www.w3schools.com"
    
    links = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=num_results)
        for r in results:
            title = r.get("title", "No Title")
            url = r.get("href") or r.get("url", "")
            if url:
                links.append((title, url))
    return links

if __name__ == "__main__":
    user_input = input("Enter a skill or role you want to learn: ")
    learning_links = get_learning_links(user_input)
    response = requests.get(f"https://www.w3schools.com/whatis/whatis_{user_input}.asp")
    soup = BeautifulSoup(response.text, 'html.parser')
    t = soup.find("title").text
    print(t)
    
    print("\nTop Learning Resources:\n")
    for i, (title, url) in enumerate(learning_links, 1):
        print(f"{i}. {title}\n   {url}\n")
