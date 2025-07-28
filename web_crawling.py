import requests
import re
import time
from urllib.parse import urljoin, quote
from typing import Dict, List, Optional, Tuple
import json
from youtubesearchpython import VideosSearch

class LinkFinder:
    """Web scraping utility to find educational links from GeeksforGeeks, W3Schools, and YouTube"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Curated YouTube educational channels with known good videos
        self.youtube_playlists = {
            "python": "https://www.youtube.com/watch?v=_uQrJ0TkZlc",  # Programming with Mosh
            "javascript": "https://www.youtube.com/watch?v=PkZNo7MFNFg",  # freeCodeCamp
            "html": "https://www.youtube.com/watch?v=UB1O30fR-EE",  # HTML Crash Course
            "css": "https://www.youtube.com/watch?v=yfoY53QXEnI",  # CSS Crash Course
            "react": "https://www.youtube.com/watch?v=w7ejDZ8SWv8",  # React Crash Course
            "java": "https://www.youtube.com/watch?v=eIrMbAQSU34",  # Java Programming
            "node": "https://www.youtube.com/watch?v=TlB_eWDSMt4",  # Node.js Crash Course
            "php": "https://www.youtube.com/watch?v=OK_JCtrrv-c",  # PHP for Beginners
            "sql": "https://www.youtube.com/watch?v=HXV3zeQKqGY",  # SQL Tutorial
            "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",  # Git and GitHub
            "django": "https://www.youtube.com/watch?v=F5mRW0jo-U4",  # Django Crash Course
            "flask": "https://www.youtube.com/watch?v=Z1RJmh_OqeA",  # Flask Tutorial
            "angular": "https://www.youtube.com/watch?v=3qBXWUpoPHo",  # Angular Crash Course
            "vue": "https://www.youtube.com/watch?v=qZXt1Aom3Cs",  # Vue.js Crash Course
            "bootstrap": "https://www.youtube.com/watch?v=4sosXZsdy-s",  # Bootstrap Tutorial
            "jquery": "https://www.youtube.com/watch?v=hMxGhHNOkCU",  # jQuery Crash Course
            "mongodb": "https://www.youtube.com/watch?v=ExcRbA7fy_A",  # MongoDB Crash Course
            "express": "https://www.youtube.com/watch?v=L72fhGm1tfE",  # Express.js Crash Course
            "typescript": "https://www.youtube.com/watch?v=BwuLxPH8IDs",  # TypeScript Crash Course
            "docker": "https://www.youtube.com/watch?v=fqMOX6JJhGo",  # Docker Tutorial
            "aws": "https://www.youtube.com/watch?v=3hLmDS179YE",  # AWS Tutorial
            "machine learning": "https://www.youtube.com/watch?v=ukzFI9rgwfU",  # Machine Learning Course
            "data science": "https://www.youtube.com/watch?v=ua-CiDNNj30",  # Data Science Course
            "web development": "https://www.youtube.com/watch?v=UB1O30fR-EE",  # Web Development Crash Course
        }
        
        # Common programming topics mapping for better search
        self.topic_mappings = {
            "javascript": ["js", "javascript", "ecmascript"],
            "python": ["python", "py"],
            "html": ["html", "html5"],
            "css": ["css", "css3", "styling"],
            "java": ["java"],
            "react": ["react", "reactjs", "react.js"],
            "node": ["node", "nodejs", "node.js"],
            "angular": ["angular", "angularjs"],
            "vue": ["vue", "vuejs", "vue.js"],
            "php": ["php"],
            "sql": ["sql", "mysql", "database"],
            "mongodb": ["mongodb", "mongo", "nosql"],
            "express": ["express", "expressjs"],
            "django": ["django"],
            "flask": ["flask"],
            "bootstrap": ["bootstrap"],
            "jquery": ["jquery"],
            "typescript": ["typescript", "ts"],
            "git": ["git", "github", "version control"],
            "docker": ["docker", "containerization"],
            "api": ["api", "rest api", "restful"],
            "json": ["json"],
            "ajax": ["ajax", "asynchronous"]
        }
    
    def normalize_topic(self, topic: str) -> List[str]:
        """Normalize topic name and return search variations"""
        topic_lower = topic.lower().strip()
        
        # Remove common prefixes/suffixes
        topic_clean = re.sub(r'\b(learn|learning|tutorial|basics|fundamentals|introduction|intro to|getting started with)\b', '', topic_lower).strip()
        topic_clean = re.sub(r'\s+', ' ', topic_clean)
        
        # Get mapped variations or return original
        if topic_clean in self.topic_mappings:
            return self.topic_mappings[topic_clean]
        
        return [topic_clean, topic_lower]
    
    def search_geeksforgeeks(self, topic: str) -> Optional[str]:
        """Search for GeeksforGeeks tutorial link using direct URL construction"""
        variations = self.normalize_topic(topic)
        
        for variation in variations:
            try:
                # Try direct URL construction first
                url_safe_topic = variation.replace(' ', '-').replace('.', '').lower()
                direct_urls = [
                    f"https://www.geeksforgeeks.org/{url_safe_topic}/",
                    f"https://www.geeksforgeeks.org/{url_safe_topic}-tutorial/",
                    f"https://www.geeksforgeeks.org/{url_safe_topic}-introduction/",
                    f"https://www.geeksforgeeks.org/introduction-to-{url_safe_topic}/",
                    f"https://www.geeksforgeeks.org/{url_safe_topic}-basics/"
                ]
                
                for url in direct_urls:
                    try:
                        response = self.session.head(url, timeout=5)
                        if response.status_code == 200:
                            return url
                    except:
                        continue
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error searching GeeksforGeeks for {variation}: {e}")
                continue
        
        # Fallback to main topic page
        topic_clean = variations[0].replace(' ', '-').lower()
        return f"https://www.geeksforgeeks.org/{topic_clean}/"
    
    def search_w3schools(self, topic: str) -> Optional[str]:
        """Search for W3Schools tutorial link using direct URL construction"""
        variations = self.normalize_topic(topic)
        
        for variation in variations:
            try:
                # Try direct URL construction
                url_safe_topic = variation.replace(' ', '').replace('.', '').lower()
                direct_urls = [
                    f"https://www.w3schools.com/{url_safe_topic}/",
                    f"https://www.w3schools.com/{url_safe_topic}/default.asp",
                    f"https://www.w3schools.com/{url_safe_topic}/intro.asp",
                    f"https://www.w3schools.com/{url_safe_topic}_intro.asp"
                ]
                
                for url in direct_urls:
                    try:
                        response = self.session.head(url, timeout=5)
                        if response.status_code == 200:
                            return url
                    except:
                        continue
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error searching W3Schools for {variation}: {e}")
                continue
        
        # Fallback to main topic page
        topic_clean = variations[0].replace(' ', '').lower()
        return f"https://www.w3schools.com/{topic_clean}/"
    
    def get_curated_youtube_link(self, topic: str) -> str:
    
        topic_lower = topic.lower().strip()

        # Direct or partial match from curated list
        if topic_lower in self.youtube_playlists:
            return self.youtube_playlists[topic_lower]

        for key, url in self.youtube_playlists.items():
            if key in topic_lower or any(word in topic_lower for word in key.split()):
                return url

        variations = self.normalize_topic(topic)
        for variation in variations:
            if variation in self.youtube_playlists:
                return self.youtube_playlists[variation]

        # Fallback: dynamic YouTube search
        try:
            search = VideosSearch(topic + " tutorial", limit=1)
            result = search.result().get("result", [])
            if result:
                return result[0]["link"]
        except Exception as e:
            print(f"Dynamic YouTube search failed for '{topic}': {e}")

        # Final fallback
        return "https://www.youtube.com/results?search_query=" + quote(topic + " tutorial")
    
    def get_educational_links(self, topic: str, prefer_w3schools: bool = True, include_youtube: bool = True) -> Dict[str, str]:
        """Get both GeeksforGeeks/W3Schools and optionally YouTube links for a topic"""
        results = {}
        
        print(f"🔍 Searching for links: {topic}")
        
        # Get tutorial link (prefer W3Schools or GeeksforGeeks based on topic)
        if prefer_w3schools or any(web_topic in topic.lower() for web_topic in ['html', 'css', 'javascript', 'bootstrap', 'jquery']):
            tutorial_link = self.search_w3schools(topic)
            if not tutorial_link or "w3schools.com" not in tutorial_link:
                tutorial_link = self.search_geeksforgeeks(topic)
        else:
            tutorial_link = self.search_geeksforgeeks(topic)
            if not tutorial_link or "geeksforgeeks.org" not in tutorial_link:
                tutorial_link = self.search_w3schools(topic)
        
        results['tutorial'] = tutorial_link
        
        # Get YouTube link only if requested
        if include_youtube:
            youtube_link = self.get_curated_youtube_link(topic)
            results['youtube'] = youtube_link
        
        return results
    
    def validate_links(self, links: Dict[str, str]) -> Dict[str, bool]:
        """Validate if the found links are accessible"""
        validation = {}
        
        for link_type, url in links.items():
            try:
                response = self.session.head(url, timeout=5)
                validation[link_type] = response.status_code == 200
            except:
                validation[link_type] = False
            
            time.sleep(0.3)  # Rate limiting
        
        return validation

def find_links_for_topic(topic: str, prefer_w3schools: bool = True, include_youtube: bool = True) -> Tuple[str, Optional[str]]:
    """
    Convenience function to find tutorial and optionally YouTube links for a topic
    
    Args:
        topic: The programming topic to search for
        prefer_w3schools: Whether to prefer W3Schools over GeeksforGeeks
        include_youtube: Whether to include YouTube link
    
    Returns:
        Tuple of (tutorial_link, youtube_link) where youtube_link is None if not requested
    """
    finder = LinkFinder()
    links = finder.get_educational_links(topic, prefer_w3schools, include_youtube)
    
    tutorial_link = links.get('tutorial', f"https://www.google.com/search?q={quote(topic)}+tutorial")
    youtube_link = links.get('youtube') if include_youtube else None
    
    return tutorial_link, youtube_link

def batch_find_links(topics: List[str], prefer_w3schools: bool = True, single_youtube_for_main_topic: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Find links for multiple topics in batch with rate limiting control
    
    Args:
        topics: List of topics to search for
        prefer_w3schools: Whether to prefer W3Schools over GeeksforGeeks
        single_youtube_for_main_topic: If True, only get YouTube link for first topic
    
    Returns:
        Dictionary mapping topic to {tutorial: url, youtube: url (optional)}
    """
    finder = LinkFinder()
    results = {}
    
    for i, topic in enumerate(topics):
        try:
            # Only include YouTube for first topic if single_youtube_for_main_topic is True
            include_youtube = not single_youtube_for_main_topic or i == 0
            
            links = finder.get_educational_links(topic, prefer_w3schools, include_youtube)
            results[topic] = links
            
            # Rate limiting - longer delay for first request to get YouTube
            if i == 0 and single_youtube_for_main_topic:
                time.sleep(2)  # Longer delay for YouTube request
            else:
                time.sleep(0.5)  # Shorter delay for tutorial-only requests
                
        except Exception as e:
            print(f"Error processing {topic}: {e}")
            fallback_links = {
                'tutorial': f"https://www.google.com/search?q={quote(topic)}+tutorial"
            }
            if not single_youtube_for_main_topic or i == 0:
                fallback_links['youtube'] = f"https://www.youtube.com/results?search_query={quote(topic)}+tutorial"
            
            results[topic] = fallback_links
    
    return results

def get_single_youtube_for_skill(skill: str) -> str:
    """
    Get a single curated YouTube link for the main skill
    This avoids rate limiting issues by not scraping
    
    Args:
        skill: The main skill/technology
    
    Returns:
        Curated YouTube URL for the skill
    """
    finder = LinkFinder()
    return finder.get_curated_youtube_link(skill)

# Example usage and testing
if __name__ == "__main__":
    # Test the improved link finder
    test_topics = ["Python", "JavaScript", "HTML", "React", "Node.js"]
    
    print("=== Testing Single YouTube Strategy ===")
    
    # Test 1: Get single YouTube for main skill
    main_skill = "Python"
    youtube_link = get_single_youtube_for_skill(main_skill)
    print(f"Single YouTube for {main_skill}: {youtube_link}")
    
    # Test 2: Batch processing with single YouTube
    print(f"\n=== Batch Processing ({len(test_topics)} topics) ===")
    batch_results = batch_find_links(test_topics, single_youtube_for_main_topic=True)
    
    for topic, links in batch_results.items():
        print(f"\n{topic}:")
        print(f"  Tutorial: {links['tutorial']}")
        if 'youtube' in links:
            print(f"  YouTube: {links['youtube']}")
        else:
            print(f"  YouTube: None (rate limit strategy)")