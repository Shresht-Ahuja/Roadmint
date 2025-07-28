import fitz  # PyMuPDF
from web_crawling import batch_find_links, find_links_for_topic

def extract_subtopics_from_pdf(pdf_path):
    """Extract subtopics from PDF file"""
    doc = fitz.open(pdf_path)
    subtopics = []

    for page in doc:
        text = page.get_text("text")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # Check if there are at least 2 lines
        if len(lines) >= 2:
            subtopic = lines[1]  # Second line is the topic name
            subtopics.append(subtopic)

    return list(set(subtopics))  # remove duplicates

def generate_enhanced_roadmap_from_pdf(pdf_path, generate_fn=None, use_web_links=True):
    """
    Generate roadmap from PDF with enhanced web link integration
    
    Args:
        pdf_path: Path to the PDF file
        generate_fn: Optional generation function (for backward compatibility)
        use_web_links: Whether to fetch real web links
    
    Returns:
        Enhanced roadmap with real links
    """
    subtopics = extract_subtopics_from_pdf(pdf_path)
    
    if not subtopics:
        print("❌ No subtopics found in the PDF.")
        return []

    unique_subtopics = list(set(subtopics))
    print(f"📋 Found {len(unique_subtopics)} unique subtopics")
    
    if use_web_links:
        print("🔍 Fetching real web links for all topics...")
        # Batch fetch links for all topics at once
        topic_links = batch_find_links(unique_subtopics, prefer_w3schools=True)
        
        # Generate enhanced roadmap
        roadmap_blocks = []
        
        for topic in unique_subtopics:
            topic_clean = topic.strip()
            
            # Get the links for this topic
            if topic_clean in topic_links:
                tutorial_link = topic_links[topic_clean]['tutorial']
                youtube_link = topic_links[topic_clean]['youtube']
            else:
                # Fallback if batch search failed for this topic
                tutorial_link, youtube_link = find_links_for_topic(topic_clean)
            
            # Generate description based on topic
            description = generate_topic_description(topic_clean)
            
            # Estimate time based on topic complexity
            time_estimate = estimate_learning_time(topic_clean)
            
            # Create structured block
            roadmap_block = f"""{topic_clean}
{description}
Time: {time_estimate}
Link: {tutorial_link}
YouTube: {youtube_link}"""
            
            roadmap_blocks.append(roadmap_block)
        
        return '\n\n'.join(roadmap_blocks)
    
    else:
        # Original method with generate_fn (backward compatibility)
        if generate_fn is None:
            print("❌ No generation function provided and web links disabled.")
            return ""
        
        prompt = f"""You are an AI assistant tasked with generating structured learning roadmap steps.

For each subtopic below, generate a roadmap step in **exactly** the following format:

<Subtopic Name>  
<One-line description of what the learner will gain from learning this>  
Time: <estimated days to learn this topic>  
Link: <a relevant W3Schools or GeeksforGeeks link>  
YouTube: <a relevant YouTube video link for this topic>

Here are the subtopics (do not skip any, and remove duplicates automatically):

{chr(10).join(f"- {topic}" for topic in unique_subtopics)}

⚠️ Important:
- Stick to the exact format and avoid extra commentary.
- Each subtopic should be followed by a structured block exactly as shown.
- Prefer GeeksforGeeks or W3Schools links for the 'Link' section.
- Ensure YouTube links are relevant to the subtopic and educational.
"""
        return generate_fn(prompt)

def generate_topic_description(topic):
    """Generate a one-line description for a topic"""
    topic_lower = topic.lower()
    
    # Common topic descriptions
    descriptions = {
        'html': "Learn how to structure web pages using fundamental HTML tags and elements.",
        'css': "Understand how to style web content using CSS properties, selectors, and layouts.",
        'javascript': "Learn how to add interactivity and behavior using core JavaScript features.",
        'python': "Master Python syntax, data structures, and programming fundamentals.",
        'java': "Understand object-oriented programming concepts and Java language features.",
        'react': "Learn to build interactive user interfaces using React components and hooks.",
        'node': "Understand server-side JavaScript development with Node.js runtime.",
        'sql': "Master database querying and management using Structured Query Language.",
        'git': "Learn version control, collaboration, and code management with Git.",
        'api': "Understand how to design, build, and consume RESTful APIs.",
        'database': "Learn database design, normalization, and management concepts.",
        'bootstrap': "Master responsive web design using Bootstrap framework components.",
        'jquery': "Learn DOM manipulation and event handling with jQuery library.",
        'php': "Understand server-side scripting and web development with PHP.",
        'angular': "Learn to build dynamic web applications using Angular framework.",
        'vue': "Master progressive web app development with Vue.js framework.",
        'mongodb': "Understand NoSQL database design and operations with MongoDB.",
        'express': "Learn to build web servers and APIs using Express.js framework.",
        'django': "Master web development using Django's high-level Python framework.",
        'flask': "Learn lightweight web application development with Flask framework.",
        'docker': "Understand containerization and deployment using Docker containers.",
        'aws': "Learn cloud computing services and deployment on Amazon Web Services.",
        'typescript': "Master type-safe JavaScript development with TypeScript language.",
        'graphql': "Understand modern API development using GraphQL query language.",
        'redux': "Learn state management for React applications using Redux.",
        'webpack': "Master module bundling and build optimization with Webpack.",
        'sass': "Learn advanced CSS preprocessing with Sass/SCSS features.",
        'linux': "Understand command-line operations and system administration in Linux.",
        'testing': "Learn software testing methodologies and automated testing frameworks.",
        'devops': "Understand continuous integration, deployment, and infrastructure management."
    }
    
    # Check for exact matches first
    for key, desc in descriptions.items():
        if key in topic_lower:
            return desc
    
    # Generate generic description
    return f"Learn the fundamentals and practical applications of {topic}."

def estimate_learning_time(topic):
    """Estimate learning time based on topic complexity"""
    topic_lower = topic.lower()
    
    # Time estimates based on complexity
    quick_topics = ['html', 'css', 'bootstrap', 'jquery', 'sass', 'json', 'xml']
    medium_topics = ['javascript', 'python', 'php', 'sql', 'git', 'linux', 'testing']
    complex_topics = ['java', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'express', 'mongodb', 'docker', 'aws', 'typescript', 'graphql', 'redux', 'webpack', 'devops']
    
    for topic_key in quick_topics:
        if topic_key in topic_lower:
            return "2-3 days"
    
    for topic_key in medium_topics:
        if topic_key in topic_lower:
            return "4-6 days"
    
    for topic_key in complex_topics:
        if topic_key in topic_lower:
            return "1-2 weeks"
    
    # Default estimate
    return "3-5 days"

def generate_roadmap_from_pdf(pdf_path, generate_fn=None):
    """
    Legacy function for backward compatibility
    Now calls the enhanced version with web links enabled
    """
    return generate_enhanced_roadmap_from_pdf(pdf_path, generate_fn, use_web_links=True)

# Example usage
if __name__ == "__main__":
    # Test the enhanced PDF processing
    pdf_path = "sample_topics.pdf"  # Replace with your PDF path
    
    try:
        print("🔄 Processing PDF with enhanced link finding...")
        roadmap = generate_enhanced_roadmap_from_pdf(pdf_path, use_web_links=True)
        
        print("\n📚 Generated Roadmap:")
        print("="*60)
        print(roadmap)
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")