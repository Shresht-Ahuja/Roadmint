import os
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv
import os
from huggingface_hub import login
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json

# Import the web link finder
from web_crawling import find_links_for_topic, batch_find_links

# Global variables for models (will be loaded once)
model = None
tokenizer = None
semantic_model = None
device = None

def load_models():
    """Load and initialize all required models"""
    global model, tokenizer, semantic_model, device
    
    # Login to Hugging Face
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    login(token)
    
    # Load trained model and tokenizer
    model_id = "./gemma-roadmap-lora-final"
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    # Load semantic similarity model
    print("Loading semantic similarity model...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print("✅ All models loaded successfully!")

def load_reference_from_master_file(skill):
    """Load reference roadmap from file if available"""
    filename = "reference.txt"
    if not os.path.exists(filename):
        return None

    skill_section = f"### Skill: {skill.strip().title()}"
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the section that starts with the skill
    start = content.find(skill_section)
    if start == -1:
        return None

    # Look for the next skill marker
    end = content.find("### Skill:", start + 1)
    section = content[start:end].strip() if end != -1 else content[start:].strip()
    return "\n".join(line.strip() for line in section.splitlines()[1:]).strip()

def generate_instruction_prompt(skill):
    """Generate the instruction prompt for the model"""
    return f"""### Input:
I want to learn Web Development

### Output:
HTML Basics  
Learn how to structure web pages using fundamental HTML tags and elements.
Time: 2 days  
Link: https://www.geeksforgeeks.org/html-tutorial/

CSS Fundamentals  
Understand how to style web content using CSS properties, selectors, and layouts.
Time: 3 days  
Link: https://www.w3schools.com/css/

JavaScript Basics  
Learn how to add interactivity and behavior using core JavaScript features.
Time: 4 days  
Link: https://www.geeksforgeeks.org/javascript-tutorial/

DOM Manipulation  
Explore how JavaScript interacts with HTML/CSS via the Document Object Model.
Time: 2 days  
Link: https://www.w3schools.com/js/js_htmldom.asp

Responsive Design  
Learn to make web pages responsive using media queries and flexible grids.
Time: 3 days  
Link: https://www.geeksforgeeks.org/responsive-web-design/

Git & GitHub  
Understand version control using Git and how to host code using GitHub.
Time: 2 days  
Link: https://www.geeksforgeeks.org/git/

### Input:
I want to learn Python

### Output:
Python Basics  
Understand syntax, variables, loops, and conditionals.
Time: 2 days  
Link: https://www.w3schools.com/python/

Data Structures  
Learn about lists, tuples, sets, and dictionaries.
Time: 2 days  
Link: https://www.geeksforgeeks.org/python-data-structures/

Functions & Modules  
Get familiar with defining functions, scope, and using built-in modules.
Time: 2 days  
Link: https://www.programiz.com/python-programming/function

Object-Oriented Programming  
Understand classes, objects, inheritance, and encapsulation.
Time: 3 days  
Link: https://realpython.com/python3-object-oriented-programming/

File I/O and Exceptions  
Learn how to read/write files and handle errors.
Time: 2 days  
Link: https://www.geeksforgeeks.org/file-handling-python/

### Input:
I want to learn {skill}

### Output:"""

def extract_roadmap_only(generated_text, prompt):
    """Extract only the roadmap for the requested skill"""
    # Remove the prompt part
    response = generated_text[len(prompt):].strip()
    
    # Split by lines and process
    lines = response.split('\n')
    roadmap_lines = []
    found_first_step = False
    
    for line in lines:
        line = line.strip()
        
        # Stop conditions
        if line.startswith("### Input:") or (line.startswith("I want to learn") and found_first_step):
            break
        
        # Skip empty lines at the beginning
        if not roadmap_lines and not line:
            continue
        
        # Detect if we've found the first step
        if re.match(r'Step\s+\d+:', line):
            found_first_step = True
        
        # Add valid lines
        if line:
            roadmap_lines.append(line)
        
        # Stop if we have enough content and hit an empty line after finding steps
        if found_first_step and not line and len(roadmap_lines) > 10:
            break
    
    # Join and clean up
    roadmap = '\n'.join(roadmap_lines)
    
    # Remove any content that looks like it's from another skill
    roadmap_lines_clean = []
    
    for line in roadmap.split('\n'):
        line = line.strip()
        if not line:
            roadmap_lines_clean.append(line)
            continue
            
        # If we encounter what looks like a new skill request, stop
        if any(phrase in line.lower() for phrase in ['i want to learn', 'let me learn', 'teach me']):
            break
            
        roadmap_lines_clean.append(line)
    
    return '\n'.join(roadmap_lines_clean).strip()

def enhance_roadmap_with_real_links(roadmap_text, skill):
    """Enhance generated roadmap with real web links"""
    print(f"🔗 Enhancing roadmap with real links for: {skill}")
    
    # Parse the roadmap to extract topics
    lines = roadmap_text.split('\n')
    enhanced_lines = []
    topics_for_batch_search = []
    current_topic = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            enhanced_lines.append('')
            i += 1
            continue
        
        # Check if this is a topic title (not starting with Time: or Link:)
        if not line.startswith(('Time:', 'Link:', 'YouTube:')):
            current_topic = line
            topics_for_batch_search.append(line)
            enhanced_lines.append(line)
        elif line.startswith('Time:'):
            enhanced_lines.append(line)
        elif line.startswith('Link:'):
            # Skip the old link, we'll replace it
            pass
        elif line.startswith('YouTube:'):
            # Skip the old YouTube link, we'll replace it
            pass
        else:
            enhanced_lines.append(line)
        
        i += 1
    
    # Batch search for links
    if topics_for_batch_search:
        print(f"🔍 Searching for links for {len(topics_for_batch_search)} topics...")
        topic_links = batch_find_links(topics_for_batch_search)
        
        # Rebuild the roadmap with real links
        final_lines = []
        current_topic_idx = 0
        
        i = 0
        while i < len(enhanced_lines):
            line = enhanced_lines[i].strip()
            
            if not line:
                final_lines.append('')
                i += 1
                continue
            
            # If this is a topic title
            if not line.startswith(('Time:', 'Link:', 'YouTube:')) and current_topic_idx < len(topics_for_batch_search):
                topic = topics_for_batch_search[current_topic_idx]
                final_lines.append(line)
                
                # Look for the description line
                if i + 1 < len(enhanced_lines) and enhanced_lines[i + 1].strip() and not enhanced_lines[i + 1].startswith(('Time:', 'Link:', 'YouTube:')):
                    final_lines.append(enhanced_lines[i + 1])
                    i += 1
                
                # Look for time estimate
                if i + 1 < len(enhanced_lines) and enhanced_lines[i + 1].startswith('Time:'):
                    final_lines.append(enhanced_lines[i + 1])
                    i += 1
                
                # Add the real links
                if topic in topic_links:
                    final_lines.append(f"Link: {topic_links[topic]['tutorial']}")
                    final_lines.append(f"YouTube: {topic_links[topic]['youtube']}")
                
                current_topic_idx += 1
            else:
                # For other lines (like descriptions, time estimates not yet processed)
                if not line.startswith(('Link:', 'YouTube:')):
                    final_lines.append(line)
            
            i += 1
        
        return '\n'.join(final_lines)
    
    return roadmap_text

def _generate_with_params(skill, max_tokens, temperature, do_sample):
    """Internal function to generate with specific parameters"""
    global model, tokenizer, device
    
    # Create the prompt
    prompt = generate_instruction_prompt(skill)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1200)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate
    generation_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'max_new_tokens': max_tokens,
        'temperature': temperature,
        'do_sample': do_sample,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'repetition_penalty': 1.1,
    }
    
    # Add sampling parameters only if do_sample is True
    if do_sample:
        generation_kwargs['top_p'] = 0.9
        generation_kwargs['top_k'] = 50
    
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract roadmap
    return extract_roadmap_only(generated_text, prompt)

def _is_valid_roadmap(roadmap, skill):
    """Validate if the generated roadmap is acceptable"""
    if not roadmap or len(roadmap.strip()) < 100:
        return False
    
    lines = [line.strip() for line in roadmap.split('\n') if line.strip()]
    
    # Check that we have some descriptions and links
    desc_lines = [line for line in lines if not line.startswith(('Time: ', 'Link: '))]
    link_lines = [line for line in lines if line.startswith('Link: ')]
    
    if len(desc_lines) < 2 or len(link_lines) < 2:
        return False
    
    # Check that content is relevant to the skill (basic keyword check)
    roadmap_lower = roadmap.lower()
    skill_lower = skill.lower()
    
    # Should contain the skill name or related terms
    if skill_lower not in roadmap_lower and len(skill_lower) > 3:
        return False
    
    return True

def _fix_roadmap_formatting(roadmap):
    """Clean and normalize roadmap formatting by removing 'Step X:' and ensuring consistent spacing."""
    lines = roadmap.split('\n')
    fixed_lines = []
    temp_block = []

    for line in lines:
        line = line.strip()

        if not line:
            # If empty line and we have a completed block, push it
            if temp_block:
                fixed_lines.append('\n'.join(temp_block))
                temp_block = []
            continue

        # Skip any line like "Step X:"
        if re.match(r'Step\s*\d+\s*:', line, re.IGNORECASE):
            continue

        temp_block.append(line)

    # Add the final block if any
    if temp_block:
        fixed_lines.append('\n'.join(temp_block))

    # Join all blocks with 2 newlines between them
    return '\n\n'.join(fixed_lines)

def _generate_fallback_roadmap(skill):
    """Generate a basic fallback roadmap if model fails"""
    skill_title = skill.title()
    
    # Get real links for the fallback roadmap
    print(f"🔗 Getting links for fallback roadmap: {skill}")
    tutorial_link, youtube_link = find_links_for_topic(skill)
    
    return f"""{skill_title} Fundamentals
Learn the basic concepts and syntax of {skill}.
Time: 3-5 days
Link: {tutorial_link}
YouTube: {youtube_link}

Core Features
Understand the main features and capabilities of {skill}.
Time: 4-6 days
Link: {tutorial_link}
YouTube: {youtube_link}

Practical Projects
Build small projects to apply your {skill} knowledge.
Time: 7-10 days
Link: {tutorial_link}
YouTube: {youtube_link}

Advanced Concepts
Explore more advanced topics and best practices in {skill}.
Time: 5-7 days
Link: {tutorial_link}
YouTube: {youtube_link}

Real-world Applications
Work on larger projects and understand industry practices.
Time: 2-3 weeks
Link: {tutorial_link}
YouTube: {youtube_link}"""

def generate_roadmap(skill, max_attempts=3, enhance_links=True):
    """Generate a learning roadmap for the given skill with fallback strategies"""
    global model, tokenizer
    
    # Check if models are loaded
    if model is None or tokenizer is None:
        raise Exception("Models not loaded. Please call load_models() first.")
    
    for attempt in range(max_attempts):
        try:
            print(f"   Attempt {attempt + 1}/{max_attempts}...")
            
            # Try different generation strategies
            if attempt == 0:
                # Conservative approach
                roadmap = _generate_with_params(skill, max_tokens=400, temperature=0.1, do_sample=False)
            elif attempt == 1:
                # Moderate sampling
                roadmap = _generate_with_params(skill, max_tokens=350, temperature=0.3, do_sample=True)
            else:
                # Greedy decoding as last resort
                roadmap = _generate_with_params(skill, max_tokens=600, temperature=0.0, do_sample=False)
            
            print("Generated roadmap:\n", roadmap)

            # Validate the output
            if _is_valid_roadmap(roadmap, skill):
                formatted_roadmap = _fix_roadmap_formatting(roadmap)
                
                # Enhance with real links if requested
                if enhance_links:
                    enhanced_roadmap = enhance_roadmap_with_real_links(formatted_roadmap, skill)
                    return enhanced_roadmap
                
                return formatted_roadmap
            else:
                print(f"   Attempt {attempt + 1} produced invalid output, trying again...")
                
        except Exception as e:
            print(f"   Attempt {attempt + 1} failed: {e}")
            continue
    
    # If all attempts fail, return a basic template with real links
    print("   All attempts failed, generating fallback roadmap...")
    return _generate_fallback_roadmap(skill)

def generate_roadmap_for_pdf_mode(skill_or_prompt, enhance_links=True):
    """
    Generate roadmap specifically for PDF mode with enhanced link finding
    This function can handle both individual skills and full prompts from PDF processing
    """
    if "You are an AI assistant" in skill_or_prompt:
        # This is a full prompt from PDF mode
        # Extract the topics from the prompt
        lines = skill_or_prompt.split('\n')
        topics = []
        for line in lines:
            if line.strip().startswith('- '):
                topic = line.strip()[2:]  # Remove '- '
                topics.append(topic)
        
        if not topics:
            return "❌ No topics found in the prompt."
        
        # Generate structured roadmap for each topic
        roadmap_parts = []
        
        if enhance_links:
            # Batch find links for all topics
            print(f"🔍 Finding links for {len(topics)} topics...")
            topic_links = batch_find_links(topics)
        
        for topic in topics:
            # Generate a simple structured block for each topic
            topic_clean = topic.strip()
            
            if enhance_links and topic_clean in topic_links:
                tutorial_link = topic_links[topic_clean]['tutorial']
                youtube_link = topic_links[topic_clean]['youtube']
            else:
                tutorial_link, youtube_link = find_links_for_topic(topic_clean)
            
            # Create structured roadmap block
            roadmap_block = f"""{topic_clean}
Learn the fundamentals and practical applications of {topic_clean.lower()}.
Time: 3-5 days
Link: {tutorial_link}
YouTube: {youtube_link}"""
            
            roadmap_parts.append(roadmap_block)
        
        return '\n\n'.join(roadmap_parts)
    
    else:
        # This is a single skill, use the regular generation method
        return generate_roadmap(skill_or_prompt, enhance_links=enhance_links)

class RoadmapEvaluator:
    def __init__(self, model_name_or_path='all-MiniLM-L6-v2', baseline_model_name=None):
        """Initialize with either model name or path"""
        if isinstance(model_name_or_path, SentenceTransformer):
            self.semantic_model = model_name_or_path
        else:
            self.semantic_model = SentenceTransformer(model_name_or_path)
            
        self.baseline_model = None
        self.baseline_tokenizer = None
        
        if baseline_model_name:
            self.load_baseline_model(baseline_model_name)
    
    def load_baseline_model(self, model_name: str):
        """Load a baseline model for comparison"""
        print(f"Loading baseline model: {model_name}")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.baseline_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto"
        )
        
        if self.baseline_tokenizer.pad_token is None:
            self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token
        
        print("✅ Baseline model loaded successfully!")
    
    def generate_baseline_roadmap(self, skill: str, instruction_prompt: str) -> str:
        """Generate roadmap using baseline model"""
        if not self.baseline_model or not self.baseline_tokenizer:
            raise ValueError("Baseline model not loaded. Call load_baseline_model() first.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        prompt = instruction_prompt.format(skill=skill)
        
        inputs = self.baseline_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1200)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        generation_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'max_new_tokens': 400,
            'temperature': 0.1,
            'do_sample': False,
            'pad_token_id': self.baseline_tokenizer.eos_token_id,
            'eos_token_id': self.baseline_tokenizer.eos_token_id,
            'repetition_penalty': 1.1,
        }
        
        with torch.no_grad():
            outputs = self.baseline_model.generate(**generation_kwargs)
        
        generated_text = self.baseline_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        return self._clean_generated_text(response)
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean generated text to extract only the roadmap"""
        lines = text.split('\n')
        roadmap_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("### Input:") or (line.startswith("I want to learn") and roadmap_lines):
                break
            if line:
                roadmap_lines.append(line)
        
        return '\n'.join(roadmap_lines).strip()
    
    def extract_roadmap_steps(self, text: str) -> List[Dict]:
        """Extract structured steps from roadmap text"""
        steps = []
        current_step = {}
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue
            
            # New step starts when we find a line that isn't a metadata line
            if not line.startswith(('Time:', 'Link:', 'YouTube:')):
                if current_step:  # Save previous step if exists
                    steps.append(current_step)
                current_step = {'title': line}

                # Check if next line is description
                if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(('Time:', 'Link:', 'YouTube:')):
                    current_step['description'] = lines[i + 1].strip()
                    i += 1  # Skip the description line
            elif line.startswith('Time:'):
                current_step['time'] = line.replace('Time:', '').strip()
            elif line.startswith('Link:'):
                current_step['link'] = line.replace('Link:', '').strip()

            i += 1

        # Add the last step if it exists
        if current_step:
            steps.append(current_step)

        return steps
    
    def calculate_structure_score(self, roadmap: str) -> Dict:
        """Evaluate the structural quality of the roadmap"""
        steps = self.extract_roadmap_steps(roadmap)

        if not steps:
            return {
                'total_steps': 0,
                'complete_steps': 0,
                'completeness_ratio': 0.0,
                'has_descriptions': 0,
                'has_time_estimates': 0,
                'has_links': 0,
                'avg_title_length': 0,
                'avg_description_length': 0
            }

        complete_steps = 0
        has_descriptions = 0
        has_time_estimates = 0
        has_links = 0
        title_lengths = []
        description_lengths = []

        for step in steps:
            # A step is complete if it has title, description, and time
            if all(key in step for key in ['title', 'description', 'time']):
                complete_steps += 1

            if 'description' in step:
                has_descriptions += 1
                description_lengths.append(len(step['description']))

            if 'time' in step:
                has_time_estimates += 1

            if 'link' in step:
                has_links += 1

            if 'title' in step:
                title_lengths.append(len(step['title']))

        return {
            'total_steps': len(steps),
            'complete_steps': complete_steps,
            'completeness_ratio': complete_steps / len(steps) if steps else 0.0,
            'has_descriptions': has_descriptions,
            'has_time_estimates': has_time_estimates,
            'has_links': has_links,
            'avg_title_length': np.mean(title_lengths) if title_lengths else 0,
            'avg_description_length': np.mean(description_lengths) if description_lengths else 0
        }
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two roadmaps"""
        steps1 = self.extract_roadmap_steps(text1)
        steps2 = self.extract_roadmap_steps(text2)
        
        if not steps1 or not steps2:
            return 0.0
        
        # Combine title and description for better semantic representation
        def get_step_text(step):
            text_parts = []
            if 'title' in step:
                text_parts.append(step['title'])
            if 'description' in step:
                text_parts.append(step['description'])
            return ' '.join(text_parts)
        
        texts1 = [get_step_text(step) for step in steps1 if get_step_text(step).strip()]
        texts2 = [get_step_text(step) for step in steps2 if get_step_text(step).strip()]
        
        if not texts1 or not texts2:
            return 0.0
        
        # Get embeddings
        embeddings1 = self.semantic_model.encode(texts1)
        embeddings2 = self.semantic_model.encode(texts2)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        
        # For each step in roadmap1, find best match in roadmap2
        max_similarities = np.max(similarity_matrix, axis=1)
        
        return float(np.mean(max_similarities))
    
    def calculate_content_coverage(self, model_output: str, reference: str) -> Dict:
        """Calculate content coverage metrics"""
        if not reference:
            return {'coverage_ratio': 0.0, 'unique_terms': 0, 'covered_terms': 0}
        
        # Extract technical terms and concepts
        tech_pattern = r'\b(?:[A-Z][a-z]*|HTML|CSS|JavaScript|Python|API|SQL|Git|JSON|XML|HTTP|HTTPS|REST|OOP|MVC|CRUD|JWT|OAuth|Docker|AWS|React|Angular|Vue|Node|Express|Django|Flask|Spring|Laravel|PHP|Ruby|Java|C\+\+|C#|Go|Rust|Swift|Kotlin|Dart|TypeScript|MongoDB|PostgreSQL|MySQL|Redis|GraphQL|Webpack|Babel|ESLint|Jest|Cypress|Selenium|Jenkins|Travis|GitHub|GitLab|Bitbucket|VSCode|IntelliJ|Eclipse|Xcode|Android|iOS|Flutter|React Native|Ionic|PhoneGap|Cordova|Electron|Bootstrap|Tailwind|NumPy|Pandas|Matplotlib|Scikit|TensorFlow|PyTorch|Keras|OpenCV|Flask|FastAPI|Jupyter|Anaconda|Pip|Conda|Virtual|Environment)\b'
        
        model_terms = set(re.findall(tech_pattern, model_output, re.IGNORECASE))
        ref_terms = set(re.findall(tech_pattern, reference, re.IGNORECASE))
        
        # Convert to lowercase for comparison
        model_terms = {term.lower() for term in model_terms}
        ref_terms = {term.lower() for term in ref_terms}
        
        if not ref_terms:
            return {'coverage_ratio': 0.0, 'unique_terms': len(model_terms), 'covered_terms': 0}
        
        covered_terms = model_terms.intersection(ref_terms)
        coverage_ratio = len(covered_terms) / len(ref_terms)
        
        return {
            'coverage_ratio': coverage_ratio,
            'unique_terms': len(model_terms),
            'covered_terms': len(covered_terms),
            'reference_terms': len(ref_terms)
        }
    
    def load_reference_roadmap(self, skill: str, reference_file: str = "reference.txt") -> Optional[str]:
        """Load reference roadmap from file"""
        if not os.path.exists(reference_file):
            return None
        
        skill_section = f"### Skill: {skill.strip().title()}"
        
        with open(reference_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        start = content.find(skill_section)
        if start == -1:
            return None
        
        end = content.find("### Skill:", start + 1)
        section = content[start:end].strip() if end != -1 else content[start:].strip()
        
        return "\n".join(line.strip() for line in section.splitlines()[1:]).strip()
    
    def evaluate_roadmap(self, 
                               skill: str, 
                               model_output: str, 
                               instruction_prompt: str = None,
                               reference_file: str = "reference.txt") -> Dict:
        """
        Perform comprehensive evaluation of a generated roadmap
        
        Args:
            skill: The skill being evaluated
            model_output: Generated roadmap from your model
            instruction_prompt: Prompt template for baseline comparison
            reference_file: Path to reference file
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        evaluation_results = {
            'skill': skill,
            'timestamp': datetime.now().isoformat(),
            'model_structure': self.calculate_structure_score(model_output),
        }
        
        # Load reference roadmap
        reference = self.load_reference_roadmap(skill, reference_file)
        evaluation_results['has_reference'] = reference is not None
        
        if reference:
            # Compare with reference
            evaluation_results['reference_similarity'] = self.calculate_semantic_similarity(
                model_output, reference
            )
            evaluation_results['content_coverage'] = self.calculate_content_coverage(
                model_output, reference
            )
            evaluation_results['reference_structure'] = self.calculate_structure_score(reference)
        
        # Generate baseline comparison if baseline model is available and prompt is provided
        if self.baseline_model and instruction_prompt:
            try:
                baseline_output = self.generate_baseline_roadmap(skill, instruction_prompt)
                evaluation_results['baseline_output'] = baseline_output
                evaluation_results['baseline_structure'] = self.calculate_structure_score(baseline_output)
                evaluation_results['baseline_similarity'] = self.calculate_semantic_similarity(
                    model_output, baseline_output
                )
                
                # If we have reference, compare baseline to reference too
                if reference:
                    evaluation_results['baseline_vs_reference'] = self.calculate_semantic_similarity(
                        baseline_output, reference
                    )
                    evaluation_results['baseline_content_coverage'] = self.calculate_content_coverage(
                        baseline_output, reference
                    )
                
            except Exception as e:
                print(f"Warning: Could not generate baseline comparison: {e}")
                evaluation_results['baseline_error'] = str(e)
        
        return evaluation_results
    
    def print_evaluation_summary(self, results: Dict):
        """Print a formatted summary of evaluation results"""
        print(f"\n📊 Evaluation Summary for: {results['skill']}")
        print("=" * 60)
        
        # Model structure
        struct = results['model_structure']
        print(f"🏗️ Model Structure:")
        print(f"  Steps: {struct['total_steps']}")
        print(f"  Complete steps: {struct['complete_steps']} ({struct['completeness_ratio']:.1%})")
        print(f"  Descriptions: {struct['has_descriptions']}")
        print(f"  Time estimates: {struct['has_time_estimates']}")
        print(f"  Links: {struct['has_links']}")

        
        # Reference comparison
        if results['has_reference']:
            print(f"\n📚 Reference Comparison:")
            print(f"  Similarity to reference: {results['reference_similarity']:.1%}")
            coverage = results['content_coverage']
            print(f"  Content coverage: {coverage['coverage_ratio']:.1%}")
            print(f"  Terms covered: {coverage['covered_terms']}/{coverage['reference_terms']}")
        
        # Baseline comparison
        if 'baseline_similarity' in results:
            print(f"\n🔄 Baseline Comparison:")
            print(f"  Similarity to baseline: {results['baseline_similarity']:.1%}")
            baseline_struct = results['baseline_structure']
            print(f"  Baseline steps: {baseline_struct['total_steps']}")
            
            if results['has_reference']:
                print(f"  Baseline vs reference: {results['baseline_vs_reference']:.1%}")
        
        print("=" * 60)

def generate_single_roadmap(skill, show_evaluation=True, enhance_links=True):
    """Generate roadmap for a single skill (useful for scripting)"""
    global semantic_model
    
    # Load models if not already loaded
    if model is None:
        load_models()
    
    evaluator = RoadmapEvaluator()  # Let it use default model
    # OR
    evaluator = RoadmapEvaluator('all-MiniLM-L6-v2')  # Explicit model name
    
    print(f"🔄 Generating roadmap for: {skill}")
    roadmap = generate_roadmap(skill, enhance_links=enhance_links)
    
    print(f"\n📚 Learning Roadmap for: {skill.title()}")
    print("="*60)
    print(roadmap)
    
    if show_evaluation:
        evaluation = evaluator.evaluate_roadmap(
            skill=skill,
            model_output=roadmap,
            instruction_prompt=generate_instruction_prompt(skill)
        )
        evaluator.print_evaluation_summary(evaluation)
    
    return roadmap

# Initialize semantic model for use in other modules
try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    semantic_model = None
    print("Warning: Could not load semantic model. Some features may be limited.")

if __name__ == "__main__":
    # Load models and run a test
    load_models()
    test_roadmap = generate_single_roadmap("Python", enhance_links=True)
    print("Test completed successfully!")