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
    
    return f"""{skill_title} Fundamentals
Learn the basic concepts and syntax of {skill}.
Time: 3-5 days
Link: https://www.w3schools.com/{skill.lower()}/

Core Features
Understand the main features and capabilities of {skill}.
Time: 4-6 days  
Link: https://www.geeksforgeeks.org/{skill.lower()}/

Practical Projects
Build small projects to apply your {skill} knowledge.
Time: 7-10 days
Link: https://github.com/topics/{skill.lower()}

Advanced Concepts
Explore more advanced topics and best practices in {skill}.
Time: 5-7 days
Link: https://developer.mozilla.org/

Real-world Applications
Work on larger projects and understand industry practices.
Time: 2-3 weeks
Link: https://www.freecodecamp.org/"""

def generate_roadmap(skill, max_attempts=3):
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
                return _fix_roadmap_formatting(roadmap)
            else:
                print(f"   Attempt {attempt + 1} produced invalid output, trying again...")
                
        except Exception as e:
            print(f"   Attempt {attempt + 1} failed: {e}")
            continue
    
    # If all attempts fail, return a basic template
    print("   All attempts failed, generating fallback roadmap...")
    return _generate_fallback_roadmap(skill)

class RoadmapEvaluator:
    """Comprehensive Roadmap Evaluation System"""
    
    def __init__(self, semantic_model):
        self.semantic_model = semantic_model
    
    def extract_steps(self, text):
        """Extract steps from roadmap text"""
        steps = []
        current_step = {}

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            if not line.startswith(('Time:', 'Link:')):
                # New step begins
                if current_step:
                    steps.append(current_step)
                current_step = {'title': line}
            elif line.startswith('Time:'):
                current_step['time'] = line
            elif line.startswith('Link:'):
                current_step['link'] = line
            elif line and 'title' in current_step and 'description' not in current_step:
                current_step['description'] = line

        if current_step:
            steps.append(current_step)

        return steps
    
    def structure_score(self, output):
        """Evaluate the structural quality of the roadmap"""
        lines = [line.strip() for line in output.split('\n') if line.strip()]

        step_count = 0
        i = 0
        while i < len(lines):
            # Assume any line not starting with Time or Link is a new step title
            if not lines[i].startswith(('Time:', 'Link:')):
                step_count += 1
            i += 1

        has_descriptions = len([line for line in lines if line and not line.startswith(('Time:', 'Link:'))])
        has_time_estimates = len([line for line in lines if line.startswith('Time:')])
        has_links = len([line for line in lines if line.startswith('Link:')])

        steps = self.extract_steps(output)
        complete_steps = sum(1 for step in steps if all(key in step for key in ['title', 'description', 'time', 'link']))

        scores = {
            'step_count': step_count,
            'has_descriptions': has_descriptions,
            'has_time_estimates': has_time_estimates,
            'has_links': has_links,
            'proper_formatting': True,  # Placeholder; optionally check spacing or line order
            'completeness_ratio': complete_steps / len(steps) if steps else 0,
            'total_steps': len(steps)
        }

        return scores
    
    def semantic_similarity_score(self, model_output, reference):
        """Calculate semantic similarity between model output and reference"""
        if not reference:
            return 0.0
        
        # Extract steps without relying on 'Step X:' prefix
        model_steps = self.extract_steps(model_output)
        ref_steps = self.extract_steps(reference)
        
        if not model_steps or not ref_steps:
            return 0.0
        
        # Use descriptions or fallback to titles
        model_descriptions = [step.get('description', step.get('title', '')) for step in model_steps]
        ref_descriptions = [step.get('description', step.get('title', '')) for step in ref_steps]
        
        model_descriptions = [desc for desc in model_descriptions if desc.strip()]
        ref_descriptions = [desc for desc in ref_descriptions if desc.strip()]
        
        if not model_descriptions or not ref_descriptions:
            return 0.0
        
        model_embeddings = self.semantic_model.encode(model_descriptions)
        ref_embeddings = self.semantic_model.encode(ref_descriptions)
        
        similarity_matrix = cosine_similarity(model_embeddings, ref_embeddings)
        max_similarities = np.max(similarity_matrix, axis=1)
        
        return np.mean(max_similarities)
    
    def content_coverage_score(self, model_output, reference):
        """Evaluate how well the model covers key technical terms"""
        if not reference:
            return 0.0
        
        # Extract technical terms (capitalized words, common tech terms)
        tech_pattern = r'\b(?:[A-Z][a-z]*|HTML|CSS|JavaScript|Python|API|SQL|Git|JSON|XML|HTTP|HTTPS|REST|OOP|MVC|CRUD|JWT|OAuth|Docker|AWS|React|Angular|Vue|Node|Express|Django|Flask|Spring|Laravel|PHP|Ruby|Java|C\+\+|C#|Go|Rust|Swift|Kotlin|Dart|TypeScript|MongoDB|PostgreSQL|MySQL|Redis|GraphQL|Webpack|Babel|ESLint|Jest|Cypress|Selenium|Jenkins|Travis|GitHub|GitLab|Bitbucket|VSCode|IntelliJ|Eclipse|Xcode|Android|iOS|Flutter|React Native|Ionic|PhoneGap|Cordova|Electron|Bootstrap|Tailwind)\b'
        
        model_terms = set(re.findall(tech_pattern, model_output))
        ref_terms = set(re.findall(tech_pattern, reference))
        
        if not ref_terms:
            return 0.0
        
        # Calculate coverage ratio
        covered_terms = model_terms.intersection(ref_terms)
        coverage_ratio = len(covered_terms) / len(ref_terms)
        
        return coverage_ratio
    
    def evaluate_roadmap(self, skill, model_output):
        """Comprehensive evaluation of the generated roadmap"""
        reference = load_reference_from_master_file(skill)
        
        # Structure evaluation
        structure_scores = self.structure_score(model_output)
        
        # Semantic similarity (if reference exists)
        semantic_score = self.semantic_similarity_score(model_output, reference) if reference else 0.0
        
        # Content coverage (if reference exists)
        coverage_score = self.content_coverage_score(model_output, reference) if reference else 0.0
        
        evaluation = {
            'structure': structure_scores,
            'semantic_similarity': semantic_score,
            'content_coverage': coverage_score,
            'has_reference': reference is not None
        }
        
        return evaluation

def generate_single_roadmap(skill, show_evaluation=True):
    """Generate roadmap for a single skill (useful for scripting)"""
    global semantic_model
    
    # Load models if not already loaded
    if model is None:
        load_models()
    
    evaluator = RoadmapEvaluator(semantic_model)
    
    print(f"🔄 Generating roadmap for: {skill}")
    roadmap = generate_roadmap(skill)
    
    print(f"\n📚 Learning Roadmap for: {skill.title()}")
    print("="*60)
    print(roadmap)
    
    if show_evaluation:
        evaluation = evaluator.evaluate_roadmap(skill, roadmap)
        print(f"\n📊 Evaluation Summary:")
        print(f"Steps: {evaluation['structure']['total_steps']}")
        if evaluation['has_reference']:
            print(f"Similarity: {evaluation['semantic_similarity']:.2%}")
    
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
    test_roadmap = generate_single_roadmap("Python")
    print("Test completed successfully!")