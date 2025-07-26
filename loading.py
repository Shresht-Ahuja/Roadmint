import os
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from huggingface_hub import login
load_dotenv()
token = os.getenv('HF_TOKEN')
login(token)

# ✅ Load trained model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


# ✅ Load semantic similarity model
print("Loading semantic similarity model...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ Load reference from file
def load_reference_from_master_file(skill):
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

# ✅ Build an instruction-tuned prompt with few-shot example
def generate_instruction_prompt(skill):
    return f"""### Input:
I want to learn Web Development

### Output:
Step 1: HTML Basics  
Learn how to structure web pages using fundamental HTML tags and elements.
Time: 2 days  
Link: https://www.geeksforgeeks.org/html-tutorial/

Step 2: CSS Fundamentals  
Understand how to style web content using CSS properties, selectors, and layouts.
Time: 3 days  
Link: https://www.w3schools.com/css/

Step 3: JavaScript Basics  
Learn how to add interactivity and behavior using core JavaScript features.
Time: 4 days  
Link: https://www.geeksforgeeks.org/javascript-tutorial/

Step 4: DOM Manipulation  
Explore how JavaScript interacts with HTML/CSS via the Document Object Model.
Time: 2 days  
Link: https://www.w3schools.com/js/js_htmldom.asp

Step 5: Responsive Design  
Learn to make web pages responsive using media queries and flexible grids.
Time: 3 days  
Link: https://www.geeksforgeeks.org/responsive-web-design/

Step 6: Git & GitHub  
Understand version control using Git and how to host code using GitHub.
Time: 2 days  
Link: https://www.geeksforgeeks.org/git/

### Input:
I want to learn Python

### Output:
Step 1: Python Basics  
Understand syntax, variables, loops, and conditionals.
Time: 2 days  
Link: https://www.w3schools.com/python/

Step 2: Data Structures  
Learn about lists, tuples, sets, and dictionaries.
Time: 2 days  
Link: https://www.geeksforgeeks.org/python-data-structures/

Step 3: Functions & Modules  
Get familiar with defining functions, scope, and using built-in modules.
Time: 2 days  
Link: https://www.programiz.com/python-programming/function

Step 4: Object-Oriented Programming  
Understand classes, objects, inheritance, and encapsulation.
Time: 3 days  
Link: https://realpython.com/python3-object-oriented-programming/

Step 5: File I/O and Exceptions  
Learn how to read/write files and handle errors.
Time: 2 days  
Link: https://www.geeksforgeeks.org/file-handling-python/

### Input:
I want to learn {skill}

### Output:"""

# ✅ Generate roadmap using the model
def generate_roadmap(skill, max_tokens=500, temperature=0.7):
    """Generate a learning roadmap for the given skill"""
    
    # Create the prompt using few-shot learning
    prompt = generate_instruction_prompt(skill)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    generated_roadmap = generated_text[len(prompt):].strip()
    
    # Clean up the output
    lines = generated_roadmap.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('###'):
            clean_lines.append(line)
    
    return '\n'.join(clean_lines)

# ✅ Comprehensive Roadmap Evaluation System
class RoadmapEvaluator:
    def __init__(self, semantic_model):
        self.semantic_model = semantic_model
    
    def extract_steps(self, text):
        """Extract structured steps from roadmap text"""
        steps = []
        current_step = {}
        
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('Step '):
                if current_step:
                    steps.append(current_step)
                current_step = {'title': line}
            elif line.startswith('Time: '):
                current_step['time'] = line
            elif line.startswith('Link: '):
                current_step['link'] = line
            elif line and 'title' in current_step and 'description' not in current_step:
                current_step['description'] = line
        
        if current_step:
            steps.append(current_step)
        
        return steps
    
    def structure_score(self, output):
        """Evaluate the structural quality of the roadmap"""
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        
        scores = {
            'step_count': len([line for line in lines if line.startswith('Step ')]),
            'has_descriptions': len([line for line in lines if line and not line.startswith(('Step ', 'Time: ', 'Link: '))]),
            'has_time_estimates': len([line for line in lines if line.startswith('Time: ')]),
            'has_links': len([line for line in lines if line.startswith('Link: ')]),
            'proper_formatting': True
        }
        
        # Check if format is consistent
        steps = self.extract_steps(output)
        complete_steps = sum(1 for step in steps if all(key in step for key in ['title', 'description', 'time', 'link']))
        
        scores['completeness_ratio'] = complete_steps / len(steps) if steps else 0
        scores['total_steps'] = len(steps)
        
        return scores
    
    def semantic_similarity_score(self, model_output, reference):
        """Calculate semantic similarity between model output and reference"""
        if not reference:
            return 0.0
        
        # Split into meaningful chunks (each step)
        model_steps = self.extract_steps(model_output)
        ref_steps = self.extract_steps(reference)
        
        if not model_steps or not ref_steps:
            return 0.0
        
        # Get embeddings for each step description
        model_descriptions = [step.get('description', '') for step in model_steps]
        ref_descriptions = [step.get('description', '') for step in ref_steps]
        
        # Remove empty descriptions
        model_descriptions = [desc for desc in model_descriptions if desc]
        ref_descriptions = [desc for desc in ref_descriptions if desc]
        
        if not model_descriptions or not ref_descriptions:
            return 0.0
        
        # Calculate embeddings
        model_embeddings = self.semantic_model.encode(model_descriptions)
        ref_embeddings = self.semantic_model.encode(ref_descriptions)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(model_embeddings, ref_embeddings)
        
        # Take the maximum similarity for each model step
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Return average of maximum similarities
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

# ✅ Main interactive function
def main():
    """Main function to run the interactive roadmap generator"""
    print("🚀 Welcome to the AI-Powered Learning Roadmap Generator!")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = RoadmapEvaluator(semantic_model)
    
    while True:
        print("\n" + "="*60)
        skill = input("💡 Enter the skill you want to learn (or 'quit' to exit): ").strip()
        
        if skill.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using the Roadmap Generator! Happy learning!")
            break
        
        if not skill:
            print("❌ Please enter a valid skill name.")
            continue
        
        print(f"\n🔄 Generating learning roadmap for: {skill}")
        print("⏳ This might take a moment...")
        
        try:
            # Generate the roadmap
            roadmap = generate_roadmap(skill)
            
            if not roadmap:
                print("❌ Failed to generate roadmap. Please try again.")
                continue
            
            # Display the generated roadmap
            print("\n" + "="*60)
            print(f"📚 Learning Roadmap for: {skill.title()}")
            print("="*60)
            print(roadmap)
            
            # Evaluate the roadmap
            print("\n" + "="*60)
            print("📊 ROADMAP EVALUATION")
            print("="*60)
            
            evaluation = evaluator.evaluate_roadmap(skill, roadmap)
            
            # Display structure scores
            struct = evaluation['structure']
            print(f"📋 Structure Analysis:")
            print(f"   • Total Steps: {struct['total_steps']}")
            print(f"   • Steps with Descriptions: {struct['has_descriptions']}")
            print(f"   • Time Estimates: {struct['has_time_estimates']}")
            print(f"   • Resource Links: {struct['has_links']}")
            print(f"   • Completeness Ratio: {struct['completeness_ratio']:.2%}")
            
            # Display semantic analysis if reference exists
            if evaluation['has_reference']:
                print(f"\n🧠 Semantic Analysis:")
                print(f"   • Similarity to Reference: {evaluation['semantic_similarity']:.2%}")
                print(f"   • Content Coverage: {evaluation['content_coverage']:.2%}")
            else:
                print(f"\n⚠️  No reference found for '{skill}' - semantic analysis unavailable")
            

            
        except Exception as e:
            print(f"❌ Error generating roadmap: {e}")
            print("Please check your model setup and try again.")

# ✅ Alternative function for single skill generation
def generate_single_roadmap(skill, show_evaluation=True):
    """Generate roadmap for a single skill (useful for scripting)"""
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

if __name__ == "__main__":
    # You can either run the interactive version or generate a single roadmap
    
    # Interactive version (default)
    main()
    
    # Or use the single roadmap function:
    # generate_single_roadmap("Machine Learning")