import torch
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer

from loading_my_model import (
    load_models,  # Remove this from load_models_cached()
    generate_roadmap,
    RoadmapEvaluator,
    semantic_model,
    model,        # Add these to check loaded status
    tokenizer
)

def generate_baseline_roadmap(model_name, skill):
    """Generate roadmap using different baseline approaches"""
    try:
        if model_name == "template_based":
            return generate_template_roadmap(skill)
        elif model_name == "flan_t5_small":
            return generate_flan_t5_roadmap(skill)
        elif model_name == "gpt2_basic":
            return generate_gpt2_basic_roadmap(skill)
        else:
            # For actual Hugging Face model IDs
            generator = pipeline('text-generation', model=model_name, device=0 if torch.cuda.is_available() else -1)
            
            prompt = f"""Generate a learning roadmap for {skill}:

1. """
            
            output = generator(
                prompt,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                truncation=True,
                pad_token_id=50256  # GPT-2 EOS token
            )
            
            full_text = output[0]['generated_text']
            return full_text[len(prompt):].strip()
            
    except Exception as e:
        return f"Error generating roadmap with {model_name}: {str(e)}"

def generate_template_roadmap(skill):
    """Simple template-based roadmap generation"""
    skill_lower = skill.lower()
    
    # Basic templates for common skills
    templates = {
        "python": [
            "1. Python Basics\nLearn syntax, variables, and data types\nTime: 1-2 weeks\nLink: https://python.org/tutorial",
            "2. Control Structures\nIf statements, loops, and functions\nTime: 1 week\nLink: https://docs.python.org/3/tutorial/controlflow.html",
            "3. Data Structures\nLists, dictionaries, and sets\nTime: 1 week\nLink: https://docs.python.org/3/tutorial/datastructures.html",
            "4. Object-Oriented Programming\nClasses, objects, and inheritance\nTime: 2 weeks\nLink: https://realpython.com/python3-object-oriented-programming/",
            "5. Libraries and Frameworks\nNumPy, Pandas, Flask/Django\nTime: 3-4 weeks\nLink: https://pypi.org"
        ],
        "javascript": [
            "1. JavaScript Fundamentals\nVariables, functions, and objects\nTime: 2 weeks\nLink: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
            "2. DOM Manipulation\nSelecting and modifying HTML elements\nTime: 1 week\nLink: https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model",
            "3. Asynchronous JavaScript\nPromises, async/await, and fetch API\nTime: 2 weeks\nLink: https://javascript.info/async",
            "4. Modern ES6+ Features\nArrow functions, destructuring, modules\nTime: 1 week\nLink: https://babeljs.io/docs/en/learn",
            "5. Frontend Frameworks\nReact, Vue, or Angular\nTime: 4-6 weeks\nLink: https://reactjs.org/docs/getting-started.html"
        ],
        "web development": [
            "1. HTML Fundamentals\nStructure, semantic tags, and forms\nTime: 1 week\nLink: https://developer.mozilla.org/en-US/docs/Web/HTML",
            "2. CSS Styling\nSelectors, layout, and responsive design\nTime: 2 weeks\nLink: https://developer.mozilla.org/en-US/docs/Web/CSS",
            "3. JavaScript Basics\nInteractivity and DOM manipulation\nTime: 2 weeks\nLink: https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "4. Frontend Framework\nReact, Vue, or Angular basics\nTime: 4 weeks\nLink: https://developer.mozilla.org/en-US/docs/Learn/Tools_and_testing/Client-side_JavaScript_frameworks",
            "5. Backend Development\nNode.js, databases, and APIs\nTime: 4 weeks\nLink: https://nodejs.org/en/docs/"
        ],
        "react": [
            "1. React Fundamentals\nComponents, JSX, and props\nTime: 1 week\nLink: https://reactjs.org/docs/getting-started.html",
            "2. State Management\nuseState and useEffect hooks\nTime: 1 week\nLink: https://reactjs.org/docs/hooks-intro.html",
            "3. Component Lifecycle\nMounting, updating, and unmounting\nTime: 1 week\nLink: https://reactjs.org/docs/state-and-lifecycle.html",
            "4. Advanced Patterns\nContext API, custom hooks\nTime: 2 weeks\nLink: https://reactjs.org/docs/context.html",
            "5. Ecosystem\nRouting, testing, and deployment\nTime: 2-3 weeks\nLink: https://create-react-app.dev/"
        ]
    }
    
    # Generic template for unknown skills
    generic_template = [
        f"1. {skill.title()} Fundamentals\nLearn basic concepts and terminology\nTime: 1-2 weeks\nLink: Search for official documentation",
        f"2. Core Concepts\nDeep dive into main {skill} principles\nTime: 2-3 weeks\nLink: Find comprehensive tutorials",
        f"3. Practical Applications\nBuild simple projects using {skill}\nTime: 2-4 weeks\nLink: Look for project-based tutorials",
        f"4. Advanced Topics\nExplore complex {skill} features\nTime: 3-4 weeks\nLink: Advanced courses and documentation",
        f"5. Real-world Projects\nCreate portfolio projects with {skill}\nTime: 4-6 weeks\nLink: Build personal projects"
    ]
    
    # Get appropriate template
    roadmap_steps = None
    for key in templates.keys():
        if key in skill_lower or skill_lower in key:
            roadmap_steps = templates[key]
            break
    
    if not roadmap_steps:
        roadmap_steps = generic_template
    
    return "\n\n".join(roadmap_steps)

def generate_flan_t5_roadmap(skill):
    """Generate using FLAN-T5 small model"""
    try:
        model_name = "google/flan-t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        prompt = f"Create a detailed learning roadmap for {skill}. Include step titles, descriptions, time estimates, and resource links. Format each step clearly."
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=400,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
        
    except Exception as e:
        return f"Error with FLAN-T5: {str(e)}"

def generate_gpt2_basic_roadmap(skill):
    """Generate using basic GPT-2 with better prompting"""
    try:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        prompt = f"""Learning Roadmap for {skill}:

Step 1: """
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=300,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt
        generated_part = result[len(prompt):].strip()
        return f"Step 1: {generated_part}"
        
    except Exception as e:
        return f"Error with GPT-2: {str(e)}"

# Updated function for your Streamlit app
def run_model_comparison(compare_skill, semantic_model):
    """Run comparison with properly implemented baseline models"""
    
    # Your model
    your_roadmap = generate_roadmap(compare_skill)
    your_eval = RoadmapEvaluator(semantic_model).evaluate_roadmap(compare_skill, your_roadmap)
    
    # Better baseline models for comparison
    baseline_models = {
        "Template-Based": "template_based",           # Rule-based baseline
        "FLAN-T5-Small": "flan_t5_small",            # Instruction-tuned but small
        "GPT-2-Basic": "gpt2_basic",                  # Basic GPT-2 with better prompting
        "GPT-Neo-125M": "EleutherAI/gpt-neo-125M",   # Small but coherent
        "DialoGPT-Small": "microsoft/DialoGPT-small" # Conversational model
    }
    
    results = {}
    for name, model_id in baseline_models.items():
        try:
            roadmap = generate_baseline_roadmap(model_id, compare_skill)
            eval_result = RoadmapEvaluator(semantic_model).evaluate_roadmap(compare_skill, roadmap)
            results[name] = {
                "roadmap": roadmap,
                "evaluation": eval_result
            }
        except Exception as e:
            print(f"Failed to generate with {name}: {str(e)}")
            results[name] = {
                "roadmap": f"Failed to generate: {str(e)}",
                "evaluation": None
            }
    
    return your_roadmap, your_eval, results