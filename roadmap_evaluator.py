import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

class RoadmapEvaluator:
    """Comprehensive Roadmap Evaluation System"""
    
    def __init__(self, semantic_model_name='all-MiniLM-L6-v2', baseline_model_name=None):
        """
        Initialize the evaluator with semantic similarity model and optional baseline model
        
        Args:
            semantic_model_name: Name of the sentence transformer model for similarity
            baseline_model_name: Name of the baseline model to compare against (optional)
        """
        self.semantic_model = SentenceTransformer(semantic_model_name)
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
        
        # Use the same prompt format as your main model
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
        
        # Extract only the roadmap part (remove prompt)
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
            
            # Check if this is a step title (not starting with Time:, Link:, YouTube:)
            if not line.startswith(('Time:', 'Link:', 'YouTube:')):
                # Save previous step if exists
                if current_step:
                    steps.append(current_step)
                
                # Start new step
                current_step = {'title': line}
                
                # Look for description on next line
                if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(('Time:', 'Link:', 'YouTube:')):
                    current_step['description'] = lines[i + 1].strip()
                    i += 1
                    
            elif line.startswith('Time:'):
                current_step['time'] = line.replace('Time:', '').strip()
            elif line.startswith('Link:'):
                current_step['link'] = line.replace('Link:', '').strip()
            elif line.startswith('YouTube:'):
                current_step['youtube'] = line.replace('YouTube:', '').strip()
            
            i += 1
        
        # Add the last step
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
                'has_youtube_links': 0,
                'avg_title_length': 0,
                'avg_description_length': 0
            }
        
        complete_steps = 0
        has_descriptions = 0
        has_time_estimates = 0
        has_links = 0
        has_youtube_links = 0
        title_lengths = []
        description_lengths = []
        
        for step in steps:
            # Count complete steps (have at least title, description, and time)
            if all(key in step for key in ['title', 'description', 'time']):
                complete_steps += 1
            
            if 'description' in step:
                has_descriptions += 1
                description_lengths.append(len(step['description']))
            
            if 'time' in step:
                has_time_estimates += 1
            
            if 'link' in step:
                has_links += 1
            
            if 'youtube' in step:
                has_youtube_links += 1
            
            if 'title' in step:
                title_lengths.append(len(step['title']))
        
        return {
            'total_steps': len(steps),
            'complete_steps': complete_steps,
            'completeness_ratio': complete_steps / len(steps) if steps else 0.0,
            'has_descriptions': has_descriptions,
            'has_time_estimates': has_time_estimates,
            'has_links': has_links,
            'has_youtube_links': has_youtube_links,
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
    
    def comprehensive_evaluation(self, 
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
    
    def save_evaluation_results(self, results: Dict, output_file: str = None):
        """Save evaluation results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Evaluation results saved to: {output_file}")
    
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
        print(f"  YouTube links: {struct['has_youtube_links']}")
        
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

# Helper function for batch evaluation
def batch_evaluate_skills(evaluator: RoadmapEvaluator, 
                         skills: List[str], 
                         model_outputs: List[str],
                         instruction_prompt: str = None) -> List[Dict]:
    """
    Evaluate multiple skills in batch
    
    Args:
        evaluator: RoadmapEvaluator instance
        skills: List of skill names
        model_outputs: List of corresponding model outputs
        instruction_prompt: Prompt template for baseline comparison
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for skill, output in zip(skills, model_outputs):
        print(f"\n🔍 Evaluating: {skill}")
        result = evaluator.comprehensive_evaluation(skill, output, instruction_prompt)
        evaluator.print_evaluation_summary(result)
        results.append(result)
    
    return results

if __name__ == "__main__":
    # Example usage
    evaluator = RoadmapEvaluator()
    
    # Example roadmap for testing
    sample_roadmap = """Python Basics
Learn the fundamental concepts and syntax of Python programming.
Time: 3 days
Link: https://www.python.org/
YouTube: https://www.youtube.com/watch?v=example

Data Structures
Understand lists, dictionaries, and other data structures.
Time: 2 days
Link: https://docs.python.org/
YouTube: https://www.youtube.com/watch?v=example2"""
    
    # Evaluate the roadmap
    results = evaluator.comprehensive_evaluation("Python", sample_roadmap)
    evaluator.print_evaluation_summary(results)
    evaluator.save_evaluation_results(results)