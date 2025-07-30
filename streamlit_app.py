import streamlit as st
import re
import torch
from io import StringIO
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import pipeline

from gemma_loader import generate_gemma_roadmap

# Import your existing functions from the main file
# Assuming your main file is named 'roadmap_generator.py'
from loading_my_model import (
    load_models,  # Remove this from load_models_cached()
    generate_roadmap,
    RoadmapEvaluator,
    semantic_model,
    model,        # Add these to check loaded status
    tokenizer,
)

# Set page config
st.set_page_config(
    page_title="AI Learning Roadmap Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 5px;
    }
    .roadmap-container {
        line-height: 1.8;
        padding: 20px;
    }
    .roadmap-container br {
        display: block;
        margin: 10px 0;
        content: "";
    }
    .evaluation-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border: 1px solid #bee5eb;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #dc3545;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #17a2b8;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        border: 1px solid #dee2e6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'roadmap_generated' not in st.session_state:
        st.session_state.roadmap_generated = False
    if 'current_roadmap' not in st.session_state:
        st.session_state.current_roadmap = ""
    if 'current_evaluation' not in st.session_state:
        st.session_state.current_evaluation = None

def load_models_cached():
    """Load models with caching and error handling"""
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models... This may take a few minutes on first run."):
            try:
                # Actually call the load_models function from roadmap_generator
                from loading_my_model import load_models
                load_models()
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
                return True
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                return False
    return True

def get_youtube_link(skill):
    youtube_playlists = {
            "python": "https://www.youtube.com/watch?v=_uQrJ0TkZlc",
            "javascript": "https://www.youtube.com/watch?v=PkZNo7MFNFg",
            "html": "https://www.youtube.com/watch?v=UB1O30fR-EE",
            "css": "https://www.youtube.com/watch?v=yfoY53QXEnI",
            "react": "https://www.youtube.com/watch?v=w7ejDZ8SWv8",
            "java": "https://www.youtube.com/watch?v=eIrMbAQSU34",
            "node": "https://www.youtube.com/watch?v=TlB_eWDSMt4",
            "php": "https://www.youtube.com/watch?v=OK_JCtrrv-c",
            "sql": "https://www.youtube.com/watch?v=HXV3zeQKqGY",
            "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
            "django": "https://www.youtube.com/watch?v=F5mRW0jo-U4",
            "flask": "https://www.youtube.com/watch?v=Z1RJmh_OqeA",
            "angular": "https://www.youtube.com/watch?v=3qBXWUpoPHo",
            "vue": "https://www.youtube.com/watch?v=qZXt1Aom3Cs",
            "bootstrap": "https://www.youtube.com/watch?v=4sosXZsdy-s",
            "jquery": "https://www.youtube.com/watch?v=hMxGhHNOkCU",
            "mongodb": "https://www.youtube.com/watch?v=ExcRbA7fy_A",
            "express": "https://www.youtube.com/watch?v=L72fhGm1tfE",
            "typescript": "https://www.youtube.com/watch?v=BwuLxPH8IDs",
            "docker": "https://www.youtube.com/watch?v=fqMOX6JJhGo",
            "aws": "https://www.youtube.com/watch?v=3hLmDS179YE",
            "machine learning": "https://www.youtube.com/watch?v=ukzFI9rgwfU",
            "data science": "https://www.youtube.com/watch?v=ua-CiDNNj30",
            "web development": "https://www.youtube.com/watch?v=UB1O30fR-EE",
        }
    return youtube_playlists.get(skill.lower())

def display_roadmap(roadmap, skill=None, title="Generated Learning Roadmap"):
    st.markdown(f'<div class="sub-header">{title}</div>', unsafe_allow_html=True)
    
    # Process and format the roadmap text
    formatted_roadmap = []
    current_step = []
    
    for line in roadmap.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a new step (line starts with number or bullet)
        if re.match(r'^\d+\.|^[•-]', line):
            if current_step:  # If we have a previous step, add it before starting new one
                formatted_roadmap.append('\n'.join(current_step))
                formatted_roadmap.append('')  # Add empty line between steps
            current_step = [f"**{line}**"]  # Start new step with bold title
        else:
            # Add description, time, or link with proper formatting
            if line.startswith('Time:'):
                current_step.append(f"{line}")
            elif line.startswith('Link:'):
                current_step.append(f"{line.replace('Link: ', '')}")
            elif line.startswith('YouTube:'):
                current_step.append(f"[Watch Tutorial]({line.replace('YouTube: ', '')})")
            else:
                current_step.append(line)
    
    # Add the last step
    if current_step:
        formatted_roadmap.append('\n'.join(current_step))
    
    # Display with proper formatting
    st.markdown(
        f'<div class="roadmap-container">{"<br><br>".join(formatted_roadmap)}</div>',
        unsafe_allow_html=True
    )
    
    # YouTube link if available
    if skill:
        youtube_link = get_youtube_link(skill.lower())
        if youtube_link:
            st.markdown(f"📺 **Recommended YouTube Playlist:** [Watch Here]({youtube_link})")


def display_evaluation(evaluation):
    """Display evaluation metrics in a nice format"""
    if not evaluation:
        return
    
    st.markdown('<div class="sub-header">Roadmap Quality Analysis</div>', unsafe_allow_html=True)
    
    # Handle both old and new evaluation formats
    if 'model_structure' in evaluation:  # New format from comprehensive_evaluation
        struct = evaluation['model_structure']
        has_reference = evaluation['has_reference']
    else:  # Old format
        struct = evaluation['structure']
        has_reference = evaluation.get('has_reference', False)
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Steps", struct['total_steps'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Descriptions", struct['has_descriptions'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Time Estimates", struct['has_time_estimates'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Resource Links", struct['has_links'])
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Completeness ratio
    st.markdown('<div class="evaluation-box">', unsafe_allow_html=True)
    st.write(f"**Completeness Ratio:** {struct['completeness_ratio']:.1%} of steps have all required components")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional metrics if reference exists
    if has_reference:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Semantic Similarity", f"{evaluation.get('reference_similarity', 0):.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Content Coverage", f"{evaluation.get('content_coverage', {}).get('coverage_ratio', 0):.1%}")
            st.markdown('</div>', unsafe_allow_html=True)

def generate_baseline_roadmap(model_id, skill):
    """Safe generation with memory management"""
    try:
        # Special handling for different model types
        if model_id == "microsoft/phi-1_5":
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            prompt = f"""Write a learning roadmap for {skill} with:
1. Clear steps
2. Time estimates
3. Resource links

Example:
1. Basics
Learn core concepts
Time: 2 weeks
Link: example.com/basics

Now generate for {skill}:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=400)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        
        elif model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
            generator = pipeline(
                "text-generation",
                model=model_id,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16
            )
            prompt = f"""<|system|>
            Generate a learning roadmap for {skill} with:
            - Numbered steps
            - Time estimates
            - Resource links</s>
            <|user|>
            Please create the roadmap</s>
            <|assistant|>"""
            
            output = generator(
                prompt,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True
            )
            return output[0]['generated_text'][len(prompt):].strip()
        
        else:  # For GPT-2 and other standard models
            generator = pipeline(
                "text-generation",
                model=model_id,
                device=0 if torch.cuda.is_available() else -1,
                model_kwargs={"load_in_4bit": True}  # Quantization for memory
            )
            prompt = f"""Create a learning roadmap for {skill} containing:
1. Step titles
2. Descriptions
3. Time required
4. Resource links

Example:
1. Introduction
Learn basic syntax
Time: 1 week
Link: example.com/intro

Now generate for {skill}:"""
            
            output = generator(
                prompt,
                max_length=600,
                temperature=0.7,
                num_return_sequences=1
            )
            return output[0]['generated_text'][len(prompt):].strip()
    
    except Exception as e:
        return f"Error: {str(e)}"



def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">AI-Powered Learning Roadmap Generator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading status
        st.header("Model Status")
        
        # Check actual model status from the core module
        try:
            from loading_my_model import model, tokenizer
            actual_models_loaded = model is not None and tokenizer is not None
        except:
            actual_models_loaded = False
        
        if st.session_state.models_loaded and actual_models_loaded:
            st.success("✅ Models Ready")
        elif st.session_state.models_loaded and not actual_models_loaded:
            st.warning("⚠️ Session says loaded but models not found")
            st.session_state.models_loaded = False  # Reset the session state
        else:
            st.warning("Models Not Loaded")
        
        if st.button("Load Models"):
            success = load_models_cached()
            if success:
                st.rerun()  # Refresh the page to update status
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("Advanced Options"):
            show_evaluation = st.checkbox("Show Quality Analysis", value=True)
            max_attempts = st.slider("Generation Attempts", 1, 5, 3)
        
        st.markdown("---")
        
        # About section
        with st.expander("About"):
            st.write("""
            This AI-powered tool generates personalized learning roadmaps using:
            - **Fine-tuned Gemma Model** for roadmap generation
            - **Semantic Analysis** for quality assessment
            - **PDF Processing** for custom content extraction
            - **Interactive UI** for easy use
            """)
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Skill-Based Roadmap", "Model Comparison"])
    
    with tab1:
        st.markdown('<div class="info-message">Enter a skill or technology you want to learn, and I\'ll generate a comprehensive learning roadmap for you!</div>', unsafe_allow_html=True)
        
        # Skill input
        col1, col2 = st.columns([3, 1])
        with col1:
            skill_input = st.text_input(
                "What would you like to learn?", 
                placeholder="e.g., Python, Web Development, React, etc.",
                help="Enter any programming language, framework, or technology"
            )
        
        with col2:
            st.write("")  # Add some space
            generate_skill_btn = st.button("Generate Roadmap", type="primary", use_container_width=True)
        
        # Popular skills quick selection
        st.write("**Quick Select Popular Skills:**")
        skill_cols = st.columns(6)
        popular_skills = ["Python", "JavaScript", "React", "Web Development"]
        
        for i, skill in enumerate(popular_skills):
            with skill_cols[i]:
                if st.button(skill, key=f"skill_{i}"):
                    skill_input = skill
                    generate_skill_btn = True
        
        # Generate roadmap for skill
        if generate_skill_btn and skill_input:
            if not st.session_state.models_loaded:
                st.error("❌ Please load the models first using the sidebar.")
            else:
                with st.spinner(f"Generating learning roadmap for {skill_input}..."):
                    try:
                        # Ensure models are actually loaded in the core module
                        from loading_my_model import model, tokenizer
                        if model is None or tokenizer is None:
                            st.error("❌ Models not properly loaded. Please try reloading models.")
                        else:
                            # Generate roadmap
                            roadmap = generate_roadmap(skill_input)
                            st.session_state.current_roadmap = roadmap
                            st.session_state.roadmap_generated = True
                            
                            # Evaluate roadmap if enabled
                            if show_evaluation:
                                evaluator = RoadmapEvaluator(semantic_model)
                                evaluation = evaluator.evaluate_roadmap(
                                    skill=skill_input,
                                    model_output=roadmap,
                                    instruction_prompt=None  # Skip baseline comparison if not needed
                                )
                                st.session_state.current_evaluation = evaluation
                            
                            st.markdown('<div class="success-message">Roadmap generated successfully!</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.markdown(f'<div class="error-message">❌ Error generating roadmap: {str(e)}</div>', unsafe_allow_html=True)
        
        # Display results for skill-based roadmap
        if st.session_state.roadmap_generated and st.session_state.current_roadmap:
            display_roadmap(st.session_state.current_roadmap)
            
            if show_evaluation and st.session_state.current_evaluation:
                display_evaluation(st.session_state.current_evaluation)
            
            # Download option
            st.download_button(
                label="📥 Download Roadmap",
                data=st.session_state.current_roadmap,
                file_name=f"{skill_input.lower().replace(' ', '_')}_roadmap.txt",
                mime="text/plain"
            )
    
    # Replace the comparison tab section in your Streamlit app with this:

    with tab2:
        st.markdown('<div class="sub-header">Model Comparison</div>', unsafe_allow_html=True)

        # Skill selection
        compare_skill = st.selectbox(
            "Select skill to compare:",
            ["Python", "JavaScript", "Machine Learning", "Web Development"],
            key="compare_skill"
        )

        if st.button("Compare with other models", type="primary"):
            if not st.session_state.models_loaded:
                st.error("Please load your model first in the sidebar")
            else:
                with st.spinner("Generating comparison..."):
                    cols = st.columns(2)

                    # Your model
                    with cols[0]:
                        st.markdown("#### Your Fine-Tuned Model")
                        your_roadmap = generate_roadmap(compare_skill)
                        display_roadmap(your_roadmap)
                        your_eval = RoadmapEvaluator(semantic_model).evaluate_roadmap(compare_skill, your_roadmap)
                        display_evaluation(your_eval)

                    # Gemma-2B
                    with cols[1]:
                        st.markdown("#### Gemma-2B Baseline")
                        gemma_roadmap = generate_gemma_roadmap(compare_skill)

                        if gemma_roadmap.startswith("Gemma-2B Error"):
                            st.error(gemma_roadmap)
                        else:
                            display_roadmap(gemma_roadmap)
                            gemma_eval = RoadmapEvaluator(semantic_model).evaluate_roadmap(compare_skill, gemma_roadmap)
                            display_evaluation(gemma_eval)

                    # Metrics comparison
                    st.markdown("---")
                    st.markdown("### 📊 Direct Comparison")

                    if not gemma_roadmap.startswith("Gemma-2B Error"):
                        comparison = {
                            "Metric": ["Steps", "Complete Steps", "Content Coverage"],
                            "Your Model": [
                                your_eval['structure']['total_steps'],
                                your_eval['structure']['complete_steps'],
                                f"{your_eval.get('content_coverage', {}).get('coverage_ratio', 0):.1%}"
                            ],
                            "Gemma-2B": [
                                gemma_eval['structure']['total_steps'],
                                gemma_eval['structure']['complete_steps'],
                                f"{gemma_eval.get('content_coverage', {}).get('coverage_ratio', 0):.1%}"
                            ]
                        }
                        st.table(comparison)
        
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            Made using Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()