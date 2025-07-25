import streamlit as st
import os
import tempfile
import torch
from io import StringIO

# Import your existing functions from the main file
# Assuming your main file is named 'roadmap_generator.py'
from loading_my_model import (
    load_models, 
    generate_roadmap, 
    generate_single_roadmap,
    RoadmapEvaluator,
    semantic_model
)
from pdf_mode import generate_roadmap_from_pdf

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
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #1f77b4;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #000000;
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
        with st.spinner("🔄 Loading AI models... This may take a few minutes on first run."):
            try:
                # Actually call the load_models function from roadmap_generator
                from loading_my_model import load_models
                load_models()
                st.session_state.models_loaded = True
                st.success("✅ Models loaded successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                return False
    return True

def display_roadmap(roadmap, title="Generated Learning Roadmap"):
    """Display the roadmap in a nice formatted way"""
    st.markdown(f'<div class="sub-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="roadmap-container">{roadmap}</div>', unsafe_allow_html=True)

def display_evaluation(evaluation):
    """Display evaluation metrics in a nice format"""
    if not evaluation:
        return
    
    st.markdown('<div class="sub-header">📊 Roadmap Quality Analysis</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        struct = evaluation['structure']
        
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
    
    # Additional metrics if reference exists
    if evaluation['has_reference']:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Semantic Similarity", f"{evaluation['semantic_similarity']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Content Coverage", f"{evaluation['content_coverage']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Completeness ratio
    st.markdown('<div class="evaluation-box">', unsafe_allow_html=True)
    st.write(f"**Completeness Ratio:** {struct['completeness_ratio']:.1%} of steps have all required components (title, description, time, link)")
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI-Powered Learning Roadmap Generator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading status
        st.header("⚙️ Model Status")
        
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
            st.warning("⏳ Models Not Loaded")
        
        if st.button("🔄 Load Models"):
            success = load_models_cached()
            if success:
                st.rerun()  # Refresh the page to update status
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            show_evaluation = st.checkbox("Show Quality Analysis", value=True)
            max_attempts = st.slider("Generation Attempts", 1, 5, 3)
            temperature = st.slider("Creativity Level", 0.0, 1.0, 0.3)
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About"):
            st.write("""
            This AI-powered tool generates personalized learning roadmaps using:
            - **Fine-tuned Gemma Model** for roadmap generation
            - **Semantic Analysis** for quality assessment
            - **PDF Processing** for custom content extraction
            - **Interactive UI** for easy use
            """)
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["📝 Skill-Based Roadmap", "📄 PDF-Based Roadmap"])
    
    with tab1:
        st.markdown('<div class="info-message">Enter a skill or technology you want to learn, and I\'ll generate a comprehensive learning roadmap for you!</div>', unsafe_allow_html=True)
        
        # Skill input
        col1, col2 = st.columns([3, 1])
        with col1:
            skill_input = st.text_input(
                "What would you like to learn?", 
                placeholder="e.g., Python, Web Development, Machine Learning, React, etc.",
                help="Enter any programming language, framework, or technology"
            )
        
        with col2:
            st.write("")  # Add some space
            generate_skill_btn = st.button("🚀 Generate Roadmap", type="primary", use_container_width=True)
        
        # Popular skills quick selection
        st.write("**Quick Select Popular Skills:**")
        skill_cols = st.columns(6)
        popular_skills = ["Python", "JavaScript", "React", "Machine Learning", "Web Development", "Data Science"]
        
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
                with st.spinner(f"🔄 Generating learning roadmap for {skill_input}..."):
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
                                evaluation = evaluator.evaluate_roadmap(skill_input, roadmap)
                                st.session_state.current_evaluation = evaluation
                            
                            st.markdown('<div class="success-message">✅ Roadmap generated successfully!</div>', unsafe_allow_html=True)
                        
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
    
    with tab2:
        st.markdown('<div class="info-message">Upload a PDF textbook or document, and I\'ll extract key topics to generate a structured learning roadmap!</div>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a textbook, course material, or any educational PDF"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size} bytes"
            }
            st.write("**File Details:**")
            for key, value in file_details.items():
                st.write(f"- **{key}:** {value}")
            
            # Generate roadmap button
            if st.button("📄 Generate PDF Roadmap", type="primary"):
                if not st.session_state.models_loaded:
                    st.error("❌ Please load the models first using the sidebar.")
                else:
                    # Double-check that models are actually loaded in the core module
                    try:
                        from loading_my_model import model, tokenizer
                        if model is None or tokenizer is None:
                            st.error("❌ Models not properly loaded. Please try reloading models.")
                        else:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            try:
                                with st.spinner("🔄 Processing PDF and generating roadmap... This may take several minutes."):
                                    # Generate roadmap from PDF
                                    roadmap_steps = generate_roadmap_from_pdf(tmp_file_path, generate_roadmap)
                                    
                                    if roadmap_steps:
                                        # Combine all steps into one roadmap
                                        full_roadmap = "\n\n".join(
                                            f"**Topic {i+1}: {topic}**\n{output}" 
                                            for i, (topic, output) in enumerate(roadmap_steps)
                                        )
                                        
                                        st.markdown('<div class="success-message">✅ PDF roadmap generated successfully!</div>', unsafe_allow_html=True)
                                        display_roadmap(full_roadmap, "PDF-Based Learning Roadmap")
                                        
                                        # Download option
                                        st.download_button(
                                            label="📥 Download PDF Roadmap",
                                            data=full_roadmap,
                                            file_name=f"{uploaded_file.name}_roadmap.txt",
                                            mime="text/plain"
                                        )
                                        
                                        # Show individual topics
                                        with st.expander("📋 View Individual Topics"):
                                            for i, (topic, output) in enumerate(roadmap_steps):
                                                st.write(f"**Topic {i+1}: {topic}**")
                                                st.write(output)
                                                st.markdown("---")
                                    else:
                                        st.markdown('<div class="error-message">❌ Failed to generate roadmap. No topics found in the PDF.</div>', unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.markdown(f'<div class="error-message">❌ Error processing PDF: {str(e)}</div>', unsafe_allow_html=True)
                            
                            finally:
                                # Clean up temporary file
                                if os.path.exists(tmp_file_path):
                                    os.unlink(tmp_file_path)
                    
                    except ImportError as e:
                        st.error(f"❌ Error importing models: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            Made with ❤️ using Streamlit | Powered by AI
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()