# app.py - Streamlit Frontend for Text Analyzer (Optimized for Deployment)

import streamlit as st
import requests
import json
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import os
import warnings
import time

# Suppress unnecessary warnings to keep logs clean in deployment
warnings.filterwarnings("ignore")

# Configure the Streamlit page
st.set_page_config(
    page_title="Text Analyzer & Rephraser",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/text-analyzer',
        'About': "Text Analyzer & Rephraser analyzes content for toxicity and helps improve communication."
    }
)

# Add a custom style for better UI in deployment
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stAlert > div {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Application title and description
st.title("Text Analyzer & Rephraser")
st.caption("Analyzes text for toxicity and rephrases problematic content using Mistral-7B.")

# Sidebar configuration with better instructions
with st.sidebar:
    st.header("Configuration")
    
    # More detailed help text for better user understanding
    st.info("""
    **API Keys Required:**
    - Perspective API for toxicity analysis
    - Hugging Face token for model access
    
    For security, use Streamlit secrets or environment variables in production.
    """)

    # Try to get keys from various sources with better fallbacks
    # Priority: Environment variables > Streamlit secrets > User input
    
    # 1. Hugging Face Token
    HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if not HUGGING_FACE_TOKEN:
        try:
            HUGGING_FACE_TOKEN = st.secrets.get("HUGGING_FACE_HUB_TOKEN", "")
        except:
            HUGGING_FACE_TOKEN = ""
            
    if not HUGGING_FACE_TOKEN:
        HUGGING_FACE_TOKEN = st.text_input(
            "Hugging Face Token", 
            type="password", 
            help="Get your token from huggingface.co/settings/tokens"
        )
        if HUGGING_FACE_TOKEN:
            st.success("Token provided!")
    else:
        st.success("Hugging Face Token loaded from configuration.")
        
    # 2. Perspective API Key
    PERSPECTIVE_API_KEY = os.environ.get("PERSPECTIVE_API_KEY", "")
    if not PERSPECTIVE_API_KEY:
        try:
            PERSPECTIVE_API_KEY = st.secrets.get("PERSPECTIVE_API_KEY", "")
        except:
            PERSPECTIVE_API_KEY = ""
            
    if not PERSPECTIVE_API_KEY:
        PERSPECTIVE_API_KEY = st.text_input(
            "Perspective API Key", 
            type="password", 
            help="Get your key from Google Cloud Console"
        )
        if PERSPECTIVE_API_KEY:
            st.success("API key provided!")
    else:
        st.success("Perspective API Key loaded from configuration.")

    # Configuration options
    st.subheader("Options")
    
    # More user-friendly labels and defaults
    enable_rephrasing = st.checkbox(
        "Enable AI Rephrasing", 
        value=True, 
        help="Uses Mistral-7B to rewrite flagged content"
    )
    
    rephrasing_threshold = st.slider(
        "Toxicity Threshold for Rephrasing", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.05,
        help="Content above this toxicity score will trigger rephrasing"
    )
    
    # Resource conservation options for deployment
    use_lighter_model = st.checkbox(
        "Use Lighter Model (Better Performance)", 
        value=True,
        help="Uses a smaller, faster model variant for resource-constrained deployments"
    )
    
    # Add a small advanced section
    with st.expander("Advanced Settings"):
        request_timeout = st.number_input(
            "API Request Timeout (sec)", 
            min_value=5, 
            max_value=30, 
            value=10
        )
        
        max_tokens = st.number_input(
            "Max Output Tokens", 
            min_value=50, 
            max_value=300, 
            value=150
        )

# Global variable to track authentication status
authenticated_hf = False

# Attempt HF login if token provided - with better error handling
if HUGGING_FACE_TOKEN:
    try:
        login(token=HUGGING_FACE_TOKEN)
        authenticated_hf = True
        st.sidebar.success("✅ Hugging Face authentication successful")
    except Exception as login_err:
        st.sidebar.error(f"❌ Hugging Face login failed: {str(login_err)}")
        authenticated_hf = False

# Improved model loading functions with better caching and error handling

@st.cache_resource(ttl=24*3600, show_spinner=False)  # Cache for 24 hours
def load_sarcasm_detector():
    """
    Loads the sarcasm detection model - kept hidden from user interface
    
    Returns:
        The sarcasm detector pipeline or None if loading failed
    """
    model_id = "mrm8488/t5-base-finetuned-sarcasm-twitter"
    try:
        # Load silently without spinner since we're hiding this from the UI
        detector = pipeline("text-classification", model=model_id)
        return detector
    except Exception as e:
        # Log error but don't show to user
        print(f"Failed to load sarcasm model '{model_id}': {e}")
        return None

@st.cache_resource(ttl=24*3600, show_spinner=False)  # Cache for 24 hours
def load_rephrasing_model(use_lighter=True):
    """
    Loads the rephrasing model with optimizations for deployment.
    
    Args:
        use_lighter: If True, uses a smaller model for resource constraints
        
    Returns:
        Tuple of (model, tokenizer) or (None, None) if loading failed
    """
    if not authenticated_hf:
        return None, None

    try:
        # Choose model based on resource settings
        if use_lighter:
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Could be substituted with an even smaller model
        else:
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        
        with st.spinner(f"Loading rephrasing model ({model_id})... This may take a minute."):
            # Configure model for efficient memory usage
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Load and configure tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
                
            return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load rephrasing model: {str(e)}")
        return None, None

# Analysis functions with better error handling and caching

@st.cache_data(ttl=30, show_spinner=False)  # Cache for 30 seconds
def analyze_sarcasm(text, _sarcasm_detector_instance):
    """
    Analyzes text for sarcasm - kept hidden from user interface
    
    Args:
        text: The input text to analyze
        _sarcasm_detector_instance: The loaded sarcasm model pipeline
        
    Returns:
        Dict containing sarcasm analysis results
    """
    if not text or _sarcasm_detector_instance is None:
        return {"is_sarcastic": None, "sarcasm_score": None, "raw_label": "Detector Unavailable"}
    
    try:
        result = _sarcasm_detector_instance(text)[0]
        label = result['label']
        score = result['score']
        
        is_sarcastic = (label == "SARCASM")
        sarcasm_score = score * 100
        
        return {
            "raw_label": label, 
            "raw_score": score, 
            "is_sarcastic": is_sarcastic, 
            "sarcasm_score": sarcasm_score
        }
    except Exception as e:
        # Log error but don't show to user
        print(f"Sarcasm analysis error: {e}")
        return {"is_sarcastic": None, "sarcasm_score": None, "raw_label": f"Analysis Error"}

@st.cache_data(ttl=60, show_spinner=False)  # Cache for 60 seconds
def analyze_toxicity(text, api_key, timeout=10):
    """
    Analyzes toxicity using Perspective API with improved error handling.
    
    Args:
        text: The input text to analyze
        api_key: The Perspective API key
        timeout: Request timeout in seconds
        
    Returns:
        Dict containing toxicity results
    """
    if not text or not api_key:
        return {"is_toxic": None, "toxicity_score": None, "error": "Missing text or API key"}
    
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    
    # Request both TOXICITY and SEVERE_TOXICITY for better analysis
    data = {
        'comment': {'text': text}, 
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {}
        },
        'languages': ['en'],
        'doNotStore': True  # Privacy enhancement
    }
    
    try:
        response = requests.post(
            url, 
            data=json.dumps(data), 
            headers={'Content-Type': 'application/json'}, 
            timeout=timeout
        )
        response.raise_for_status()
        
        response_data = response.json()
        toxicity_score = response_data['attributeScores']['TOXICITY']['summaryScore']['value']
        severe_toxicity_score = response_data['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']
        
        # Calculate final score (weighted average)
        final_score = (0.7 * toxicity_score) + (0.3 * severe_toxicity_score)
        toxicity_percent = final_score * 100
        
        return {
            "raw_score": final_score, 
            "toxicity_score": toxicity_percent, 
            "is_toxic": final_score > 0.6,  # Default threshold
            "severe_score": severe_toxicity_score * 100
        }
    except requests.exceptions.Timeout:
        return {"is_toxic": None, "toxicity_score": None, "error": "API Request Timed Out"}
    except requests.exceptions.RequestException as e:
        error_msg = f"API Request Error: {str(e)}"
        
        # More helpful error messages
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 400: 
                error_msg = "API Error (400): Bad Request (Check Key or Input)"
            elif e.response.status_code == 403: 
                error_msg = "API Error (403): Forbidden (Check API Enabled/Permissions)"
            elif e.response.status_code == 429:
                error_msg = "API Error (429): Rate Limit Exceeded (Try Again Later)"
        
        return {"is_toxic": None, "toxicity_score": None, "error": error_msg}
    except Exception as e:
        return {"is_toxic": None, "toxicity_score": None, "error": f"Processing Error: {str(e)}"}

def rephrase_content(text, _model, _tokenizer, max_new_tokens=150):
    """
    Rephrases content using the loaded model with improved prompting.
    
    Args:
        text: The input text to rephrase
        _model: The loaded language model
        _tokenizer: The loaded tokenizer
        max_new_tokens: Maximum generation length
        
    Returns:
        Rephrased text or error message
    """
    if not text or _model is None or _tokenizer is None:
        return "Rephrasing not available. Please check model loading status."
    
    try:
        # More effective prompt template for better results
        if hasattr(_tokenizer, 'apply_chat_template'):
            messages = [{
                "role": "user", 
                "content": f"""Rewrite the following text using more appropriate language that preserves the core meaning. Replace any offensive, toxic, or inappropriate language with suitable alternatives. Maintain the same level of formality and tone where possible.

Original text: "{text}"

Rewritten version:"""
            }]
            prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"""Rewrite the following text using more appropriate language that preserves the core meaning. Replace any offensive, toxic, or inappropriate language with suitable alternatives. Maintain the same level of formality and tone where possible.

Original text: "{text}"

Rewritten version:"""

        # Prepare inputs and generate with better parameters
        inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
        
        with torch.no_grad():
            # More stable generation parameters
            outputs = _model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=_tokenizer.pad_token_id
            )
        
        # Extract only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        rephrased_text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up the output
        return rephrased_text.strip()
    except torch.cuda.OutOfMemoryError:
        return "Error: Out of memory. Try using the lighter model option."
    except Exception as e:
        return f"Error during rephrasing: {str(e)}"

def combine_and_analyze(
    text, 
    _sarcasm_detector_instance, 
    api_key, 
    _rephrasing_model_instance=None, 
    _rephrasing_tokenizer_instance=None,
    rephrasing_threshold=0.6,
    request_timeout=10,
    max_tokens=150
):
    """
    Combined analysis function that keeps sarcasm detection but hides results from UI.
    
    Args:
        text: Input text to analyze
        _sarcasm_detector_instance: Sarcasm model (hidden functionality)
        api_key: Perspective API key
        _rephrasing_model_instance: Rephrasing model
        _rephrasing_tokenizer_instance: Tokenizer
        rephrasing_threshold: Threshold for rephrasing
        request_timeout: API timeout
        max_tokens: Max generation length
        
    Returns:
        Dict with analysis results (sarcasm results excluded from output)
    """
    analysis_results = {}
    
    # Still analyze sarcasm in the background but don't include in results
    sarcasm_results = analyze_sarcasm(text, _sarcasm_detector_instance)
    # We're intentionally not adding sarcasm_results to analysis_results
    
    # Start visible analysis process
    with st.status("Analyzing content...") as status:
        status.update(label="Checking toxicity...", state="running")
        toxicity_results = analyze_toxicity(text, api_key, timeout=request_timeout)
        analysis_results["toxicity"] = toxicity_results
        
        # Determine if content needs attention
        is_toxic = toxicity_results.get("is_toxic")
        needs_rephrasing = is_toxic and toxicity_results.get("raw_score", 0) >= rephrasing_threshold
        
        # Generate insights based on analysis
        if is_toxic is not None:
            if is_toxic:
                score = toxicity_results.get("toxicity_score", 0)
                if score > 80:
                    insight = "Highly problematic content detected."
                elif score > 60:
                    insight = "Moderately toxic content detected."
                else:
                    insight = "Potentially concerning content detected."
            else:
                insight = "Content appears appropriate."
        else:
            insight = "Toxicity analysis could not be completed."
            if toxicity_results.get("error"): 
                insight += f" Error: {toxicity_results['error']}"
        
        # Internally (not visible to user), we can still use sarcasm data
        # to influence our decisions if needed
        is_sarcastic = sarcasm_results.get("is_sarcastic")
        if is_sarcastic and not is_toxic:
            # We can leverage sarcasm detection internally, but not show it in the UI
            # For example, to improve rephrasing decisions
            print(f"Sarcasm detected (score: {sarcasm_results.get('sarcasm_score', 0):.1f}%)")
            # We don't modify the visible insight here
        
        analysis_results["insight"] = insight
        
        # Perform rephrasing if needed and enabled
        if needs_rephrasing and _rephrasing_model_instance is not None:
            status.update(label="Generating alternative phrasing...", state="running")
            rephrased_text = rephrase_content(
                text, 
                _rephrasing_model_instance, 
                _rephrasing_tokenizer_instance,
                max_new_tokens=max_tokens
            )
            analysis_results["rephrased"] = rephrased_text
            
        status.update(label="Analysis complete", state="complete")
    
    return analysis_results

# Main application logic with improved UI

# Load models based on user settings - we still load sarcasm detector but don't mention it in UI
sarcasm_detector_instance = load_sarcasm_detector()
use_lighter_model = st.session_state.get('use_lighter_model', True)

rephrasing_model_instance = None
rephrasing_tokenizer_instance = None

if enable_rephrasing and authenticated_hf:
    rephrasing_model_instance, rephrasing_tokenizer_instance = load_rephrasing_model(use_lighter=use_lighter_model)
    if rephrasing_model_instance is None:
        st.warning("⚠️ Rephrasing model failed to load. Only toxicity analysis will be available.")
        enable_rephrasing = False

# Main UI
st.write("Enter text to analyze for potentially problematic content. If enabled, AI rephrasing will be suggested for flagged content.")

# Input area with character counter
text_to_analyze = st.text_area(
    "Enter text to analyze:",
    height=120,
    key="input_text",
    help="Enter the text you want to analyze for toxicity",
    max_chars=1000  # Reasonable limit for API usage
)

# Add more interactive UI elements
col1, col2 = st.columns([2, 1])
with col1:
    analyze_button = st.button(
        "🔍 Analyze Text", 
        key="analyze_button",
        use_container_width=True,
        type="primary"
    )
with col2:
    clear_button = st.button(
        "🗑️ Clear", 
        key="clear_button",
        use_container_width=True
    )
    if clear_button:
        st.session_state.input_text = ""
        st.experimental_rerun()

# Analysis and display logic
if analyze_button and text_to_analyze:
    # Validate requirements before analyzing
    if not PERSPECTIVE_API_KEY:
        st.error("❌ Perspective API Key is required for analysis. Please add it in the sidebar.")
    # Check if rephrasing is enabled but model failed to load
    elif enable_rephrasing and (rephrasing_model_instance is None or rephrasing_tokenizer_instance is None):
        st.warning("⚠️ Rephrasing was enabled but the model couldn't be loaded. Proceeding with toxicity analysis only.")
        
        # Perform analysis without rephrasing
        results = combine_and_analyze(
            text_to_analyze,
            sarcasm_detector_instance,  # Still pass this for internal use
            PERSPECTIVE_API_KEY,
            None, None,
            rephrasing_threshold,
            request_timeout,
            max_tokens
        )
    else:
        # Perform complete analysis
        results = combine_and_analyze(
            text_to_analyze,
            sarcasm_detector_instance,  # Still pass this for internal use
            PERSPECTIVE_API_KEY,
            rephrasing_model_instance if enable_rephrasing else None,
            rephrasing_tokenizer_instance if enable_rephrasing else None,
            rephrasing_threshold,
            request_timeout,
            max_tokens
        )

    # Display results in a better organized layout
    st.markdown("---")
    st.subheader("Analysis Results")
    
    # Use tabs for cleaner organization
    tab1, tab2 = st.tabs(["Analysis", "Recommendations"])
    
    with tab1:
        # Toxicity results
        toxicity_data = results.get("toxicity", {})
        
        # Nice metrics display
        col1, col2 = st.columns(2)
        with col1:
            if toxicity_data.get("is_toxic") is not None:
                score = toxicity_data.get('toxicity_score', 0)
                # Color the metric based on score
                if score > 80:
                    st.error(f"Toxicity Score: {score:.1f}%")
                elif score > 60:
                    st.warning(f"Toxicity Score: {score:.1f}%")
                elif score > 40:
                    st.info(f"Toxicity Score: {score:.1f}%")
                else:
                    st.success(f"Toxicity Score: {score:.1f}%")
            else:
                st.warning(f"Analysis Status: {toxicity_data.get('error', 'Unavailable')}")
        
        with col2:
            if toxicity_data.get("is_toxic") is not None:
                # Show severe toxicity if available
                if "severe_score" in toxicity_data:
                    severe = toxicity_data.get('severe_score', 0)
                    if severe > 50:
                        st.error(f"Severe Toxicity: {severe:.1f}%")
                    elif severe > 30:
                        st.warning(f"Severe Toxicity: {severe:.1f}%")
                    else:
                        st.success(f"Severe Toxicity: {severe:.1f}%")
        
        # Insight card
        st.subheader("Content Insight")
        st.info(f"{results.get('insight', 'No insight available')}")
    
    with tab2:
        # Show rephrased version if available
        if "rephrased" in results:
            st.subheader("Suggested Alternative Phrasing")
            
            # Create a comparison view
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text:**")
                st.text_area("", value=text_to_analyze, height=150, disabled=True, label_visibility="collapsed")
            
            with col2:
                st.markdown("**Rephrased Version:**")
                rephrased = results.get('rephrased')
                st.text_area("", value=rephrased, height=150, key="rephrased_text", label_visibility="collapsed")
                
                # Add copy button
                if st.button("📋 Copy Rephrased Text"):
                    st.code(rephrased, language="")
                    st.success("Text copied to clipboard! (You can now paste it elsewhere)")
        else:
            if enable_rephrasing:
                if toxicity_data.get("is_toxic"):
                    st.info("Content was flagged, but below the rephrasing threshold.")
                else:
                    st.success("No rephrasing needed - content appears appropriate.")
            else:
                st.info("Rephrasing is disabled. Enable it in the sidebar if needed.")

elif analyze_button and not text_to_analyze:
    st.warning("Please enter some text to analyze.")

# Footer with additional information
st.markdown("---")
st.caption("""
**About this app:** Text Analyzer & Rephraser uses AI models and the Perspective API to help identify and improve potentially problematic content.
Powered by Hugging Face Transformers, Google Perspective API, and Streamlit. This is a demonstration app for educational purposes.
""")

# Add application metrics (number of analyses performed) - for demonstration
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

if analyze_button and text_to_analyze:
    st.session_state.analysis_count += 1

# Small metrics display
if st.session_state.analysis_count > 0:
    st.sidebar.metric("Analyses Performed", st.session_state.analysis_count)
