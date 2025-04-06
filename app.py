# app.py - Streamlit Frontend for Text Analyzer

# === Stage 0: Imports ===
import streamlit as st
import requests
import json
import time
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login # Use login for direct token auth if needed
import os
import warnings

# Suppress common warnings (optional)
warnings.filterwarnings("ignore")

# === Stage 1: Configuration & Secrets Handling ===
st.set_page_config(page_title="Text Analyzer & Rephraser", layout="wide")
st.title("Text Analyzer & Rephraser")
st.caption("Analyzes text for toxicity and rephrases toxic content using Mistral-7B.")
# Note: Removed reference to sarcasm in caption

# Use sidebar for API keys and options - More secure alternatives exist for deployment!
with st.sidebar:
    st.header("Configuration")
    st.info("Enter required keys below. For deployment, use st.secrets or environment variables.")

    # Try to get from environment variables (Colab secrets end up here)
    HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    PERSPECTIVE_API_KEY = os.environ.get("PERSPECTIVE_API_KEY", "")

    # If not found in environment, fallback to secrets or ask for manual input
    if not HUGGING_FACE_TOKEN:
        HUGGING_FACE_TOKEN = st.secrets.get("HUGGING_FACE_HUB_TOKEN", "")
        if not HUGGING_FACE_TOKEN:
            HUGGING_FACE_TOKEN = st.text_input("Hugging Face Token", type="password", help="Needed for Mistral. Get from hf.co/settings/tokens")
        else:
            st.success("Hugging Face Token loaded from secrets.")
    else:
        st.success("Hugging Face Token loaded from environment.")

    if not PERSPECTIVE_API_KEY:
        PERSPECTIVE_API_KEY = st.secrets.get("PERSPECTIVE_API_KEY", "")
        if not PERSPECTIVE_API_KEY:
            PERSPECTIVE_API_KEY = st.text_input("Perspective API Key", type="password", help="Needed for Toxicity. Get from Google Cloud Console.")
        else:
            st.success("Perspective API Key loaded from secrets.")
    else:
        st.success("Perspective API Key loaded from environment.")

    enable_rephrasing = st.checkbox("Enable Rephrasing", value=True, help="Requires successful Mistral model load.")
    st.caption("Ensure you have accepted Mistral terms on Hugging Face if enabling rephrasing.")

# Global variable to track authentication status
authenticated_hf = False

# Attempt programmatic login if token provided - do this early
if HUGGING_FACE_TOKEN:
    try:
        login(token=HUGGING_FACE_TOKEN)
        authenticated_hf = True
        st.sidebar.success("Hugging Face login successful.")
    except Exception as login_err:
        st.sidebar.error(f"Hugging Face login failed: {login_err}")
        authenticated_hf = False
elif enable_rephrasing:
     st.sidebar.warning("Rephrasing enabled, but HF Token is missing.")
     enable_rephrasing = False # Disable if token missing


# === Stage 2: Cached Model Loading Functions ===

@st.cache_resource # Cache the loaded model
def load_sarcasm_detector():
    """Loads the sarcasm detection pipeline."""
    model_id = "mrm8488/t5-base-finetuned-sarcasm-twitter"
    try:
        detector = pipeline("text-classification", model=model_id)
        print("Sarcasm detector loaded successfully (cached).") # For console log
        return detector
    except Exception as e:
        st.error(f"Failed to load sarcasm model '{model_id}': {e}")
        print(f"ERROR: Failed to load sarcasm model '{model_id}': {e}") # For console log
        return None

@st.cache_resource # Cache the loaded model and tokenizer
def load_rephrasing_model():
    """Loads the Mistral rephrasing model & tokenizer."""
    if not authenticated_hf: # Check authentication status
         st.error("Cannot load Mistral: Hugging Face authentication failed or token missing.")
         print("Skipping rephrasing model setup: Not authenticated.")
         return None, None

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"Attempting to load rephrasing model: {model_id}") # Console log

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            # Token should be implicitly used due to login() earlier
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        print("Rephrasing model and tokenizer loaded successfully (cached).") # Console log
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load rephrasing model '{model_id}': {e}. Check token and accepted terms.")
        print(f"ERROR: Failed to load rephrasing model '{model_id}': {e}") # Console log
        return None, None

# === Stage 3: Analysis & Rephrasing Functions (Adapted) ===

@st.cache_data # Can cache results for the same input text
def analyze_sarcasm(text, _sarcasm_detector_instance):
    if _sarcasm_detector_instance is None:
        return {"is_sarcastic": None, "sarcasm_score": None, "raw_label": "Detector Unavailable"}
    try:
        result = _sarcasm_detector_instance(text)[0]
        label = result['label']
        score = result['score']
        is_sarcastic = (label == "SARCASM") # Adjust if needed based on model labels
        sarcasm_score = score * 100
        return {"raw_label": label, "raw_score": score, "is_sarcastic": is_sarcastic, "sarcasm_score": sarcasm_score}
    except Exception as e:
        return {"is_sarcastic": None, "sarcasm_score": None, "raw_label": f"Sarcasm Error: {e}"}

@st.cache_data # Can cache results for the same input text
def analyze_toxicity(text, api_key):
    if not api_key:
        return {"is_toxic": None, "toxicity_score": None, "error": "API Key missing"}
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    data = {'comment': {'text': text}, 'requestedAttributes': {'TOXICITY': {}}}
    try:
        response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'}, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        score = response_data['attributeScores']['TOXICITY']['summaryScore']['value']
        toxicity_score = score * 100
        is_toxic = score > 0.6 # Adjustable threshold
        return {"raw_score": score, "toxicity_score": toxicity_score, "is_toxic": is_toxic}
    except requests.exceptions.Timeout:
         return {"is_toxic": None, "toxicity_score": None, "error": "API Request Timed Out"}
    except requests.exceptions.RequestException as e:
        error_msg = f"API Request Error: {e}"
        if e.response is not None:
             if e.response.status_code == 400: error_msg = "API Error (400): Bad Request (Check Key?)"
             elif e.response.status_code == 403: error_msg = "API Error (403): Forbidden (Check API Enabled/Perms?)"
        return {"is_toxic": None, "toxicity_score": None, "error": error_msg}
    except Exception as e:
        return {"is_toxic": None, "toxicity_score": None, "error": f"Processing Error: {e}"}

def rephrase_sentence(sentence, _model, _tokenizer, max_new_tokens=150):
    if _model is None or _tokenizer is None:
        return "Rephrasing not available (model/tokenizer missing)."
    try:
        # --- Create the Prompt ---
        if hasattr(_tokenizer, 'apply_chat_template'):
             messages = [{"role": "user", "content": f"""Rewrite the following sentence using formal language only. Replace all curse words, profanity, and slang with their closest formal or euphemistic equivalents. Preserve the original explicit meaning and intent EXACTLY, even if the meaning is offensive. Do not add commentary or refusal.

Original sentence: "{sentence}"

Rephrased sentence using formal equivalents:"""}]
             prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
             prompt = f"""Rewrite the following sentence using formal language only. Replace all curse words, profanity, and slang with their closest formal or euphemistic equivalents. Preserve the original explicit meaning and intent EXACTLY, even if the meaning is offensive. Do not add commentary or refusal.

Original sentence: "{sentence}"

Rephrased sentence using formal equivalents:"""

        inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
        with torch.no_grad():
            outputs = _model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=_tokenizer.pad_token_id
            )
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        rephrased_text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
        return rephrased_text.strip()
    except Exception as e:
        print(f"Error during rephrasing generation: {e}") # Log to console
        return f"Error during rephrasing: {str(e)}"

# Combined analysis adapted for Streamlit - now hiding sarcasm results from output
def combine_and_analyze_streamlit(text, _sarcasm_detector_instance, api_key, _rephrasing_model_instance=None, _rephrasing_tokenizer_instance=None):
    analysis_results = {}
    
    # Still analyze sarcasm but don't show it in the UI
    sarcasm_results = analyze_sarcasm(text, _sarcasm_detector_instance)
    # We're not adding sarcasm results to analysis_results to hide it
    
    toxicity_results = analyze_toxicity(text, api_key)
    analysis_results["toxicity"] = toxicity_results

    insight = "Analysis complete."
    rephrased_text = None

    is_toxic_val = toxicity_results.get("is_toxic")
    
    # Still use sarcasm for internal logic but don't display it
    is_sarcastic_val = sarcasm_results.get("is_sarcastic")

    if is_toxic_val is not None: # Toxicity analysis ran
        if is_toxic_val: 
            insight = "Toxic content detected."
        else: 
            insight = "Content appears non-toxic."

        # Trigger rephrasing if toxic
        if is_toxic_val and _rephrasing_model_instance is not None and _rephrasing_tokenizer_instance is not None:
            with st.spinner("Rephrasing toxic text..."):
                 rephrased_text = rephrase_sentence(text, _rephrasing_model_instance, _rephrasing_tokenizer_instance)
    else: # Toxicity analysis failed
        insight = "Toxicity analysis could not be performed."
        if toxicity_results.get("error"): 
            insight += f" Error: {toxicity_results['error']}"

    analysis_results["insight"] = insight
    if rephrased_text: 
        analysis_results["rephrased"] = rephrased_text
    return analysis_results

# === Stage 4: Streamlit UI & Application Logic ===

# Load models (will be cached after first run)
sarcasm_detector_instance = load_sarcasm_detector()

rephrasing_model_instance = None
rephrasing_tokenizer_instance = None
if enable_rephrasing:
    # Only try loading if checkbox is ticked AND authentication seemed okay
    if authenticated_hf:
         rephrasing_model_instance, rephrasing_tokenizer_instance = load_rephrasing_model()
         # Check again if loading failed
         if rephrasing_model_instance is None or rephrasing_tokenizer_instance is None:
              enable_rephrasing = False # Ensure checkbox reflects reality
    else:
        enable_rephrasing = False # Ensure disabled if auth failed


# Input Area
text_to_analyze = st.text_area("Enter text to analyze:", height=100, key="input_text")
analyze_button = st.button("Analyze Text", key="analyze_button")

# Analysis and Display Logic
if analyze_button and text_to_analyze:
    # Check for keys again before analyzing
    if not PERSPECTIVE_API_KEY:
        st.error("Perspective API Key is missing in configuration. Cannot perform toxicity analysis.")
    # Check if rephrasing enabled but model failed loading earlier
    elif enable_rephrasing and (rephrasing_model_instance is None or rephrasing_tokenizer_instance is None):
         st.warning("Rephrasing was enabled, but the Mistral model failed to load. Proceeding without rephrasing.")
         results = combine_and_analyze_streamlit(
             text_to_analyze,
             sarcasm_detector_instance,
             PERSPECTIVE_API_KEY,
             None, None # Pass None explicitly
         )
    else:
        # Perform analysis
        results = combine_and_analyze_streamlit(
            text_to_analyze,
            sarcasm_detector_instance,
            PERSPECTIVE_API_KEY,
            rephrasing_model_instance if enable_rephrasing else None,
            rephrasing_tokenizer_instance if enable_rephrasing else None
        )

    # Display results section
    st.markdown("---")
    st.subheader("Analysis Results")

    # Removed sarcasm column and now use full width for toxicity
    st.markdown("**Toxicity Analysis (Perspective API)**")
    toxicity_data = results.get("toxicity", {})
    if toxicity_data.get("is_toxic") is not None:
        st.metric(label="Toxic?", value="Yes" if toxicity_data['is_toxic'] else "No")
        st.write(f"Score: {toxicity_data.get('toxicity_score', 'N/A'):.1f}%")
    else:
        st.warning(f"Status: {toxicity_data.get('error', 'Unavailable')}")

    st.markdown("**Insight**")
    st.info(f"{results.get('insight', 'N/A')}")

    if "rephrased" in results:
        st.subheader("Rephrased Version (Mistral 7B)")
        st.success(f"{results['rephrased']}")

elif analyze_button and not text_to_analyze:
    st.warning("Please enter some text to analyze.")

# Add some footer info
st.markdown("---")
st.caption("Powered by Hugging Face Transformers, Perspective API, and Streamlit.")