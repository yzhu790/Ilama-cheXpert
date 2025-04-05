# vlm_handler.py
"""Handles LLaVA GGUF model loading, inference, and output parsing."""

import os
import re
import base64
import io
from PIL import Image, UnidentifiedImageError
import sys

# Import constants
from constants import CHEXPERT_LABELS

# Check for llama_cpp installation
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! Error: llama-cpp-python is not installed.                  !!!")
    print("!!! Please install it, potentially with GPU/Metal support.     !!!")
    print("!!! See https://github.com/abetlen/llama-cpp-python for install instructions. !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit(1)


def initialize_llava_gguf_model(gguf_model_path, mmproj_model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False):
    """
    Loads the LLaVA GGUF model using llama-cpp-python.
    Returns the loaded Llama object or None if loading fails.
    """
    if not os.path.exists(gguf_model_path):
        print(f"Error: GGUF model file not found: {gguf_model_path}")
        return None
    if not os.path.exists(mmproj_model_path):
        print(f"Error: Multimodal projector file not found: {mmproj_model_path}")
        return None

    print("Loading LLaVA GGUF model...")
    print(f"  GGUF Path: {gguf_model_path}")
    print(f"  MMPROJ Path: {mmproj_model_path}")
    print(f"  N_GPU_Layers: {n_gpu_layers}")
    print(f"  N_CTX: {n_ctx}")

    try:
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_model_path, verbose=verbose)
        llm = Llama(
            model_path=gguf_model_path,
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=True, # Required for multimodal embeddings
            verbose=verbose
        )
        print("Model loaded successfully.")
        return llm
    except Exception as e:
        print(f"Error loading LLaVA GGUF model: {e}")
        return None

def _image_to_base64(image: Image.Image, format="JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def _parse_vlm_output(text_output):
    """
    Attempts to parse the VLM's text output for CheXpert labels.
    Returns a dictionary with labels mapped to 0 or 1.
    (This function might need significant tuning based on model output)
    """
    diagnosis = {label: 0 for label in CHEXPERT_LABELS} # Default to 0
    lines = text_output.strip().split('\n')
    found_labels_count = 0

    # Primary Strategy: Look for "Label: 0" or "Label: 1"
    pattern = re.compile(r"^\s*(" + "|".join(re.escape(label) for label in CHEXPERT_LABELS) + r")\s*:?\s*([01])\s*$")
    for line in lines:
        match = pattern.search(line.strip())
        if match:
            label_name = match.group(1).strip()
            value = int(match.group(2))
            if label_name in diagnosis:
                diagnosis[label_name] = value
                found_labels_count += 1

    # Fallback Strategy (Basic Keyword Spotting)
    if found_labels_count < len(CHEXPERT_LABELS) // 2:
        # print(f"\nDebug: Strict parsing found {found_labels_count} labels. Trying keyword spotting.") # Optional debug
        for label in CHEXPERT_LABELS:
            if label != "No Finding" and diagnosis[label] == 0:
                 if re.search(r"\b" + re.escape(label) + r"\b", text_output, re.IGNORECASE):
                     diagnosis[label] = 1
        if diagnosis["No Finding"] == 0:
            if re.search(r"\bno findings?\b|\bnormal\b|unremarkable", text_output, re.IGNORECASE):
                 diagnosis["No Finding"] = 1

    # Refinement Rule
    if diagnosis.get("No Finding") == 1:
        for label in CHEXPERT_LABELS:
            if label != "No Finding":
                diagnosis[label] = 0
    return diagnosis

def run_inference(llm_model: Llama, image_path: str, prompt_template: str, max_tokens: int = 150):
    """
    Runs inference on a single image using the loaded LLaVA model.

    Args:
        llm_model: The loaded Llama object.
        image_path: Path to the image file.
        prompt_template: The text prompt to use.
        max_tokens: Maximum tokens to generate.

    Returns:
        tuple: (diagnosis_dict, raw_response_text, error_message)
               diagnosis_dict contains labels mapped to 0/1, or -1 on error.
               raw_response_text is the model's output string.
               error_message is non-empty if an error occurred.
    """
    if not llm_model:
        return {label: -1 for label in CHEXPERT_LABELS}, "", "Model not loaded"

    error_msg = ""
    raw_response_text = ""
    diagnosis_dict = {label: -1 for label in CHEXPERT_LABELS} # Default to error state

    try:
        # Load image and convert to base64
        image = Image.open(image_path).convert("RGB")
        image_b64 = _image_to_base64(image)

        # Prepare messages for chat completion
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt_template}
                ]
            }
        ]

        # Generate completion
        response = llm_model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )

        # Process response
        if response and 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message'].get('content', '')
            if content:
                raw_response_text = content.strip()
                diagnosis_dict = _parse_vlm_output(raw_response_text) # Parse successful output
            else:
                error_msg = "Model returned empty content"
        else:
            error_msg = f"Unexpected response structure: {str(response)[:200]}..."

    except FileNotFoundError:
        error_msg = f"Image file not found: {image_path}"
    except UnidentifiedImageError:
        error_msg = f"Cannot identify image file (corrupted/unsupported): {image_path}"
    except Exception as e:
        error_msg = f"Inference Error for {image_path}: {e}"
        # Consider logging the full traceback here for debugging
        # import traceback
        # traceback.print_exc()

    if error_msg:
         print(f"\nWarning: {error_msg}") # Print warnings/errors during processing
         # Ensure diagnosis_dict indicates error if msg is set
         diagnosis_dict = {label: -1 for label in CHEXPERT_LABELS}

    return diagnosis_dict, raw_response_text, error_msg