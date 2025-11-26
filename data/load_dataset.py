from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
import os

def load_hf_dataset():
    """
    Loads the Arabic AI-Generated Abstracts dataset from HuggingFace.
    Requires an HF token stored in a .env file.

    """

    # Load environment variables
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN not found in .env file.")

    # Authenticate to Hugging Face
    login(token=hf_token)

    # Load dataset
    dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")
    return dataset
