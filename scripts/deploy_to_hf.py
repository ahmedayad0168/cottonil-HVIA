"""Upload this project to a Hugging Face Space (Gradio SDK).

    pip install huggingface_hub
    huggingface-cli login                      # paste a WRITE token from hf.co/settings/tokens
    python scripts/deploy_to_hf.py  YOUR_USERNAME/cottonil-ai-hub  [--private]

Uploads only what the app needs (code + artifacts + assets + README). Your raw data, notebooks and tests are NOT uploaded.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("repo_id")
ap.add_argument("--private", action="store_true", help="recommended while the data is confidential")
a = ap.parse_args()

api = HfApi()
api.create_repo(a.repo_id, repo_type="space", space_sdk="gradio", private=a.private, exist_ok=True)
api.upload_folder(
    repo_id=a.repo_id, repo_type="space", folder_path=str(ROOT),
    allow_patterns=["app.py", "requirements.txt", "README.md", "cottonil/*.py", "artifacts/**", "assets/**"],
    ignore_patterns=["**/__pycache__/**", "data/**", "notebooks/**", "tests/**"],
)
print(f"Uploaded. Space URL: https://huggingface.co/spaces/{a.repo_id}  (first build takes a few minutes)")
