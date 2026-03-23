
from huggingface_hub import HfApi, create_repo
import os

HF_TOKEN = os.getenv("HF_TOKEN")
repo_id = "kalrap/predictive-maintenance-app"

api = HfApi(token=HF_TOKEN)

create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True
)

api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["*.gdoc", "*.pdf", "*.ipynb"]
)

print("Deployment successful!")
