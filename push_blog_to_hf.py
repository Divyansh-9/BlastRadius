"""
push_blog_to_hf.py  —  reads HF_TOKEN from .env and uploads blog.md to HF Space
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

# Load .env from parent directory
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
env_path = os.path.normpath(env_path)
token = ""
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                token = line.split("=", 1)[1].strip()
                break

token = os.environ.get("HF_TOKEN", token)
if not token or token == "hf_your-token-here":
    print("ERROR: No valid HF_TOKEN found in .env or environment.")
    sys.exit(1)

print(f"Using HF token: {token[:8]}...")

from huggingface_hub import HfApi

HF_SPACE_ID = "Idred/BlastRadius-OpenEnv"
blog_path = os.path.join(os.path.dirname(__file__), "blog.md")

with open(blog_path, encoding="utf-8") as f:
    blog_content = f.read()

space_readme = f"""---
title: BlastRadius
emoji: \U0001f4a5
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

{blog_content}"""

api = HfApi(token=token)
print(f"Uploading blog.md to {HF_SPACE_ID} as README.md ...")
api.upload_file(
    path_or_fileobj=space_readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=HF_SPACE_ID,
    repo_type="space",
    commit_message="docs: add technical blog post as Space README for hackathon judges",
)
print(f"Done! View at: https://huggingface.co/spaces/{HF_SPACE_ID}")
