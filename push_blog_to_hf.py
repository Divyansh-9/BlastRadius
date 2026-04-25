"""
push_blog_to_hf.py
Uploads blog.md as the README of the HF Space so it appears
as the model card / blog post that judges will read.

Usage:
    HF_TOKEN=hf_xxx python push_blog_to_hf.py

The HF_TOKEN must have WRITE access to the space Idred/BlastRadius-OpenEnv.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

HF_SPACE_ID = "Idred/BlastRadius-OpenEnv"
BLOG_FILE = "blog.md"

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

token = os.environ.get("HF_TOKEN", "")
if not token:
    print("ERROR: HF_TOKEN environment variable not set.")
    print("  Set it with: $env:HF_TOKEN='hf_your_token_here'  (PowerShell)")
    print("  Then re-run:  python push_blog_to_hf.py")
    sys.exit(1)

with open(BLOG_FILE, encoding="utf-8") as f:
    blog_content = f.read()

# Prepend HF Space YAML frontmatter so it renders correctly
space_readme = f"""---
title: BlastRadius
emoji: 💥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

{blog_content}"""

api = HfApi(token=token)

print(f"Uploading blog.md to {HF_SPACE_ID} README.md ...")
api.upload_file(
    path_or_fileobj=space_readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=HF_SPACE_ID,
    repo_type="space",
    commit_message="docs: add technical blog post as Space README for hackathon judges",
)
print(f"Done! View at: https://huggingface.co/spaces/{HF_SPACE_ID}")
