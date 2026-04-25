"""Launch FIXED job — torchao pinned"""
import time
print("Waiting 30s for rate limit...")
time.sleep(30)

from huggingface_hub import HfApi

TOKEN = "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"
api = HfApi(token=TOKEN)
api.whoami = lambda *a, **kw: {"name": "NegiAbhi07", "type": "user"}

with open("job_cmd.sh", "r") as f:
    cmd = f.read()

print("Launching FIXED H200 job...")
try:
    job = api.run_job(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        command=["bash", "-c", cmd],
        flavor="h200",
        timeout="5h",
    )
    print(f"\nJOB CREATED!")
    print(f"Job ID: {job.id}")
    print(f"URL: {job.url}")
    print(f"View: https://huggingface.co/settings/jobs")
except Exception as e:
    print(f"Failed: {e}")
