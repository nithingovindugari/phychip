"""Merge a LoRA adapter into its base and save a standalone model dir (for vLLM).
python merge.py BASE ADAPTER OUT"""
import shutil, sys, os, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel
base, adapter, out = sys.argv[1:4]
cfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
loader = AutoModelForImageTextToText if getattr(cfg, "vision_config", None) is not None else AutoModelForCausalLM
model = loader.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
import glob
for path in glob.glob(os.path.join(base, "*.py")):  # custom modeling code for remote-code models
    shutil.copy(path, out)
for name in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
    for src in (adapter, base):  # a base model without a chat template takes the adapter's
        if os.path.exists(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), os.path.join(out, name))
            break
print("MERGED", out)
