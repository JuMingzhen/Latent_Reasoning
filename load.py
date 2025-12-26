import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def load_model(path: str, peft = False, lora_path = None):
    """
    加载本地或 HuggingFace 上的因果语言模型和 tokenizer，自动放到 GPU（如可用）。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float16 if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
    )
    if peft:
        # 加载 LoRA 适配器到原始模型
        peft_model = PeftModel.from_pretrained(model, lora_path)
        model = peft_model.merge_and_unload()
    # 如果没有使用 device_map（CPU 情况），手动放到 device
    if device != "cuda":
        model.to(device)

    return model, tokenizer, device


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(
        path,
        engine="pyarrow",
    )
    return df
