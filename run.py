import logging
import re
from typing import Any, Dict, List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
import os
from pathlib import Path

from load import load_model, load_data


#################### Train #####################
class SFTDataset(Dataset):
    """SFT 训练数据集"""
    
    def __init__(self, data_df, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 准备数据
        for idx in range(len(data_df)):
            question = data_df.iloc[idx]["question"]
            answer = data_df.iloc[idx]["answer"]
            
            # 构建对话格式
            system = (
                "You are a math reasoning assistant. Solve the problem step by step. "
                "At the end, output ONLY the final numeric answer in the format "
                "\\boxed{ANSWER}"
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            
            # 使用 tokenizer 的 chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            self.data.append(text)
            
    def _normalize_answer(self, answer: str) -> str:
        """统一答案格式为 \boxed{} 格式"""
        import re
        
        # 如果已经有 \boxed{} 格式，保持不变
        if r'\boxed{' in answer:
            return answer
        
        # 尝试提取 #### 后面的数字
        hash_match = re.search(r'####\s*(\d+)', answer)
        if hash_match:
            final_answer = hash_match.group(1)
            # 如果答案不在末尾，添加到末尾
            if not answer.rstrip().endswith(f'#### {final_answer}'):
                answer = answer.rstrip() + f'\n\n\\boxed{{{final_answer}}}'
            else:
                # 替换 #### 为 \boxed{}
                answer = re.sub(r'####\s*(\d+)', r'\\boxed{\1}', answer)
            return answer
        
        # 尝试提取末尾的数字
        num_match = re.search(r'(\d+)\s*$', answer.strip())
        if num_match:
            final_answer = num_match.group(1)
            # 如果末尾没有 \boxed{}，添加
            if r'\boxed{' not in answer:
                answer = answer.rstrip() + f'\n\n\\boxed{{{final_answer}}}'
            return answer
        
        # 如果无法提取，返回原答案（让模型学习原始格式）
        return answer
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        
        # 创建 labels，将 padding 部分设为 -100（忽略损失）
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def run_train(cfg: Dict[str, Any]) -> None:
    """SFT 训练主函数"""
    logging.info("Starting SFT training")
    
    # 加载配置
    model_path = cfg["model"]["path"]
    train_path = cfg["data"]["train_path"]
    train_params = cfg.get("train", {})
    
    epochs = train_params.get("epochs", 3)
    batch_size = train_params.get("batch_size", 2)
    learning_rate = train_params.get("learning_rate", 3e-4)
    gradient_accumulation_steps = train_params.get("gradient_accumulation_steps", 1)
    max_length = train_params.get("max_length", 2048)
    save_steps = train_params.get("save_steps", 500)
    eval_steps = train_params.get("eval_steps", None)
    output_dir = train_params.get("output_dir", "./checkpoints")
    use_lora = train_params.get("use_lora", True)
    lora_r = train_params.get("lora_r", 8)
    lora_alpha = train_params.get("lora_alpha", 16)
    lora_dropout = train_params.get("lora_dropout", 0.05)
    
    logging.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}, "
                 f"lr={learning_rate}, use_lora={use_lora}")
    
    # 加载模型和 tokenizer
    logging.info(f"Loading model from {model_path}")
    model, tokenizer, device = load_model(model_path)
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 应用 LoRA（如果启用）
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            logging.info("Applying LoRA for efficient fine-tuning")
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        except ImportError:
            logging.warning("peft not installed, training without LoRA. Install with: pip install peft")
            use_lora = False
    
    # 加载训练数据
    logging.info(f"Loading training data from {train_path}")
    train_df = load_data(train_path)
    logging.info(f"Loaded {len(train_df)} training samples")
    
    # 创建数据集
    train_dataset = SFTDataset(train_df, tokenizer, max_length=max_length)
    
    # 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows 上建议设为 0
    )
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 学习率调度器
    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    model.train()
    global_step = 0
    total_loss = 0.0
    
    logging.info("Starting training loop...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            ncols=100
        )
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            total_loss += loss.item() * gradient_accumulation_steps
            
            # 梯度累积
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 更新进度条
                avg_loss = total_loss / global_step
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                
                # 保存检查点
                if save_steps > 0 and global_step % save_steps == 0:
                    checkpoint_dir = output_path / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(exist_ok=True)
                    
                    if use_lora:
                        model.save_pretrained(checkpoint_dir)
                    else:
                        model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logging.info(f"Saved checkpoint at step {global_step} to {checkpoint_dir}")
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1}/{epochs} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # 保存最终模型
    final_model_dir = output_path / "final_model"
    final_model_dir.mkdir(exist_ok=True)
    
    if use_lora:
        # 保存 LoRA 权重
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logging.info(f"LoRA weights saved to {final_model_dir}")
        
        # 如果需要合并 LoRA 权重到基础模型
        if train_params.get("merge_lora", False):
            try:
                from peft import PeftModel
                logging.info("Merging LoRA weights into base model...")
                # 重新加载基础模型（不使用 LoRA）
                base_model, _, _ = load_model(model_path)
                # 加载 LoRA 权重
                peft_model = PeftModel.from_pretrained(base_model, final_model_dir)
                # 合并并卸载
                merged_model = peft_model.merge_and_unload()
                # 保存合并后的模型
                merged_dir = final_model_dir / "merged"
                merged_dir.mkdir(exist_ok=True)
                merged_model.save_pretrained(merged_dir)
                tokenizer.save_pretrained(merged_dir)
                logging.info(f"Merged model saved to {merged_dir}")
            except Exception as e:
                logging.warning(f"Failed to merge LoRA weights: {e}")
    else:
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
    
    logging.info(f"Final model saved to {final_model_dir}")
    logging.info("Training finished")

#################### Evaluation #####################
def extract_answer(model_output: str):
    """从模型输出中提取最终答案"""
    boxed_pattern = re.compile(r'\\boxed\{(\d+)\}')
    boxed_match = boxed_pattern.search(model_output)
    if boxed_match:
        return int(boxed_match.group(1))
    
    hash_pattern = re.compile(r'####\s*(\d+)')
    hash_match = hash_pattern.search(model_output)
    if hash_match:
        return int(hash_match.group(1))
    pattern = re.compile(r'(\d+)\s*$')
    match = pattern.search(model_output)
    if match:
        return int(match.group(1))
    return None


def build_gsm8k_prompt(question: str, tokenizer = None) -> str:
    """
    构造适合 LLaMA 类指令模型的 GSM8K prompt。
    要求模型用逐步推理，并把最终数字答案放在 \\boxed{} 中（以及/或 `####` 之后），
    以便 `extract_answer` 能稳定解析。
    """
    system = (
        "You are a math reasoning assistant. Solve the problem step by step. "
        "At the end, output ONLY the final numeric answer in the format "
        "\\boxed{ANSWER}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
        ]
    if tokenizer:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        return text
    return f"<s>[SYSTEM] {system}\n[USER] {question}\n[ASSISTANT]"


def calculate_gsm8k_accuracy(
    model,
    tokenizer,
    test_df,
    device: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> float:
    """在 GSM8K 测试集上进行批量推理并计算准确率"""
    model.eval()

    correct = 0
    total = len(test_df)

    # DataFrame -> list of dict，便于遍历
    records = test_df.to_dict(orient="records")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    with torch.no_grad():   
        for i in tqdm(range(0, total, batch_size), desc="Evaluation on GMSK8 Test Data", leave=True, ncols=100):
            batch_samples = records[i : i + batch_size]
            questions = [s["question"] for s in batch_samples]
            ground_truths = [
                int(s["answer"].split("####")[-1].strip().replace(",", "")) for s in batch_samples
            ]

            prompts = [build_gsm8k_prompt(q, tokenizer) for q in questions]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if i == 0:
                print(decoded[0])
                print(decoded[1])
                print(ground_truths[0])
                print(ground_truths[1])
            for pred_text, gt in zip(decoded, ground_truths):
                pred_answer = extract_answer(pred_text)
                if pred_answer is None:
                    continue
                if pred_answer == gt:
                    correct += 1
            if i % 10 == 0:
                print(f"current accuracy in {i}'s batch is {correct / total}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def run_evaluate(cfg: Dict[str, Any]) -> None:
    logging.info("Starting evaluation")
    if cfg['api']:
        model_name = cfg['api']
        import os
        from openai import OpenAI
        data_path = cfg['data']['eval_path']
        data = load_data(data_path)
        logging.info(f"Running GSM8K evaluation on online API{model_name}")
        correct = 0
        total = len(data)
        system = (
            "You are a math reasoning assistant. Solve the problem step by step. "
            "At the end, output ONLY the final numeric answer in the format "
            "\\boxed{ANSWER}"
        )
        for i in range(total):
            question = data.iloc[i]["question"]
            ground_truth = int(data.iloc[i]["answer"].split("####")[-1].strip().replace(",", ""))
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': question}
                ]
            )
            raw_output = completion.choices[0].message.content
            output = extract_answer(raw_output)
            if output == ground_truth:
                correct += 1
            if i == 0:
                print(raw_output)
            if i % 10 == 0:
                print(f"current correct counts in {i}'s batch is {correct}")
        accuracy = correct / total if total > 0 else 0.0
        print(f"Test accuracy on GSM8K: {accuracy:.4f}")
        logging.info(f"Test accuracy on GSM8K: {accuracy:.4f}")
        logging.info("Evaluation finished")
    else:
        eval_params = cfg.get("evaluate", {})
        logging.info("Eval params: %s", eval_params)
        data_path = cfg['data']['eval_path']
        model_path = cfg['model']['path']
        data = load_data(data_path)
        model, tokenizer, device = load_model(model_path)

        batch_size = eval_params.get("batch_size", 8)
        max_new_tokens = eval_params.get("max_new_tokens", 128)

        logging.info(
            f"Running GSM8K evaluation on device={device}, "
            f"batch_size={batch_size}, max_new_tokens={max_new_tokens}"
        )

        accuracy = calculate_gsm8k_accuracy(
            model=model,
            tokenizer=tokenizer,
            test_df=data,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        print(f"Test accuracy on GSM8K: {accuracy:.4f}")
        logging.info(f"Test accuracy on GSM8K: {accuracy:.4f}")
        logging.info("Evaluation finished")

#################### Prediction #####################
def run_predict(cfg: Dict[str, Any]) -> None:
    logging.info("Starting prediction")
    predict_params = cfg.get("predict", {})
    logging.info("Predict params: %s", predict_params)
    # TODO: replace with real prediction logic
    logging.info("Prediction finished")

