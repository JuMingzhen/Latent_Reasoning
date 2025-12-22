import logging
import re
from typing import Any, Dict
from tqdm import tqdm
import torch

from load import load_model, load_data


#################### Train #####################
def run_train(cfg: Dict[str, Any]) -> None:
    logging.info("Starting training")
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    train_params = cfg.get("train", {})
    logging.info("Model: %s", model)
    logging.info("Data: %s", data)
    logging.info("Train params: %s", train_params)
    # TODO: replace with real training logic
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

