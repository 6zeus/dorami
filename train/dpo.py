import os
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


def formatting(sample):
    prompt = f"### Instruction:\n{sample['prompt']}### Input:\n\n### Response:\n"
    chosen = f"{sample['chosen']}\n"
    rejected = f"{sample['rejected']}\n"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def create_dataset(config):
    dataset = load_dataset("liyucheng/zhihu_rlhf_3k", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset


def create_trainingArguments(config):
    training_args = DPOConfig(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=1000,
        logging_steps=1000,
        learning_rate=5e-5,
        save_total_limit=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to=[],
        max_length=512
    )
    return training_args


def create_trainer(model, ref_model, training_args, dataset, tokenizer):
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer
    )
    return trainer


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def dpo():
    parser = argparse.ArgumentParser(description="dpo config")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="指定配置文件的路径，例如: config/dpo.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print("config:\n", json.dumps(config, indent=4, ensure_ascii=False))

    model_path = config["model_path"]
    output_dir = config["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_path)
    training_args = create_trainingArguments(config)
    dataset = create_dataset(config)
    trainer = create_trainer(model, ref_model, training_args, dataset, tokenizer)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    dpo()
