import os
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def formatting_prompt(sample):
    instruction = sample["INSTRUCTION"]
    input = sample.get("INPUT", "")
    response = sample["RESPONSE"]
    prompt = f"### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n{response}\n"
    return prompt


def create_dataset(config):
    dataset = load_dataset("wangrui6/Zhihu-KOL", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset


def create_trainingArguments(config):
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=5000,
        logging_steps=2000,
        learning_rate=5e-5,
        save_total_limit=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to=[],
        max_seq_length=512
    )
    return training_args


def create_trainer(model, training_args, data_collator, dataset, formatting_func):
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_func,
    )
    return trainer


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def sft():
    parser = argparse.ArgumentParser(description="pretrain config")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="指定配置文件的路径，例如: config/pretrain.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print("config:\n", json.dumps(config, indent=4, ensure_ascii=False))

    model_path = config["model_path"]
    output_dir = config["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    training_args = create_trainingArguments(config)
    response_template = "### Response:\n"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
    dataset = create_dataset(config)
    trainer = create_trainer(model, training_args, data_collator, dataset, formatting_prompt)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    sft()
