import os
import json
import argparse
import logging
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_model(config):
    model_path = config["model_path"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"load {model_path} to device:{device}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print("original model:\n", model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["lm_head"]
    )
    model.add_adapter(lora_config, "clm_lora")
    # model.set_adapter("clm_lora")# Once added, use set_adapter() to force a model to use the specified adapter and disable the other adapters.
    print("lora model:\n", model)
    #model.disable_adapters()
    #print("disable_adapters:\n", model)
    #model.enable_adapters()
    #print("enable_adapters:\n", model)
    return model


def create_dataset(tokenizer, config):
    def tokenize(samples):
        prompts = [
            f"### Instruction:\n{prompt}\n### Input:\n\n### Response:\n{chosen}\n"
            for prompt, chosen in zip(samples["prompt"], samples["chosen"])
        ]
        return tokenizer(prompts, truncation=True, padding="max_length", max_length=config['max_len'])

    dataset = load_dataset("liyucheng/zhihu_rlhf_3k", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"].shuffle()
    eval_dataset = dataset["test"].shuffle()
    tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['prompt', 'chosen', 'rejected'])
    tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=['prompt', 'chosen', 'rejected'])
    return tokenized_train_dataset, tokenized_eval_dataset


def create_trainingArguments(config):
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=1000,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        fp16=True,
        fp16_full_eval=True,
        dataloader_num_workers=2,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        report_to=[],
    )
    return training_args


def create_trainer(model, training_args, data_collator, train_data, eval_data, tokenizer):
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )
    return trainer


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def lora():
    parser = argparse.ArgumentParser(description="lora config")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="指定配置文件的路径，例如: config/lora.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print("config:\n", json.dumps(config, indent=4, ensure_ascii=False))

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    model = create_model(config)
    training_args = create_trainingArguments(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_data, eval_data = create_dataset(tokenizer, config)
    trainer = create_trainer(model, training_args, data_collator, train_data, eval_data, tokenizer)
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == '__main__':
    lora()
