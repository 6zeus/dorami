import os
import json
import argparse
import logging
import torch
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, BertTokenizerFast, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_tokenizer(config):
    bertTokenizer = BertTokenizerFast.from_pretrained(config["tokenizer_name"])
    bertTokenizer.bos_token = bertTokenizer.cls_token
    bertTokenizer.eos_token = bertTokenizer.sep_token
    if bertTokenizer.pad_token is None:
        bertTokenizer.pad_token = bertTokenizer.eos_token
    return bertTokenizer


def create_model(tokenizer, config):
    model = config["model_name"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"load {model} to device:{device}")
    config = GPT2Config.from_pretrained(model, vocab_size=tokenizer.vocab_size,
                                        n_positions=config["max_len"], n_ctx=config["max_len"],
                                        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    gpt = GPT2LMHeadModel(config)
    gpt.resize_token_embeddings(len(tokenizer))
    gpt.to(device)
    return gpt


def create_dataset(tokenizer, config):
    def tokenize(sample):
        return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=config["max_len"])

    dataset = load_dataset("wiki40b", "zh-cn", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"].shuffle()  # .select(range(100000)) #暂时取100000用于测试
    eval_dataset = dataset["test"].shuffle()  # .select(range(10000)) #暂时取10000用于测试
    tokenized_train_dataset = train_dataset.map(tokenize, batched=True, num_proc=4,
                                                remove_columns=["text", 'wikidata_id', 'version_id'])
    tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True, num_proc=4,
                                              remove_columns=["text", 'wikidata_id', 'version_id'])
    return tokenized_train_dataset, tokenized_eval_dataset


def create_trainingArguments(config):
    training_args = TrainingArguments(
        output_dir=config["model_path"],
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


def pretrain():
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

    tokenizer = create_tokenizer(config)
    model = create_model(tokenizer, config)
    training_args = create_trainingArguments(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_data, eval_data = create_dataset(tokenizer, config)
    trainer = create_trainer(model, training_args, data_collator, train_data, eval_data, tokenizer)
    trainer.train()
    trainer.save_model(config["model_path"])
    tokenizer.save_pretrained(config["model_path"])


if __name__ == '__main__':
    pretrain()
