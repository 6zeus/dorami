import torch
from transformers import GPT2LMHeadModel, BertTokenizerFast, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

max_len=512
model_path="../model/dorami-pre"
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"use device:{device}")

def create_tokenizer():
    bertTokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    if bertTokenizer.pad_token is None:
        bertTokenizer.pad_token = bertTokenizer.eos_token
    return bertTokenizer

def create_model(tokenizer):
    config = GPT2Config.from_pretrained("gpt2", vocab_size=tokenizer.vocab_size,n_positions=max_len)
    gpt = GPT2LMHeadModel(config)
    gpt.resize_token_embeddings(len(tokenizer))
    gpt.to(device)
    return gpt

def create_dataset(tokenizer):
    def tokenize(sample):
        return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=max_len)
    dataset = load_dataset("wiki40b", "zh-cn", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"].shuffle().select(range(100000)) #暂时取100000用于测试
    eval_dataset = dataset["test"].shuffle().select(range(10000)) #暂时取10000用于测试
    tokenized_train_dataset = train_dataset.map(tokenize, batched=True, num_proc=4,remove_columns=["text", 'wikidata_id', 'version_id'])
    tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True, num_proc=4,remove_columns=["text", 'wikidata_id', 'version_id'])
    return tokenized_train_dataset,tokenized_eval_dataset

tokenizer = create_tokenizer()
model = create_model(tokenizer)
train_data, eval_data=create_dataset(tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir=model_path,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=1000,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

