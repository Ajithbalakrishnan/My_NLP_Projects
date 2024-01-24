import torch
from datasets import load_dataset
from transformers import  DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer

# Loading train and test datasets
data = load_dataset('squad')
 # medmcqa
# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


# Defining a function to process data
def process_data_to_model_inputs(example):
    inputs = tokenizer("question: "+example['question']+"  context: "+example['context'], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    # Setup the tokenizer for targets
    target = " ".join(example['answers']['text'])
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    model_inputs = {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "labels": labels["input_ids"][0]
    }

    return model_inputs

train_dataset = data["train"].map(process_data_to_model_inputs)
valid_dataset = data["validation"].map(process_data_to_model_inputs)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")

# Define training arguments
# args = TrainingArguments("test_trainer", per_device_train_batch_size=1, per_device_eval_batch_size=1)
args = TrainingArguments( 
                        output_dir="saved_model",
                        evaluation_strategy="epoch",
                        learning_rate=2e-5,
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=4,
                        weight_decay=0.01,
                        save_total_limit=3,
                        num_train_epochs=4,
                        fp16=True,
                        )
# Define the trainer
trainer = Trainer(model=model, 
                  args=args, 
                  train_dataset=train_dataset, 
                  eval_dataset=valid_dataset,
                  tokenizer=tokenizer,
                  data_collator=data_collator,
                  )

# Train the model
trainer.train()