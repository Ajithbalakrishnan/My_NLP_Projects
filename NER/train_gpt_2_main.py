from transformers import GPT2ForTokenClassification, GPT2TokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset
from torch.nn import functional as F
import torch

# Load CoNLL 2003 English data set
dataset = load_dataset("conll2003")

# Load GPT-2 tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
model = GPT2ForTokenClassification.from_pretrained("distilgpt2", num_labels=9)

# Prepare data for model
def prepare_data(example):
    encoded_data = tokenizer.encode_plus(example['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=512)
    labels = example['ner_tags'] + [8] * (512 - len(example['ner_tags']))  # PAD token label for GPT-2 is 8
    return encoded_data.input_ids, encoded_data.attention_mask, labels

input_ids, attention_mask, labels = zip(*dataset['train'].map(prepare_data)['train'])

input_ids = torch.tensor(input_ids)
attention_mask = torch.tensor(attention_mask)
labels = torch.tensor(labels)

# Define training arguments
args = TrainingArguments(
    "test-GPT2-ner",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
)

# Define loss function
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss = F.cross_entropy(logits.view(-1, model.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

# Create a Trainer
trainer = Trainer(
    model,
    args,
    compute_loss=compute_loss,
    train_dataset=torch.utils.data.TensorDataset(input_ids, attention_mask, labels),
)

# Train the model
trainer.train()