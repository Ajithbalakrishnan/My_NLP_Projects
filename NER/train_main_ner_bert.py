import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import evaluate
import numpy as np

from transformers import AutoTokenizer
# from transformers import RobertaTokenizerFast
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification


model_name = "bert-base-uncased" # "roberta-base"    


tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True)#, add_prefix_space=True)

wnut = load_dataset("ncbi_disease")#'wnut_17')

tag_names = wnut["test"].features["ner_tags"].feature.names

def compute_metrics(eval_preds):
    metric = evaluate.load("ncbi_disease")#"glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_and_align_tags(records):
    # Tokenize the input words. This will break words into subtokens if necessary.
    # For instance, "ChatGPT" might become ["Chat", "##G", "##PT"].
    tokenized_results = tokenizer(records["tokens"], truncation=True, is_split_into_words=True)

    input_tags_list = []

    # Iterate through each set of tags in the records.
    for i, given_tags in enumerate(records["ner_tags"]):
        # Get the word IDs corresponding to each token. This tells us to which original word each token corresponds.
        word_ids = tokenized_results.word_ids(batch_index=i)

        previous_word_id = None
        input_tags = []

        # For each token, determine which tag it should get.
        for wid in word_ids:
            # If the token does not correspond to any word (e.g., it's a special token), set its tag to -100.
            if wid is None:
                input_tags.append(-100)
            # If the token corresponds to a new word, use the tag for that word.
            elif wid != previous_word_id:
                input_tags.append(given_tags[wid])
            # If the token is a subtoken (i.e., part of a word we've already tagged), set its tag to -100.
            else:
                input_tags.append(-100)
            previous_word_id = wid

        input_tags_list.append(input_tags)

    # Add the assigned tags to the tokenized results.
    # Hagging Face trasformers use 'labels' parameter in a dataset to compute losses.
    tokenized_results["labels"] = input_tags_list

    return tokenized_results


tokenized_wnut = wnut.map(tokenize_and_align_tags, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

id2label = dict(enumerate(tag_names))

label2id = dict(zip(id2label.values(), id2label.keys()))

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()