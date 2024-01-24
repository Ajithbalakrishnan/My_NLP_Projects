import os
import evaluate
import numpy as np

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


billsum = load_dataset("sumedh/MeQSum", split="ca_test")

billsum = billsum.train_test_split(test_size=0.2)

print("billsum Dataset : ",billsum)

example = billsum["train"][0]
# for key in example:
#     print("A key of the example: \"{}\"".format(key))
#     print("The value corresponding to the key-\"{}\"\n \"{}\"".format(key, example[key]))


tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# tokenized_text = tokenizer(example['text'])
# for key in tokenized_text:
#     print(key)
#     print(tokenized_text[key])


def preprocess_function(examples):
    # Prepends the string "summarize: " to each document in the 'text' field of the input examples.
    # This is done to instruct the T5 model on the task it needs to perform, which in this case is summarization.
    inputs = ["summarize: " + doc for doc in examples["text"]]

    # Tokenizes the prepended input texts to convert them into a format that can be fed into the T5 model.
    # Sets a maximum token length of 1024, and truncates any text longer than this limit.
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Tokenizes the 'summary' field of the input examples to prepare the target labels for the summarization task.
    # Sets a maximum token length of 128, and truncates any text longer than this limit.
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    # Assigns the tokenized labels to the 'labels' field of model_inputs.
    # The 'labels' field is used during training to calculate the loss and guide model learning.
    model_inputs["labels"] = labels["input_ids"]

    # Returns the prepared inputs and labels as a single dictionary, ready for training.
    return model_inputs


tokenized_billsum = billsum.map(preprocess_function, batched=True)


print(tokenized_billsum['test'][0].keys())

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")


rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    # Unpacks the evaluation predictions tuple into predictions and labels.
    predictions, labels = eval_pred

    # Decodes the tokenized predictions back to text, skipping any special tokens (e.g., padding tokens).
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replaces any -100 values in labels with the tokenizer's pad_token_id.
    # This is done because -100 is often used to ignore certain tokens when calculating the loss during training.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decodes the tokenized labels back to text, skipping any special tokens (e.g., padding tokens).
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Computes the ROUGE metric between the decoded predictions and decoded labels.
    # The use_stemmer parameter enables stemming, which reduces words to their root form before comparison.
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Calculates the length of each prediction by counting the non-padding tokens.
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    # Computes the mean length of the predictions and adds it to the result dictionary under the key "gen_len".
    result["gen_len"] = np.mean(prediction_lens)

    # Rounds each value in the result dictionary to 4 decimal places for cleaner output, and returns the result.
    return {k: round(v, 4) for k, v in result.items()}



# training_args = Seq2SeqTrainingArguments(
#     output_dir="saved_model",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=4,
#     predict_with_generate=True,
#     fp16=True,
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_billsum["train"],
#     eval_dataset=tokenized_billsum["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

# trainer.save_model("saved_model")



# Inference Step

text = billsum['test'][100]['text']
text = "summarize: " + text
print(text)



# Inference Method -1
summarizer = pipeline("summarization", model="saved_model")
pred = summarizer(text)
# print(pred)


# Inference Method -2 
# # tokenizer = AutoTokenizer.from_pretrained("saved_model")
# inputs = tokenizer(text, return_tensors="pt").input_ids
# model = AutoModelForSeq2SeqLM.from_pretrained("saved_model")
# outputs = model.generate(inputs, max_new_tokens=200, do_sample=False)
# final_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("Final output : ", final_out)



# Evaluation step

print(pred[0]['summary_text'])

preds = [pred[0]['summary_text']]

labels = [billsum['test'][100]['summary']]

print(rouge.compute(predictions=preds, references=labels, use_stemmer=True))
