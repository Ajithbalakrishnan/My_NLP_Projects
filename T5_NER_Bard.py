from accelerate import Accelerator
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from datasets import load_from_disk, load_dataset, load_metric
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader


def preprocess_fn(examples,tokenizer):
    # Define your NER specific preprocessing here
    # Replace this with the logic for your chosen dataset
    inputs = tokenizer(examples["tokens"], padding="max_length", truncation=True)
    outputs = {"labels": examples["ner_tags"]}
    return inputs, outputs


def train_t5_ner(model_name, dataset_name, batch_size, epochs, eval_interval):
    # Load model and dataset
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    dataset = load_dataset(dataset_name) #load_from_disk(DatasetHub.get_url(dataset_name))
    train_data = dataset["train"]
    eval_data = dataset["validation"] if "validation" in dataset else dataset["dev"]
    test_data = dataset["test"]

    # Preprocess data           
    train_data = train_data.map(lambda examples: preprocess_fn(examples, tokenizer), batched=True)
    eval_data = eval_data.map(lambda examples: preprocess_fn(examples, tokenizer), batched=True)
    test_data = test_data.map(lambda examples: preprocess_fn(examples, tokenizer), batched=True)

    # Define and configure model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    accelerator = Accelerator()
    model, train_data, eval_data = accelerator.prepare(model, train_data, eval_data)

    # Define training dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in tqdm(range(epochs)):
        for step, batch in enumerate(tqdm(train_loader)):
            model.train()
            inputs, outputs = batch
            labels = outputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, len(tokenizer)), labels.view(-1))
            accelerator.backpropagate(loss)
            optimizer.step()
            accelerator.cleanup()

        # Evaluation and saving model
        if epoch % eval_interval == 0:
            model.eval()
            # Implement your evaluation logic here
            # Replace this with evaluation for your NER task
            # For example, calculate F1 score

            accelerator.save_model(model, f"model_epoch_{epoch}.pt")

    # Save final model
    accelerator.save_model(model, "final_model.pt")


def predict_ner(model_path, text):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    inputs = tokenizer(text, padding="max_length", truncation=True)
    outputs = model(**inputs)
    # Process and display NER predictions based on your chosen dataset format
    # Replace this with the logic for your specific NER task outputs

    print(f"Predicted NER tags for text: {text}")
    # ...


# Example usage
model_name = "t5-small"
dataset_name = "conll2003"  # Replace with your chosen dataset
batch_size = 8
epochs = 5
eval_interval = 1

train_t5_ner(model_name, dataset_name, batch_size, epochs, eval_interval)

predict_ner("final_model.pt", "This is a great restaurant in Paris.")
