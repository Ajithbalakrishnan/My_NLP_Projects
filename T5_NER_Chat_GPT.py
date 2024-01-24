
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW

# Define NER dataset
dataset_name = "conll2003"
raw_datasets = load_dataset(dataset_name)

# Split data into Train, Test, and Evaluation
train_dataset = raw_datasets["train"]
test_dataset = raw_datasets["test"]
eval_dataset = raw_datasets["validation"]

print(test_dataset)

def preprocess_data(examples, tokenizer, max_length=128):
    """
    Preprocesses NER dataset examples for T5 model.

    Args:
        examples: A single example from the dataset.
        tokenizer: The T5 tokenizer.
        max_length: Maximum token length for padding.

    Returns:
        A dictionary containing preprocessed inputs ("input_ids" and "attention_mask").
    """

    # Ensure consistent data format
    processed_inputs = {"input_ids": [], "attention_mask": []}

    for tokens in examples["tokens"]:
        # Check for non-list tokens
        if not isinstance(tokens, list):
            raise ValueError(f"Invalid token type detected: {type(tokens)}")

        # Tokenize and pad
        inputs = tokenizer(tokens, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")

        # Convert to lists (avoid unexpected conversions)
        processed_inputs["input_ids"].append(inputs["input_ids"].squeeze().tolist())
        processed_inputs["attention_mask"].append(inputs["attention_mask"].squeeze().tolist())

    # Check for consistent data format after processing
    if not all(isinstance(item, list) for item in processed_inputs.values()):
        raise ValueError("Inconsistent data format after preprocessing.")

    print(f"Type : {type(processed_inputs['input_ids'])}      Len: {len(processed_inputs['input_ids'])}   {processed_inputs['input_ids'][:10]} ")
    print(f"Type : {type(processed_inputs['attention_mask'])}      Len: {len(processed_inputs['attention_mask'])}   {processed_inputs['attention_mask'][:10]} ")

    return processed_inputs

# Define training arguments
batch_size = 2
epochs = 3
evaluation_interval = 100

# Initialize model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Prepare data
train_data = train_dataset.map(lambda examples: preprocess_data(examples, tokenizer), batched=True)
test_data = test_dataset.map(lambda examples: preprocess_data(examples, tokenizer), batched=True)
eval_data = eval_dataset.map(lambda examples: preprocess_data(examples, tokenizer), batched=True)


print("#########################################################################################")
# Create training loop
def train_model(model, train_data, eval_data, batch_size, epochs, evaluation_interval):
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    train_data = accelerator.prepare(torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True))

    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_data):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if step % evaluation_interval == 0:
                eval_metrics = evaluate_model(model, eval_data)
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}, Eval Metrics: {eval_metrics}")

    return model

# Create evaluation loop
def evaluate_model(model, eval_data):
    model.eval()
    eval_data = torch.utils.data.DataLoader(eval_data, batch_size=batch_size)
    total_loss = 0.0
    total_steps = 0

    for batch in eval_data:
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_steps += 1

    avg_loss = total_loss / total_steps
    return {"loss": avg_loss}

# Train the model
trained_model = train_model(model, train_data, eval_data, batch_size, epochs, evaluation_interval)

# Save the model
output_dir = "./ner_model"
trained_model.save_pretrained(output_dir)

# Inference function
def infer(model_path, text_input):
    loaded_model = T5ForConditionalGeneration.from_pretrained(model_path)
    loaded_tokenizer = T5Tokenizer.from_pretrained(model_path)

    inputs = loaded_tokenizer(text_input, return_tensors="pt")
    outputs = loaded_model.generate(**inputs)

    decoded_output = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Inference Result:", decoded_output)

# Example Inference
infer(output_dir, "Hugging Face is a great platform for NLP tasks.")

