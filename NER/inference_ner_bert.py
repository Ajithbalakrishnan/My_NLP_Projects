import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from transformers import AutoTokenizer
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoModelForTokenClassification


wnut = load_dataset("ncbi_disease")

text = wnut['test'][1]['tokens']

print(f"Classes : {wnut['test'].features['ner_tags'].feature.names}")
print("")
print("")


model = AutoModelForTokenClassification.from_pretrained("checkpoints/checkpoint-1500")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/checkpoint-1500")

# classifier = pipeline("ner", model=model, tokenizer=tokenizer)

inputs = tokenizer(" ".join(text), return_tensors="pt", padding=True)

print(f"Input Text : {text}")
print("")
print("")
with torch.no_grad():
    logits = model(**inputs).logits
    

predictions = torch.argmax(logits, dim=2)
predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

def extract_entities(tokens, predicted_labels):
    entities = []
    current_entity = {"text": "", "type": None}

    for token, label in zip(tokens, predicted_labels):
        if label.startswith("B-"):
            # Start a new entity
            if current_entity["text"]:
                entities.append(current_entity.copy())
            current_entity["text"] = token
            current_entity["type"] = label[2:]
        elif label.startswith("I-"):
            # Extend the current entity
            current_entity["text"] += " " + token
        else:
            # End the current entity
            if current_entity["text"]:
                entities.append(current_entity.copy())
                current_entity = {"text": "", "type": None}

    # Add the last entity if any
    if current_entity["text"]:
        entities.append(current_entity)

    return entities

entities = extract_entities(' '.join(text), predicted_token_class)
for entity in entities:
    print(f"Entity: {entity['text']}, Type: {entity['type']}")