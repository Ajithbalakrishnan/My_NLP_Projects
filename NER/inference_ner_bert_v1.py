import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForTokenClassification

# Load the test dataset
wnut = load_dataset("ncbi_disease")
text = wnut['test'][1]['tokens']

# Load the pretrained model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("checkpoints/checkpoint-1500")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/checkpoint-1500")

# Tokenize the input text
inputs = tokenizer(" ".join(text), return_tensors="pt", padding=True)

# Make predictions
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted labels
predictions = torch.argmax(logits, dim=2)
predicted_labels = [model.config.id2label[t.item()] for t in predictions[0]]

# Postprocess the output
entity_spans = []
current_entity = None

for i, (word, label) in enumerate(zip(text, predicted_labels)):
    if label.startswith('B-'):
        if current_entity:
            entity_spans.append(current_entity)
        current_entity = {"start": i, "end": i + 1, "entity_type": label[2:]}
    elif label.startswith('I-'):
        if current_entity:
            current_entity["end"] = i + 1
    else:
        if current_entity:
            entity_spans.append(current_entity)
            current_entity = None

# Print the extracted entities
for entity_span in entity_spans:
    entity_text = " ".join(text[entity_span["start"]:entity_span["end"]])
    print(f"Entity: {entity_text}, Type: {entity_span['entity_type']}")
