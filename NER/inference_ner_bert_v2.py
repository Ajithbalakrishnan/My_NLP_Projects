from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from datasets import load_dataset


def load_model_and_tokenizer(model_path, tokenizer_path):
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def predict_entities(model, tokenizer, text):
    inputs = tokenizer(" ".join(text), return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=2)

    predicted_labels = [model.config.id2label[t.item()] for t in predicted_labels[0]]

    return predicted_labels

def post_process_output(text,predicted_labels):
    entity_spans = []
    current_entity = None

    return_dict = dict()

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

    for entity_span in entity_spans:
        entity_text = " ".join(text[entity_span["start"]:entity_span["end"]])
        # print(f"Entity: {entity_text}, Type: {entity_span['entity_type']}")

        return_dict[entity_text] = entity_span['entity_type']

    return return_dict

if __name__ == "__main__":
    model_path = "checkpoints/checkpoint-1500"
    tokenizer_path = "checkpoints/checkpoint-1500"

    ner_model, ner_tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    wnut = load_dataset("ncbi_disease")

    for i in range(10):

        input_text = wnut['test'][i]['tokens']
        
        # print("Input Text:", input_text)

        predicted_labels = predict_entities(ner_model, ner_tokenizer,input_text)

        pp_out = post_process_output(input_text,predicted_labels)

        print(pp_out)


 
