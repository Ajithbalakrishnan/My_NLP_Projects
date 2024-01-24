import os
import evaluate

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

rouge = evaluate.load("rouge")

def evaluate_performance(preds,labels):
    return rouge.compute(predictions=preds, references=labels, use_stemmer=True)

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def inference_main(model_loaded, tokenizer, text ):
    # summarizer = pipeline("summarization", model=model_loaded)
    # pred = summarizer(text)
    text = "summarize: " + text
    inputs = tokenizer(text, return_tensors="pt").input_ids
    outputs = model_loaded.generate(inputs, max_new_tokens=300, do_sample=False)
    final_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return final_out


if __name__ == "__main__":
    model_path = "saved_model/checkpoint-6500"

    pubmed = load_dataset("ccdv/pubmed-summarization")
    print("Pubmed Dataset : ",pubmed)

    sample_text = pubmed['test'][100]
    
    model_loaded, tokenizer_loaded = load_model_and_tokenizer(model_path, model_path)

    prediction = inference_main(model_loaded, tokenizer_loaded,sample_text['article'])

    print(f"Prediction : {prediction}")

    # evaluation_metric_out = evaluate_performance(prediction, sample_text['abstract'])

    # print("Eval Mtric : ", evaluation_metric_out)


