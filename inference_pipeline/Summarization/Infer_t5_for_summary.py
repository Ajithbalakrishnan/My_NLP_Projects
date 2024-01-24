import os
# import evaluate

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# rouge = evaluate.load("rouge")

# def evaluate_performance(preds,labels):
#     return rouge.compute(predictions=preds, references=labels, use_stemmer=True)

class SummaryGenerator:
    def __init__(self):
        model_path = "Summarization/saved_model/checkpoint-6500"
        self.model_loaded, self.tokenizer_loaded = self.load_model_and_tokenizer(model_path, model_path)
        print("Summary Generator model loaded successfully")

    def load_model_and_tokenizer(self, model_path, tokenizer_path):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer

    def inference_main(self, model_loaded, tokenizer, text ):
        # summarizer = pipeline("summarization", model=model_loaded)
        # pred = summarizer(text)
        text = "summarize: " + text
        inputs = tokenizer(text, return_tensors="pt").input_ids
        outputs = model_loaded.generate(inputs, max_new_tokens=300, do_sample=False)
        final_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return final_out
    
    def run_application(self, text_input):
        prediction = self.inference_main(self.model_loaded, self.tokenizer_loaded,text_input)
        return prediction


    # evaluation_metric_out = evaluate_performance(prediction, sample_text['abstract'])
    # print("Eval Mtric : ", evaluation_metric_out)


