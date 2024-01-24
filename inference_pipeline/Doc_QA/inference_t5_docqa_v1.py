import os
# import evaluate
import logging

from datasets import load_dataset
# from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# rouge = evaluate.load("rouge")

class DocQa:
    def __init__(self):
        model_path = "Doc_QA/saved_model/checkpoint-9500"
        self.model_loaded, self.tokenizer_loaded = self.load_model_and_tokenizer(model_path, model_path)
        print("DocQA model loaded successfully")

    # def evaluate_performance(self, preds,labels):
    #     return rouge.compute(predictions=preds, references=labels, use_stemmer=True)

    def load_model_and_tokenizer(self, model_path, tokenizer_path):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer

    def inference_main(self, model_loaded, tokenizer, text_input, qn_input):
        # summarizer = pipeline("summarization", model=model_loaded)
        # pred = summarizer(text)
        text = "question: "+ qn_input +"  context: "+text_input
        print("----------> ",text)
        inputs = tokenizer(text, return_tensors="pt").input_ids
        outputs = model_loaded.generate(inputs, max_new_tokens=300, do_sample=False)
        final_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return final_out
    
    def run_application(self, text_input, qn_input):
        
        prediction = self.inference_main(self.model_loaded, self.tokenizer_loaded,text_input, qn_input)

        return prediction

# if __name__ == "__main__":
#     model_path = "saved_model/checkpoint-9500"

#     dataset = load_dataset('squad')
#     print("Pubmed Dataset : ",dataset)

#     sample_text = dataset['validation'][100]

#     model_loaded, tokenizer_loaded = load_model_and_tokenizer(model_path, model_path)

#     prediction = inference_main(model_loaded, tokenizer_loaded,sample_text)

#     print(f"Prediction : {prediction}")

    # evaluation_metric_out = evaluate_performance(prediction, sample_text['abstract'])

    # print("Eval Mtric : ", evaluation_metric_out)


