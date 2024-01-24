
from Doc_QA.inference_t5_docqa_v1 import DocQa
from NER.inference_ner_bert_v2 import NERinfer
from Summarization.Infer_t5_for_summary import SummaryGenerator

 
class Api:
    
    def __init__(self):
        self.docqa_instance = DocQa()
        self.nre_instance = NERinfer()
        self.summary_generator = SummaryGenerator()

    def run_docqa(self,text_input, qn_input):
        return self.docqa_instance.run_application(text_input, qn_input)
    
    def run_ner(self, text_input):
        return self.nre_instance.run_application(text_input)
    
    def run_summarygenerator(self, text_input):
        return self.summary_generator.run_application(text_input)