from transformers import BertTokenizer, BertForQuestionAnswering,BertConfig
import torch


class QuestionAnswering():
    def __init__(self,model_configs):
        self.model_configs=model_configs
        self.pretrained_model = BertForQuestionAnswering.from_pretrained(self.model_configs['pretrained_model_name'],
                                                             cache_dir=self.model_configs['cache_dir'])
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_configs['tokenizer_name'])
    
    def predict_pretrained(self,question,text):
        input_ids = self.tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.pretrained_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        return(answer,start_scores, end_scores)