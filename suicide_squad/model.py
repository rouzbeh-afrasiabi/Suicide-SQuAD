from transformers import BertTokenizer, BertForQuestionAnswering,BertConfig
import torch
import torch.nn.functional as F
import pandas as pd


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
        answer_start=torch.argmax(start_scores)
        answer_end=torch.argmax(end_scores)
        answer = all_tokens[answer_start]
        
        for i in range(answer_start + 1, answer_end + 1):
            if all_tokens[i][0:2] == '##':
                answer += all_tokens[i][2:]
            else:
                answer += ' ' + all_tokens[i]
#         answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        return(answer)
    
    def predict_pretrained_loc(self,question,text):
        input_ids = self.tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.pretrained_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return(all_tokens,torch.argmax(start_scores),(torch.argmax(end_scores)+1))
    
    def predict_pretrained_scores(self,question,text):
        input_ids = self.tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.pretrained_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return(start_scores,end_scores,all_tokens)
    
    def predict_pretrained_plot(self,question,text):
        start_scores,end_scores,all_tokens=self.predict_pretrained_scores(question,text)
        results=pd.DataFrame({'tokens':all_tokens,'start_scores':F.softmax(start_scores.flatten(),dim=0).detach().numpy()})
        results['end_scores']=F.softmax(end_scores.flatten(),dim=0).detach().numpy()
        first,last=torch.argmax(start_scores).numpy()-5,torch.argmax(end_scores).numpy()+5
        results.iloc[first:last].plot.bar(x='tokens',legend=False,figsize=(10,7),subplots=True);
        
    def predict_pretrained_score_df(self,question,text):
        start_scores,end_scores,all_tokens=self.predict_pretrained_scores(question,text)
        results=pd.DataFrame({'tokens':all_tokens,'start_scores':F.softmax(start_scores.flatten(),dim=0).detach().numpy()})
        results['end_scores']=F.softmax(end_scores.flatten(),dim=0).detach().numpy()
        return results