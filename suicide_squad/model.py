"""
Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Copyright (c) 2019 Rouzbeh Afrasiabi
https://github.com/rouzbeh-afrasiabi/
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""



from transformers import BertTokenizer, BertForQuestionAnswering,DistilBertForQuestionAnswering, BertConfig
import torch
import torch.nn.functional as F
import pandas as pd
import re


class QuestionAnswering():
    def __init__(self,model_configs):
        self.model_configs=model_configs
        self.pretrained_model = BertForQuestionAnswering.from_pretrained(self.model_configs['pretrained_model_name'],
                                                             cache_dir=self.model_configs['cache_dir'],output_attentions=True)
        
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_configs['tokenizer_name'])
    
    def predict_pretrained(self,question,text):
        input_ids = self.tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores,_ = self.pretrained_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
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
        marked_text=' '.join(all_tokens[:answer_start]) +'<b> '+answer+' </b>'+\
        ' '.join(all_tokens[answer_end+1:])
        marked_text=marked_text.replace(' ##','')
        marked_text=re.search(r'\[SEP\].*\[SEP\]', marked_text)[0]
        marked_text=re.sub(r'\s*\[SEP\]\s*','',marked_text)
        return(answer,marked_text)
    
    def predict_pretrained_loc(self,question,text):
        input_ids = self.tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores,_= self.pretrained_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return(all_tokens,torch.argmax(start_scores),(torch.argmax(end_scores)+1))
    
    def predict_pretrained_scores(self,question,text):
        input_ids = self.tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores,_= self.pretrained_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return(start_scores,end_scores,all_tokens)
    
    def predict_pretrained_plot(self,question,text,figsize=(5,5),layout=(1,2)):
        start_scores,end_scores,all_tokens=self.predict_pretrained_scores(question,text)
        results=pd.DataFrame({'tokens':all_tokens,'start_scores':F.softmax(start_scores.flatten(),dim=0).detach().numpy()})
        results['end_scores']=F.softmax(end_scores.flatten(),dim=0).detach().numpy()
        first,last=torch.argmax(start_scores).numpy()-5,torch.argmax(end_scores).numpy()+5
        ax=results.iloc[first:last].plot.bar(x='tokens',legend=False,figsize=figsize,subplots=True,layout=layout);
        fig,ax=ax.flatten()
        return fig
        
    def predict_pretrained_score_df(self,question,text):
        start_scores,end_scores,all_tokens=self.predict_pretrained_scores(question,text)
        results=pd.DataFrame({'tokens':all_tokens,'start_scores':F.softmax(start_scores.flatten(),dim=0).detach().numpy()})
        results['end_scores']=F.softmax(end_scores.flatten(),dim=0).detach().numpy()
        return results
    
    def predict_pretrained_attn(self,question,text):
        input_ids = self.tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        _, _,attn = self.pretrained_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        return torch.cat(attn).detach().numpy(),all_tokens