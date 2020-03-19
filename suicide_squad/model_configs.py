import os


text= {'config':'DistilBertConfig', 'model_name':'DistilBertForQuestionAnswering', 'model_tokenizer':'DistilBertTokenizer'}

root_path='.'
model_configs={'pretrained_model_name':'bert-large-uncased-whole-word-masking-finetuned-squad',
               'tokenizer_name':'bert-base-uncased',
               'cache_dir':os.path.join(root_path,'cache_dir')
              }
