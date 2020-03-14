import os
import sys
import requests
import shutil
import simplejson as json
import pandas as pd

files = [
        {"v1.1-train": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json"},
        {"v1.1-dev": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"},
        {"v2.0-train": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v2.0.json"},
        {"v2.0-dev": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json"},
    ]

cwd = os.getcwd()
cwd = str(os.getcwd())
sys.path.append(cwd)
sys.path.insert(0, cwd)

download_folder=os.path.join(cwd,'download')
if(not os.path.exists(download_folder)):
    os.mkdir(download_folder)
# else:
#     print('Download folder exists', download_folder)


def check_file(filename, location=cwd):
    return (os.path.exists(os.path.join(location, filename)),os.path.join(location, filename))

def download_files(files, download_folder):
    #Author :Rouzbeh Afrasiabi :https://github.com/rouzbeh-afrasiabi
    """
    """
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    
    for file in files:
        [[_, location], ] = file.items()
        file_name = os.path.basename(location)
        (exists, _) = check_file(file_name, download_folder)
        if exists:
            print (file_name, ' Already Exists.')
            pass
        else:
            print ('*** Downloading : ', file_name)
            try:
                r = requests.get(location, auth=('usrname', 'password'
                                 ), verify=False, stream=True)
                r.raw.decode_content = True
                with open(os.path.join(download_folder, file_name), 'wb'
                          ) as f:
                    shutil.copyfileobj(r.raw, f)
            except:
                raise Exception('Failed')
                
def load_pandas_df(local_cache_path="download", squad_version="v1.1", file_split="train"):
    #https://github.com/microsoft/nlp-recipes/blob/master/utils_nlp/dataset/squad.py
    # Copyright (c) Microsoft Corporation. All rights reserved.
    # Licensed under the MIT License.
    
    """
    
    Loads the SQuAD dataset in pandas data frame.
    Args:
        local_cache_path (str, optional): Path to load the data from. If the file doesn't exist,
            download it first. Defaults to the current directory.
        squad_version (str, optional): Version of the SQuAD dataset, accepted values are: 
            "v1.1" and "v2.0". Defaults to "v1.1".
        file_split (str, optional): Dataset split to load, accepted values are: "train" and "dev".
            Defaults to "train".
    """

    if file_split not in ["train", "dev"]:
        raise ValueError("file_split should be either train or dev")
    
    file_name=os.path.join(file_split+'-'+squad_version+'.json')

    file_path = os.path.join(local_cache_path, file_name)

    with open(file_path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
    
    train_data_list = [item for topic in input_data['data'] for item in topic['paragraphs'] ]

    paragraph_text_list = []
    question_text_list = []
    answer_start_list = []
    answer_text_list = []
    qa_id_list = []
    is_impossible_list = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                answer_offset = None
                is_impossible = False

                if squad_version == "v2.0":
                    is_impossible = qa["is_impossible"]

                if file_split == "train":
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer."
                        )
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                    else:
                        orig_answer_text = ""
                else:
                    if not is_impossible:
                        orig_answer_text = []
                        answer_offset = []
                        for answer in qa["answers"]:
                            orig_answer_text.append(answer["text"])
                            answer_offset.append(answer["answer_start"])
                    else:
                        orig_answer_text = ""

                paragraph_text_list.append(paragraph_text)
                question_text_list.append(question_text)
                answer_start_list.append(answer_offset)
                answer_text_list.append(orig_answer_text)
                qa_id_list.append(qas_id)
                is_impossible_list.append(is_impossible)

    output_df = pd.DataFrame(
        {
            "doc_text": paragraph_text_list,
            "question_text": question_text_list,
            "answer_start": answer_start_list,
            "answer_text": answer_text_list,
            "qa_id": qa_id_list,
            "is_impossible": is_impossible_list,
        }
    )
    output_df.name=file_name
    return output_df

