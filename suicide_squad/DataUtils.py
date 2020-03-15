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
    
data_folder=os.path.join(cwd,'data')
if(not os.path.exists(data_folder)):
    os.mkdir(data_folder)
    
original_data_folder=os.path.join(data_folder,'original')
if(not os.path.exists(original_data_folder)):
    os.mkdir(original_data_folder)
    
processed_data_folder=os.path.join(data_folder,'processed')
if(not os.path.exists(processed_data_folder)):
    os.mkdir(processed_data_folder)

# else:
#     print('Download folder exists', download_folder)


def check_file(filename, location=cwd):
    return (os.path.exists(os.path.join(location, filename)),os.path.join(location, filename))

def download_files(files, download_folder,return_paths=True):
    #Author :Rouzbeh Afrasiabi :https://github.com/rouzbeh-afrasiabi
    """
    """
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    file_paths=[]
    for file in files:
        [[_, location], ] = file.items()
        file_name = os.path.basename(location)
        file_path=os.path.join(download_folder,file_name)
        (exists, _) = check_file(file_name, download_folder)
        if exists:
            file_paths.append(file_path)
#             print (file_name, ' Already Exists.')
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
                file_paths.append(file_path)
            except:
                raise Exception('Failed')
    if(return_paths):
        return(file_paths)
                
def load_pandas_df(file_path, download_folder):
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

#     if file_split not in ["train", "dev"]:
#         raise ValueError("file_split should be either train or dev")
    
#     file_name=os.path.join(file_split+'-'+squad_version+'.json')

#     file_path = os.path.join(local_cache_path, file_name)

    with open(file_path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
    

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

                if "v2.0" in file_path:
                    is_impossible = qa["is_impossible"]

                if "train" in file_path:
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
    return output_df

def squad_to_csv(files, download_folder,return_paths=True):
    print('Downloading Files')
    file_paths=download_files(files, download_folder)
    print('Converting Files to CSV')
    new_file_paths=[]
    for file_path in file_paths:
        file_name = os.path.basename(file_path).replace('.json','.csv')
        new_path=os.path.join(original_data_folder,file_name)
        if(not  os.path.exists(new_path)):
            temp=load_pandas_df(file_path, download_folder)
            temp.to_csv(new_path,encoding='utf8')
        new_file_paths.append(new_path)
    print('Finished Downloading and Converting to CSV')
    if(return_paths):
        return(new_file_paths)

def squad_to_df(files, download_folder):
    print('Downloading Files')
    file_paths=squad_to_csv(files, download_folder)
    print('Converting Files to DataFrame')
    dataframes={}
    for file_path in file_paths:
        file_name = os.path.basename(file_path).replace('.json','.csv')
        new_path=os.path.join(original_data_folder,file_name)
        temp=pd.read_csv(new_path,index_col=[0])
        dataframes[file_name.replace('.csv','')]=temp
    print('Finished Downloading and Converting to DataFrame')
    return(dataframes)