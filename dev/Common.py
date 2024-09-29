import json
import yaml
import sys
from SharedLogger import CustomLogger
logger = CustomLogger().get_logger()
import tiktoken
from jinja2 import Environment 
from jinja2 import FileSystemLoader
import os
import torch

tokenizer = tiktoken.get_encoding('cl100k_base')

def OpenYaml(path,key=''):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    if key:
        return config[key]
    else:
        return config

def OpenJson(path):
    try:
        with open(path, 'r') as file:
            posts = json.load(file)
        return posts
    except:
        logger.error("Error While opening json file in VigilantClassifier. File does not exist? Aborting")
        sys.exit()

def DumpToJson(path, data):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def OpenTxt(path):
    with open(path, 'r') as file:
        return file.read()

def Tokenizer(text):
    return tokenizer.encode(text=text)


def JinjaRender(template_path,input,mode='default'):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_path)
    if mode == 'default':
        rendered_prompt = template.render(contents=input)
    elif mode == 'per_user':
        rendered_prompt = template.render(contents=convert_dict_to_list(input))
    return rendered_prompt


def convert_dict_to_list(user_reports):
    text_representation = []
    for user, report in user_reports.items():
        text_representation.append(f"User: {user}\nReport: {report}\n\n")
    return text_representation



def DumpToText(text, path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)
    
    print(f"Text successfully written to {path}")    

def EndsWith(text: str, format_list: list):
    return text.endswith(tuple(format_list))


def Device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    
    else:
        return torch.device("cpu")