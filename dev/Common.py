import json
import yaml
import sys
from SharedLogger import CustomLogger
logger = CustomLogger().get_logger()
import tiktoken
from jinja2 import Environment 
from jinja2 import FileSystemLoader
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


def Tokenizer(text):
    return tokenizer.encode(text=text)


def JinjaRender(template_path,posts,query_=''):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_path)
    if query_:
        rendered_prompt = template.render(posts=posts,query_=query_)
    else:
        rendered_prompt = template.render(posts=posts)
    logger.info(f'Prompt length in tokens: {len(Tokenizer(rendered_prompt))}')
    return rendered_prompt


    