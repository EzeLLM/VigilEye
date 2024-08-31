import json
import yaml
import sys
from SharedLogger import CustomLogger
logger = CustomLogger().get_logger()

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

