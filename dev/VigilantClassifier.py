import json
import argparse
import yaml
import sys
from jinja2 import Environment, FileSystemLoader
import Common as common
import tiktoken
from SharedLogger import CustomLogger
from typing import final
from openai import OpenAI
from PromptGen import PromptGen
import os
logger = CustomLogger().get_logger()

parser = argparse.ArgumentParser(description='Vigilat Social Tracker (Description to be added)')
parser.add_argument('--config', type=str, help='Path to the YAML config file')
args = parser.parse_args()
CONFIG_DIR: final = args.config

class VigilantClassifier():
    def __init__(self,config_dir = CONFIG_DIR):
        self.config = common.OpenYaml(config_dir,'classifier')
        self.api_key = os.getenv('OPENAIKEY')
        self.model = self.config['model']
        self.client = OpenAI(api_key=self.api_key)
        self.PromptGet = PromptGen(config_path=config_dir)

    def ModelResponse(self,prompt):
        return self.model(prompt)
    
    def GetReport(self):
        prompt = self.PromptGet.ConstructPrompt()

        history = common.OpenJson(self.config['history_path'])
        history.append({'role': 'user', 'content': prompt})


        response =  self.client.chat.completions.create(
            model = self.model,
            messages = history,
            temperature = 0.3
        )
        #print(response)
        return response.choices[0].message.content

if __name__ == "__main__":
    vc = VigilantClassifier()
    print(vc.GetReport())
