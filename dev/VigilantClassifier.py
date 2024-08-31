import argparse
from jinja2 import Environment, FileSystemLoader
import Common as common
from SharedLogger import CustomLogger
from typing import final
from openai import OpenAI
from PromptGen import PromptGen
import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

logger = CustomLogger().get_logger()


class VigilantClassifier():
    def __init__(self,config_dir, input = False):
        self.config = common.OpenYaml(config_dir,'classifier')
        self.api_key = os.getenv('OPENAIAPIKEY')
        self.model = self.config['model']
        self.client = OpenAI(api_key=self.api_key)
        self.input = input
        self.PromptGet = PromptGen(config_path=config_dir,input=self.input)

    def ModelResponse(self,history,prompt=''):
        if prompt:
            history.append({'role': 'user', 'content': prompt})
        response =  self.client.chat.completions.create(
            model = self.model,
            messages = history,
            temperature = 0.3
        )

        return response.choices[0].message.content, history.append({'role':'assistant','content':response.choices[0].message.content})
    
    def GetReport(self):
        prompt = self.PromptGet.ConstructPrompt()
        history = common.OpenJson(self.config['history_path'])
        response ,_ = self.ModelResponse(history=history,prompt=prompt)
        return response
        




if __name__ == "__main__":
    vc = VigilantClassifier()
    print(vc.GetReport())
