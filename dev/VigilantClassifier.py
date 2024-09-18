import argparse
from jinja2 import Environment, FileSystemLoader
import Common as common
from SharedLogger import CustomLogger
from typing import final
from openai import OpenAI
from LLMModels import get_llm_model
from PromptGen import PromptGen
import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

logger = CustomLogger().get_logger()


class VigilantClassifier():
    def __init__(self,config_dir, template, main_field, other_fields,handle_pics=False, input__ = False, mode='default'):
        self.template = template
        self.mode = mode
        self.config = common.OpenYaml(config_dir,'classifier')
        self.llm_config = common.OpenYaml(config_dir,'llm')
        self.model_type = self.llm_config['LLM_type']
        self.model_name = self.llm_config['LLM_model']
        self.llm = get_llm_model(model_type=self.model_type,model_name=self.model_name)
        self.api_key = os.getenv('OPENAIAPIKEY')
        self.model = self.config['model']
        self.client = OpenAI(api_key=self.api_key)
        self.input = input__
        print(type(self.input))
        self.PromptGet = PromptGen(config_path=config_dir,input__=self.input,main_field=main_field,other_fields=other_fields,handle_pics=handle_pics,template=template)


    def ModelResponse(self,history,prompt=''):
        if prompt:
            history.append({'role': 'user', 'content': prompt})
        # common.DumpToJson('hist.json',history)
        response =  self.client.chat.completions.create(
            model = self.model,
            messages = history,
            temperature = 0.3
        )
        # response_ =  response.choices[0].message.content
        response_ = self.llm.generate_response(history)
        # common.DumpToText(response_,'response.txt')
        return response_, history.append({'role':'assistant','content':response.choices[0].message.content})
    
    def GetReport(self):
        if self.mode == 'peruser':
            prompt = self.PromptGet.ConstructPrompt(peruser=True)
            # print("--------------------------------------/////////////////////////")
        # default case is default mode
        else:
            prompt = self.PromptGet.ConstructPrompt()
        common.DumpToText(prompt,'response2.txt')

        history = common.OpenJson(self.config['history_path'])
        response ,_ = self.ModelResponse(history=history,prompt=prompt)
        return response
    

if __name__ == "__main__":
    vc = VigilantClassifier()
    print(vc.GetReport())
