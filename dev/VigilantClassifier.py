import argparse
from jinja2 import Environment, FileSystemLoader
import Common as common
from SharedLogger import CustomLogger
from typing import final
from LLMModels import get_llm_model
from PromptGen import PromptGen
import os
from typing import Optional
from dotenv import load_dotenv
import json

load_dotenv()

logger = CustomLogger().get_logger()

class VigilantClassifier():
    def __init__(self, config_dir, template, main_field, other_fields, handle_pics=False, input__=False, mode='default'):
        self.template = template
        self.mode = mode
        self.config = common.OpenYaml(config_dir, 'classifier')
        self.llm_config = common.OpenYaml(config_dir, 'llm')
        self.model_type = self.llm_config['LLM_type']
        self.model_name = self.llm_config['LLM_model']
        self.llm = get_llm_model(model_type=self.model_type, model_name=self.model_name)
        self.input = input__
        
        # Initialize Prompt Generator
        self.PromptGet = PromptGen(
            config_path=config_dir,
            input__=self.input,
            main_field=main_field,
            other_fields=other_fields,
            handle_pics=handle_pics,
            template=template
        )

    def ModelResponse(self, history, prompt=''):
        if prompt:
            history.append({'role': 'user', 'content': prompt})
        # Generate response using the selected LLM model
        response = self.llm.generate_response(history)
        
        # Assuming the response appends the assistant's reply to history
        history.append({'role': 'assistant', 'content': response})
        
        return response, history

    def GetReport(self):
        if self.mode == 'peruser':
            prompt = self.PromptGet.ConstructPrompt(peruser=True)
        else:
            prompt = self.PromptGet.ConstructPrompt()
        
        # Optional: Save the constructed prompt for debugging
        common.DumpToText(prompt, 'constructed_prompt.txt')

        # Load history from the specified path
        history = common.OpenJson(self.config['history_path'])
        
        # Get response from the model
        response, updated_history = self.ModelResponse(history=history, prompt=prompt)
        
        # Optionally, update the history file
        common.DumpToJson(self.config['history_path'], updated_history)

        return response
    

if __name__ == "__main__":
    # Example usage (You may need to adjust parameters based on actual usage)
    vc = VigilantClassifier(
        config_dir='config/vigileye.yaml',
        template='templates/subreddit.j2',
        main_field='title',
        other_fields=['selftext', 'score', 'num_comments'],
        handle_pics=False,
        input__=False,
        mode='default'
    )
    print(vc.GetReport())
