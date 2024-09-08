import os
from dotenv import load_dotenv
from PromptGen import PromptGen
from LLMModels import get_llm_model
import Common as common
from SharedLogger import CustomLogger

load_dotenv()
logger = CustomLogger().get_logger()

class VigilantClassifier:
    def __init__(self, config_dir, template, main_field, other_fields, handle_pics=False, input=False):
        self.template = template
        self.config = common.OpenYaml(config_dir, 'classifier')
        self.api_key = os.getenv('LLM_API_KEY')
        self.model_type = self.config['model_type']
        self.model_name = self.config['model_name']
        self.llm_model = get_llm_model(self.model_type, self.api_key, self.model_name)
        self.input = input
        self.PromptGet = PromptGen(config_path=config_dir, input=self.input, main_field=main_field, other_fields=other_fields, handle_pics=handle_pics, template=template)

    def ModelResponse(self, history, prompt=''):
        if prompt:
            history.append({'role': 'user', 'content': prompt})
        response = self.llm_model.generate_response(history)
        history.append({'role': 'assistant', 'content': response})
        return response, history

    def GetReport(self):
        prompt = self.PromptGet.ConstructPrompt()
        history = common.OpenJson(self.config['history_path'])
        response, _ = self.ModelResponse(history=history, prompt=prompt)
        return response

if __name__ == "__main__":
    vc = VigilantClassifier()
    print(vc.GetReport())
