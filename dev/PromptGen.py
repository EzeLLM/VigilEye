# from langchain.llms import OpenAI
# from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import json
import argparse
import yaml
import sys
from jinja2 import Environment, FileSystemLoader
import Common as common
import tiktoken
from SharedLogger import CustomLogger
logger = CustomLogger().get_logger()

# NOTE DEPRECATED FOR NOW
# class Post():
#     def __init__(self):
#         self.post = ''
#     def add_text(self,prefix,text):
#         self.post += f"{prefix}: {text}"
#     def get_post(self):
#         return self.post

class PromptGen():
    def __init__(self,config_path):
        self.config = common.OpenYaml(config_path,'prompt_gen')
        self.tokenizer = tiktoken.get_encoding('cl100k_base')

    def Tokenizer(self,text):
        return self.tokenizer.encode(text=text)

    def JinjaRender(self,posts,template_path):
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template(template_path)
        
        rendered_prompt = template.render(posts=posts)
        logger.info(f'Prompt length in tokens: {len(self.Tokenizer(rendered_prompt))}')
        return rendered_prompt

    def ConstructPrompt(self):
        posts_to_template = []

        json_path = self.config['json_path']
        text_field = self.config['text_field']
        other_fields = self.config['other_fields']

        posts = common.OpenJson(json_path)
        
        for post in posts:

            try:
                # print(post)
                main_text = post[text_field]
                
                post_to_template = f'Main post text: {main_text}\n'
                for other_ in other_fields:
                    if len(other_.split('>>')) > 1:
                        other_splitted = other_.split('>>')
                        value = post
                        try:
                            for key in other_splitted:
                                value = value.get(key, {})
                                if value == None:
                                    break 
                            if value != {}:
                                post_to_template+=f"{other_}: {value}\n" 
                        except:
                            logger.critical(f"{other_} not found in a post. continue.. (This can happen)")
                            pass
                        
                    else:
                        post_to_template+=f"{other_}: {post[other_]}\n"
            except KeyError:
                logger.critical(f"A post did not have a key! Maybe image post?")
                pass

            posts_to_template.append(post_to_template)
        prompt = self.JinjaRender(posts=posts_to_template,template_path=self.config['template_path'])
        return prompt
    


if __name__ == "__main__":
    vc = PromptGen("config/vigileye.yaml")
    print("Initialized VigilantClassifier")
    prompt = vc.ConstructPrompt()
    print("Prompt constructed")
    print(prompt)

                    


