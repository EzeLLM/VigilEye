from jinja2 import Environment, FileSystemLoader
import Common as common
from SharedLogger import CustomLogger
import CustomExceptons as ce
logger = CustomLogger().get_logger()


class PromptGen():
    def __init__(self,config_path,input=False):
        self.config = common.OpenYaml(config_path,'prompt_gen')
        self.input = input
    def ConstructPrompt(self):
        posts_to_template = []
        try:
            json_path = self.config['json_path']
            posts = common.OpenJson(json_path)
        except:
            if not self.input:
                raise ce.CustomError("No input was passed to PromptGen, pass either json path in yaml or json dict as an arg in init")
            posts = self.input
            
        text_field = self.config['text_field']
        other_fields = self.config['other_fields']
    
        
        for post in posts:
            try:
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
        prompt = common.JinjaRender(posts=posts_to_template,template_path=self.config['template_path'])
        return prompt
    


if __name__ == "__main__":
    vc = PromptGen("config/vigileye.yaml")
    print("Initialized VigilantClassifier")
    prompt = vc.ConstructPrompt()
    print("Prompt constructed")
    print(prompt)

                    


