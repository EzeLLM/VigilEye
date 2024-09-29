from jinja2 import Environment, FileSystemLoader
import Common as common
from SharedLogger import CustomLogger
import CustomExceptions as ce
from ImageInterpreter import ImageInterpreter as imint
logger = CustomLogger().get_logger()


class PromptGen():
    def __init__(self,config_path,main_field,other_fields,template,input__,handle_pics=False):
        self.main_field = main_field
        self.other_fields = other_fields
        self.template = template
        self.handle_pics = handle_pics
        self.input = input__
        self.config_path = config_path
    # def ConstructPrompt(self,input,peruser: bool):
    #     if peruser:
    #         prompt = common.JinjaRender_per_user(self.template,input)
    #     else:
    #         pass
    #         ## TODO implement here
    #         ## unify template rendering
    #     return prompt
    def ConstructPrompt(self,peruser: bool=False):
        if peruser:
            prompt = common.JinjaRender(self.template,input=self.input,mode='per_user')
            return prompt
        posts_to_template = []
        posts = self.input
            
        text_field = self.main_field
        other_fields = self.other_fields
    
        if self.handle_pics:
            imint_ = imint(self.config_path)
            posts = imint_.Handler(posts=posts)
        
        for post in posts:
            try:
                if not isinstance(post,dict):
                    logger.critical(f"Post is not of type dict! post>> {post}")
                    continue
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
                            if value != {} and value != '':
                                post_to_template+=f"{other_}: {value}\n" 
                        except:
                            logger.critical(f"{other_} not found in a post. continue.. (This can happen)")
                            pass
                    else:
                        post_to_template+=f"{other_}: {post[other_]}\n"
                if 'image_description' in post:
                    post_to_template+=post['image_description']
            except KeyError:
                logger.critical(f"A post did not have a key! Maybe image post?")
                pass

            posts_to_template.append(post_to_template)
        prompt = common.JinjaRender(input=posts_to_template,template_path=self.template)
        return prompt
    


if __name__ == "__main__":
    vc = PromptGen("config/vigileye.yaml")
    print("Initialized VigilantClassifier")
    prompt = vc.ConstructPrompt()
    print("Prompt constructed")
    print(prompt)

                    


