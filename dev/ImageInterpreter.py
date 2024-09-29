from PIL import Image
import Common as common
import requests
from SharedLogger import CustomLogger
import LLMModels as llm
logger = CustomLogger().get_logger()

class ImageInterpreter():
    def __init__(self,config) -> None:
        self.config = common.OpenYaml(config,'image_interpreter')
        self.img_formats = self.config['image_formats']
        self.image_field = self.config['image_field']
        self.ignore_img_not_found:bool = self.config['ignore_img_not_found']
        self.host = self.config['host']
        self.model = self.config['model']
        self.llm = llm.get_llm_model(model_type=self.host,model_name=self.model,with_images=True,ignore_img_not_found=self.ignore_img_not_found)

    def GetInterpretation(self, image_path: str):
        return self.llm.generate_response(images=[image_path])
        
    def Handler(self, posts) -> dict:
        for post in posts:
            if (self.image_field in post) and (common.EndsWith(post[self.image_field],self.img_formats)):
                post['image_description'] = f'The posts contains an image that can be described as {self.GetInterpretation(post[self.image_field])}'
        common.DumpToJson('posts.json',data=posts)
        return posts
if __name__ == '__main__':
    interpreter = ImageInterpreter('config/vigileye.yaml')
    interp = interpreter.GetInterpretation('https://canto-wp-media.s3.amazonaws.com/app/uploads/2019/08/19194138/image-url-3.jpg')
    print(interp)