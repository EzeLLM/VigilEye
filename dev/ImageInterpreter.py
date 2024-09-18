import requests
from PIL import Image
import Common as common
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import warnings
warnings.filterwarnings("ignore", message="huggingface/tokenizers: The current process just got forked")

class ImageInterpreter():
    def __init__(self,config) -> None:
        self.config = common.OpenYaml(config,'image_interpreter')
        self.img_formats = self.config['image_formats']
        self.image_field = self.config['image_field']
        self.use_local_model = self.config['use_local_model']
        self.max_new_tokens = self.config['max_new_tokens']
        
        if self.use_local_model:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.device = common.Device()
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(self.device)
        else:
            from openai import OpenAI
            self.model = self.config['model']
            self.api_key = os.getenv('OPENAIAPIKEY')
            self.client = OpenAI(api_key=self.api_key)

    def GetInterpretation(self, image_path: str):
        if self.use_local_model:
            return self._local_interpretation(image_path)
        else:
            return self._openai_interpretation(image_path)

    def _local_interpretation(self, image_path: str):
        import torch
        raw_image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        text = "a photography of"
        inputs = self.processor(raw_image, text, return_tensors="pt").to("mps", torch.float16)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def _openai_interpretation(self, image_path: str):
        try:
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {
                        "role": "system",
                        "content": "Give a brief description of what is in the picture. Try to recognize political figures and known people especially. Give information about the geography and the place if possible. If the picture is a meme, what is it mocking? Give answers to such questions in your brief description."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_path,
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_new_tokens
            )
            return response.choices[0].message.content
        except Exception as error:
            print(f"Error processing image: {error}")
            raise

    def Handler(self, posts) -> dict:
        for post in posts:
            if (self.image_field in post) and (common.EndsWith(post[self.image_field],self.img_formats)):
                post['image_description'] = f'The posts contains an image that can be described as {self.GetInterpretation(post[self.image_field])}'
        common.DumpToJson('posts.json',data=posts)
        return posts
