from abc import ABC, abstractmethod
from openai import OpenAI
#import google.generativeai as genai
from dotenv import load_dotenv
import os
import requests
import Common as common
import json
import base64
from SharedLogger import CustomLogger
logger = CustomLogger().get_logger()
load_dotenv()
GEMINI_API = os.getenv('GEMINIAPIKEY')
OPENAI_API = os.getenv('OPENAIAPIKEY')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000')
SYSTEM_PROMPT = common.OpenTxt('templates/image_inter_prompt.vigileye')

class LLMModel(ABC):
    @abstractmethod
    def generate_response(self, messages):
        pass


class OpenAIModel(LLMModel):
    def __init__(self, api_key, model_name,with_images:bool=False):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.with_images = with_images

    def generate_response(self, messages,images =None):
        """
        if with images then messages are ignored
        Open-ai supports one image at a time
        """
        if not self.with_images:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3
            )
        else:
            image = images[0] 

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image,
                                }
                            }
                        ]
                    }
                ],
            )
        return response.choices[0].message.content
    

class GeminiModel(LLMModel):
    def __init__(self, api_key, model_name):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, messages):
        # Convert messages to Gemini format
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = self.model.generate_content(prompt)
        return response.text
    
    
class OllamaModel(LLMModel):
    def __init__(self, api_url, model_name, with_images=False,ignore_img_not_found=False):
        self.api_url = api_url
        self.model_name = model_name
        self.with_images = with_images
        self.ignore_img_not_found = ignore_img_not_found

    def generate_response(self, messages=None,images=None):
        """
        images is a list of images encoded in base64
        """
        if not messages:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }]
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        if not self.with_images:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": 0.3
                },
                "stream": False,
            }
        else: 
            ## images is a list of image urls
            images = [base64_image_encoder(image, self.ignore_img_not_found) for image in images]
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": 0.4
                },
                "images": images,
                "stream": False,
            }
            print(images)

        try:
            response = requests.post(url=f"{self.api_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            print(data)
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}") from e
        

class VLLMModel(LLMModel):
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name

    def generate_response(self, messages):
        headers = {
            'Content-Type': 'application/json',
        }
        # Assuming VLLM expects a similar payload to OpenAI's API
        payload = {
            'model': self.model_name,
            'messages': messages,
            'temperature': 0.3
        }
        try:
            response = requests.post(f"{self.api_url}/generate", headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"VLLM API request failed: {e}") from e


def base64_image_encoder(image_path: str, ignore_img_not_found: bool = False):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string
    except FileNotFoundError:
        #logger.log(msg=f"File not found at {image_path}. Trying to fetch from the internet.",level="info")
        response = requests.get(image_path)
        if response.status_code != 200:
            logger.log(f"File is not on the net either!")
            if ignore_img_not_found:
                return ''
            else:
                raise Exception(f"Error fetching image from {image_path}")
        encoded_string = base64.b64encode(response.content).decode('utf-8')
        return encoded_string

def get_llm_model(model_type, model_name,with_images:bool=False,ignore_img_not_found:bool=False):

    if model_type == "openai":
        return OpenAIModel(OPENAI_API, model_name,with_images=with_images)
    elif model_type == "gemini":
        return GeminiModel(GEMINI_API, model_name)
    elif model_type == "ollama":
        return OllamaModel(OLLAMA_API_URL, model_name,with_images=with_images,ignore_img_not_found=ignore_img_not_found)
    elif model_type == "vllm":
        return VLLMModel(VLLM_API_URL, model_name)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
if __name__ == "__main__":
    llm = get_llm_model("ollama", "llava",with_images=True)
    response = llm.generate_response(images=["https://i.redd.it/jj1xbwp29xod1.png"])
    print(response)