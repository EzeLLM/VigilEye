from abc import ABC, abstractmethod
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import os
import requests
import json
load_dotenv()
GEMINI_API = os.getenv('GEMINIAPIKEY')
OPENAI_API = os.getenv('OPENAIAPIKEY')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000')


class LLMModel(ABC):
    @abstractmethod
    def generate_response(self, messages):
        pass


class OpenAIModel(LLMModel):
    def __init__(self, api_key, model_name):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3
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
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name

    def generate_response(self, messages):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": 0.3
            },
            "stream": False,
        }
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


def get_llm_model(model_type, model_name):
    if model_type == "openai":
        return OpenAIModel(OPENAI_API, model_name)
    elif model_type == "gemini":
        return GeminiModel(GEMINI_API, model_name)
    elif model_type == "ollama":
        return OllamaModel(OLLAMA_API_URL, model_name)
    elif model_type == "vllm":
        return VLLMModel(VLLM_API_URL, model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    