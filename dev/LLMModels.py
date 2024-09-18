from abc import ABC, abstractmethod
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()
GEMINI_API = os.getenv('GEMINIAPIKEY')
OPENAI_API = os.getenv('OPENAIAPIKEY')
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

def get_llm_model(model_type, model_name):
    if model_type == "openai":
        return OpenAIModel(OPENAI_API, model_name)
    elif model_type == "gemini":
        return GeminiModel(GEMINI_API, model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")