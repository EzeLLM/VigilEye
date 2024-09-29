# VigilEye

An open-source social media user classifier. The classification is done according to the Jinja template. There are a couple of default templates, and you can create your own to adapt the app for your use-case. The project still needs a lot of optimizations and has a long way to go, but a thousand miles start with a step. Example output with 27000 tokens generated for less than 2 cents with the OpenAI GPT-4o-mini API for the conspiracy subreddit is in the export folder.

## Features
*   **Multi-Platform Support**: Currently supports Reddit, with Twitter support in development.
*   **Customizable Templates**: Use Jinja2 templates to customize the classification process.
*   **Support for Images**: Can handle images in posts and generate descriptions using AI models.
*   **Risk Assessment**: Evaluates users and subreddits based on their posts and assigns a risk level.

## SOON 🔜👁️‍🗨️
*   **Groq Support**: Integration with Groq AI models for faster free on-cloud processing.
*   **VLLM Image Interpretation Support**: Support for VLLM AI models for image interpretation.
*   **GUI**: A graphical user interface for easier usage.

## Usage
For now, the project is working on Reddit; the Twitter part is still in development due to the lack of an API.
You need Reddit API and if you are not preffering Ollama , you need an OpenAI-compatible API (vllm can be used to host local models with OpenAI-compatible API).
After setting your environment variables for the APIs mentioned above, install the requirements:
```bash
pip install -r requirements.txt
```
After installing the requirements, you are ready to run the app:
```bash
python3 dev/AllEyesOnReddit.py --reddit r/conspiracy --max_tokens 30000 --config config/vigileye.yaml
```

## Configuration
The application uses YAML configuration files for defining how prompts are generated and how the classifier is set up. Here’s a brief overview of the configuration options:

### llm:
*   **LLM_type**: The type of language model to use (e.g., `openai` or `ollama`).
*   **LLM_model**: The specific model to use (e.g., `gpt-4o-mini` or `llava (with ollama)`).

### image_interpreter:
*   **image_formats**: A list containing the image formats of the pictures. This is used, for Reddit at least, since Reddit gives links that may contain the picture or a generic link from the post. This is used to know whether the link is a picture. Reddit uses png.
*   **image_field**: The field in the JSON that the picture link is contained in.
*   **model**: Specify the model name. Currently needed for API method only.
*   **ignore_img_not_found**: Whether to throw an error in case of invalid image path.
*   **host**: Which host to use to interpret images. You can choose either openai or ollama as of now. for ollama YOU MUST choose a multimodal llm.

### classifier:
*   **history_path**: Path to the JSON file where the classifier's history is stored. Default is `templates/history.json`. The history is used as an initializer and offers flexibility for few and single-shot. The current approach is zero-shot.

### all_eyes_on_reddit:
*   **criteria**: Placeholder for future implementation criteria.
*   **main_field**: The primary field to analyze (e.g., `title`).
*   **other_fields**: Additional fields to analyze (e.g., `selftext`, `score`, `num_comments`).
*   **author_field**: The field that contains the author's name (e.g., `author`).
*   **template**: Path to the Jinja2 template for subreddit analysis.
*   **user_template**: Path to the Jinja2 template for user analysis.
*   **sub_template**: Path to the Jinja2 template for subreddit user analysis.
*   **reddit_**: Default subreddit to analyze if none is provided.
*   **use_images**: Boolean indicating whether to use images in analysis.

## Example Configuration File
Here is an example configuration file (`config/vigileye.yaml`):

```yaml

llm: # used for report extraction
  LLM_type: openai # Options: openai, gemini, ollama, vllm
  LLM_model: gpt-4o-mini # Replace with your desired model name

#⛔️ Caution: Local interpretation should only be used with multimodal models. Currently supporting OLLAMA. VLLM is so soon! ⛔️'
image_interpreter:
  ignore_img_not_found: False
  image_formats:
    - 'png' # png is default for reddit
  image_field: 'url'
  host: 'ollama'
  model: 'llava'

classifier:
  history_path: templates/history.json


all_eyes_on_reddit:
  criteria: 'to be implemented later'
  main_field: 'title'
  other_fields:
    - 'selftext'
    - 'score'
    - 'num_comments'
  author_field: 'author'
  template: templates/subreddit.j2
  user_template: 'templates/security2.j2'
  sub_template: 'templates/sub_per_user.j2'
  reddit_: 'r/Turkey'
  use_images: False

  
```

## Running the Project
To run the project, use the following command:
```bash
python3 dev/AllEyesOnReddit.py --reddit r/tesla --max_tokens 2000 --config config/vigileye.yaml
```

This command will analyze the specified subreddit (`r/tesla` in this case) and generate a report based on the configuration provided in `config/vigileye.yaml`.

## Customizing the Configuration
You can customize the configuration file to suit your needs. For example, you can change the subreddit to analyze, the maximum number of tokens, or the templates used for generating prompts. Make sure to update the configuration file accordingly and pass it to the script using the `--config` argument.

For any issues or further customization, refer to the detailed comments in the configuration file and the provided templates.

## Project Structure

```
VigilEye
├── requirements.txt
├── config
│   └── vigileye.yaml
├── extractor
├── README.md
├── templates
│   ├── security2.j2
│   ├── general.j2
│   ├── subreddit.j2
│   ├── history.json
│   ├── security.j2
│   ├── sub_per_user.j2
│   └── image_inter_prompt.vigileye
├── dev
│   ├── SharedLogger.py
│   ├── AllEyesOnReddit.py
│   ├── LLMModels.py
│   ├── CustomExceptions.py
│   ├── Common.py
│   ├── AllEyesOnTwitter.py
│   ├── ImageInterpreter.py
│   ├── VigilantClassifier.py
│   └── PromptGen.py
└── export
    ├── vc_report.vigileye
    └── vc_report_per_user.vigileye
```

## Requirements

```
apify_client==1.8.1
Jinja2==3.1.4
matplotlib==3.9.2
openai==1.46.0
Pillow==10.4.0
praw==7.7.1
python-dotenv==1.0.1
PyYAML==6.0.2
PyYAML==6.0.2
Requests==2.32.3
tiktoken==0.7.0
torch==2.4.1
tqdm==4.66.5
transformers==4.44.2
```