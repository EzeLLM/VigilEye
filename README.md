# VigilEye

An open-source social media user classifier. The classification is done according to the Jinja template. There are a couple of default templates, and you can create your own to adapt the app for your use-case. The project still needs a lot of optimizations and has a long way to go, but a thousand miles start with a step. Example output with 27000 tokens generated for less than 2 cents with the OpenAI GPT-4o-mini API for the conspiracy subreddit is in the export folder.

## News
VigilEye can now interpret pictures from the posts. It can read documents, find whether any significant people are in the picture, and identify what memes are mocking! Beyond that it can analyze users individually from a subreddit

## Usage
For now, the project is working on Reddit; the Twitter part is still in development due to the lack of an API.
You need Reddit API and OpenAI-compatible API (vllm can be used to host local models with OpenAI-compatible API).
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
- **LLM_type**: The type of language model to use (e.g., `openai`).
- **LLM_model**: The specific model to use (e.g., `gpt-4o-mini`).

### image_interpreter:
- **image_formats**: A list containing the image formats of the pictures. This is used, for Reddit at least, since Reddit gives links that may contain the picture or a generic link from the post. This is used to know whether the link is a picture. Reddit uses png.
- **image_field**: The field in the JSON that the picture link is contained in.
- **max_new_tokens**: Max new tokens for image interpretation response.
- **use_local_model**: Boolean on whether to use a local model or OpenAI-compatible API.
- **model**: Specify the model name. Currently needed for API method only.

### classifier:
- **model**: The model used for classification. Default is `gpt-4o-mini`.
- **history_path**: Path to the JSON file where the classifier's history is stored. Default is `templates/history.json`. The history is used as an initializer and offers flexibility for few and single-shot. The current approach is zero-shot.
- **query**: List of fields or queries that the classifier will look for in the data. Example fields include `location`, `real name`, and `age`. This is not tested yet and will work in `templates/query.j2`.

### all_eyes_on_reddit:
- **criteria**: Placeholder for future implementation criteria.
- **main_field**: The primary field to analyze (e.g., `title`).
- **other_fields**: Additional fields to analyze (e.g., `selftext`, `score`, `num_comments`).
- **author_field**: The field that contains the author's name (e.g., `author`).
- **template**: Path to the Jinja2 template for subreddit analysis.
- **user_template**: Path to the Jinja2 template for user analysis.
- **sub_template**: Path to the Jinja2 template for subreddit user analysis.
- **reddit_**: Default subreddit to analyze if none is provided.
- **use_images**: Boolean indicating whether to use images in analysis.

## Example Configuration File
Here is an example configuration file (`config/vigileye.yaml`):

```yaml
llm:
  LLM_type: openai
  LLM_model: gpt-4o-mini

image_interpreter:
  image_formats:
    - 'png' # png is default for reddit
  image_field: 'url'
  max_new_tokens: 128
  use_local_model: False
  model: gpt-4o-mini

classifier:
  model: gpt-4o-mini
  history_path: templates/history.json
  query:
    - 'location'
    - 'real name'
    - 'age'

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
  use_images: True 
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
