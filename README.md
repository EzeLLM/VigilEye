# VigilEye
 
An open-source social media user classifier. The classification is done according to the jinja template. there are couple default default templates and you can create your own to adapt the app for you use-case. The project still needs a lot of optimizations and has a long way to go, a thousand mile starts with a step. Example output with 27000 tokens generated for less than 2 cents with open-ai gpt4o-mini api for conspiracy subreddit is in export folder. 
# Usage
For now, the project is working on reddit, twitter part is still in development due to lack of an api.
You need reddit api and open-ai compatible api (vllm can be used to host local models with open-ai compatible api)
After setting your environment variables for the apis mentioned above install the requirements:
```pip install -r requirements.txt```
after installing the requirements you are ready to run the app:
```python3 dev/AllEyesOnReddit.py --reddit r/conspiracy --max_tokens 30000 --config config/vigileye.yaml```

# Configuration
The application uses YAML configuration files for defining how prompts are generated and how the classifier is set up. Here’s a brief overview of the configuration options:

**prompt_gen**:
- json_path: Path to the JSON file containing the data to be processed, if you are calling AllEyesOnReddit then leave this empty.
- template_path: Path to the Jinja2 template used for generating prompts. Default is 'templates/security.j2'.
- text_field: The field in the JSON data that will be used as the main text for classification. Default for reddit is 'title'.
- other_fields: List of additional fields from the JSON data that may be used in the prompt. Example fields include 'score', 'num_comments', and 'selftext'. for subfields use
```outer_key>>mid_key>>target_key```


**classifier**:

- model: The model used for classification. Default is 'gpt-4o-mini'.
- history_path: Path to the JSON file where the classifier's history is stored. Default is 'templates/history.json'. The history is used as an initializer and offers fexibility for few and single shot. current approach is zero shot
- query: List of fields or queries that the classifier will look for in the data. Example fields include 'location', 'real name', and 'age'. This is not tested yet and will work in ```templates/query.j2```

# Roadmap
- Classify social media users with llm agents and export a short summary of their ideologies thoughts and risks they may present to the state and the people
- Text based connection between classified users and a user database
- Export summaries to database about each perso
- Link users in the database using followers, following, and possible pictures with face identification model and maybe feature map, this step requires extensive research and re-thought

