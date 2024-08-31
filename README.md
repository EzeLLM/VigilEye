# VigilEye
 
An open-source project meant to track people on social media and classify them based on their ideologies and thoughts.

# Roadmap
- Classify social media users with llm agents and export a short summary of their ideologies thoughts and risks they may present to the state and the people
- Text based connection between classified users and a Mernis like database
- Export summaries to database about each person's ideology, thoughts, and potential risks. 
- Link users in the database using followers, following, and possible pictures with face identification model and maybe feature map, this step requires extensive research and re-thought

# Tools to be used
- **LLM**: Thinking of using chat gpt api (4o mini) during development and going to switch to vllm open-ai-api open-source model, phi-3-5 seems reasoble but havent tried it yet.  
- **Platform**: X is a good platform, given the fact that it contains a larger portion of extremist people and the target users, turkish people, use it for politics, mostly.
- **API**: X api is not a making-sense option. Currently thinking of using rettiwt-api library from npm repo.

# Notes
- rettiwt-api did not end up well. rate limits are still present. to keep the project alive im switching to reddit for now. as the core agent part wont be affected a lot
- will solve the api later, to not stall i started for now