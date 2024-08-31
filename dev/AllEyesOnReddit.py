import praw
import datetime
import json
import os 
import Common as common
import argparse
from tqdm import tqdm
from VigilantClassifier import VigilantClassifier
parser = argparse.ArgumentParser(description="""
VigilEye app is made to track and evaluate social media. You are using the reddit branch of the app. This is meant to analyze users and subreddits. Usages can vary from marketing to security analysis. You adapt this app to your own usage purposes by making your own jinja2 prompt template and pass it to the app easily.
""")
parser.add_argument('--reddit', type=str,required=True, help='username or subreddit name. eg.: u/someUser , r/someSub')
parser.add_argument('--max_tokens',type=int,required=True,help='max tokens to be retrieved. The gpt4o-mini has context length of 128K tokens. 10\% token safe zone will be applied')
parser.add_argument('--config',type=int,required=True,help='yaml config path')

args = parser.parse_args()

class EyesOnReddit():
    def __init__(self):
        self.token_limit = (args.max_tokens * 0.9)
        self.reddit_address = args.reddit
        self.config_dir = args.config
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv("REDDIT_CLIEND_SECRET"),
            user_agent="VigilEye/1.0 (by Ezel Bayraktar)"
        )

    def PostRetriever(self,source, post_type, limit=None, after=None):
        posts = []
        
        if post_type == 'subreddit':
            subreddit = self.reddit.subreddit(source)
            submissions = subreddit.hot(limit=limit, params={'after': after})
        elif post_type == 'user':
            user = self.reddit.redditor(source)
            submissions = user.submissions.hot(limit=limit, params={'after': after})
        else:
            raise ValueError("Invalid post_type. Use 'subreddit' or 'user'.")
        
        for submission in submissions:
            post = {
                'name': submission.name,
                'title': submission.title,
                'author': submission.author.name if submission.author else '[deleted]',
                'created_utc': submission.created_utc,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'url': submission.url,
                'selftext': submission.selftext
            }
            posts.append(post)

        
        return posts



    def RedditHandler(self, limit, source):
        import time
        from tqdm import tqdm

        post_type = 'user' if source.split('/')[0] == 'u' else 'subreddit'
        limit = int(limit)  # Now limit represents token count
        posts = []
        batch_size = 99
        wait_time = 62  # seconds
        total_tokens = 0

        try:
            with tqdm(total=limit, desc="Retrieving tokens") as pbar:
                while total_tokens < limit:
                    after = posts[-1]['name'] if posts else None
                    batch = self.PostRetriever(source, post_type, batch_size, after)
                    
                    for post in batch:
                        post_tokens = len(common.Tokenizer(str(post)))
                        if total_tokens + post_tokens > limit:
                            break
                        posts.append(post)
                        total_tokens += post_tokens
                        pbar.update(post_tokens)

                    print(f"Retrieved {len(batch)} posts. Total tokens: {total_tokens}")
                    
                    if len(batch) < batch_size or total_tokens >= limit:
                        break
                    
                    print(f"Waiting {wait_time} seconds before next batch...")
                    time.sleep(wait_time)

            #filename = f"{source}_{post_type}_posts.json"
            #common.DumpToJson(posts, filename)
            print(f"\nRetrieved {len(posts)} posts from {post_type} '{source}'.")
            print(f"Total tokens: {total_tokens}")
            #print(f"Data saved to {filename}")
            return posts
        except Exception as e:
            print(f"An error occurred: {e}")

    def main(self):
        vc = VigilantClassifier(config_dir=self.config_dir,input=self.RedditHandler(self.token_limit,self.reddit_address))
        vc_report = vc.GetReport()