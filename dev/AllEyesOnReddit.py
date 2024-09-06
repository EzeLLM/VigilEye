import praw
from dotenv import load_dotenv
import os 
import Common as common
import argparse
from tqdm import tqdm
from VigilantClassifier import VigilantClassifier
import time

parser = argparse.ArgumentParser(description="""
VigilEye app is made to track and evaluate social media. You are using the reddit branch of the app. This is meant to analyze users and subreddits. Usages can vary from marketing to security analysis. You adapt this app to your own usage purposes by making your own jinja2 prompt template and pass it to the app easily.
""")
parser.add_argument('--reddit', type=str,required=True, help='username or subreddit name. eg.: u/someUser , r/someSub')
parser.add_argument('--max_tokens',type=int,required=True,help='max tokens to be retrieved. The gpt4o-mini has context length of 128K tokens. 10% token safe zone will be applied')
parser.add_argument('--config',type=str,required=True,help='yaml config path')

args = parser.parse_args()
load_dotenv()

class EyesOnReddit():
    def __init__(self):
        print("here")

        self.token_limit = int(args.max_tokens * 0.9)
        self.reddit_address = args.reddit
        self.config_dir = args.config
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent="VigilEye/1.0 (by Ezel Bayraktar)"
        )

    def PostRetriever(self, source, post_type, limit=None, after=None):
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
        post_type = 'user' if source.split('/')[0] == 'u' else 'subreddit'
        source = source.split('/')[1]
        posts = []
        batch_size = 99
        wait_time = 62  # seconds
        total_tokens = 0
        no_new_posts_count = 0
        max_attempts = 3
        threshold = limit * 0.9  # 90% of the limit

        try:
            with tqdm(total=limit, desc="Retrieving tokens", unit="tokens") as pbar:
                while total_tokens < threshold and no_new_posts_count < max_attempts:
                    after = posts[-1]['name'] if posts else None
                    batch = self.PostRetriever(source, post_type, batch_size, after)
                    
                    if not batch:
                        no_new_posts_count += 1
                        print(f"No new posts retrieved. Attempt {no_new_posts_count} of {max_attempts}")
                        if no_new_posts_count >= max_attempts:
                            print("Max attempts reached. Stopping retrieval.")
                            break
                        time.sleep(wait_time)
                        continue

                    new_posts_added = False
                    for post in batch:
                        post_tokens = len(common.Tokenizer(str(post)))
                        if total_tokens + post_tokens > limit:
                            break
                        posts.append(post)
                        total_tokens += post_tokens
                        pbar.update(post_tokens)
                        new_posts_added = True

                        # Check if we've reached or exceeded the threshold after each post
                        if total_tokens >= threshold:
                            print(f"\nReached {int((total_tokens/limit)*100)}% of the token limit. Stopping retrieval.")
                            break

                    if new_posts_added:
                        no_new_posts_count = 0  # Reset the counter if new posts were added
                    else:
                        no_new_posts_count += 1

                    if total_tokens >= threshold:
                        break
                    
                    if no_new_posts_count < max_attempts:
                        time.sleep(wait_time)

            print(f"\nRetrieved {len(posts)} posts from {post_type} '{source}'.")
            print(f"Total tokens: {total_tokens}")
            return posts
        except Exception as e:
            print(f"An error occurred: {e}")
            return posts  # Return any posts retrieved before the error

    def main(self):
        posts = self.RedditHandler(self.token_limit, self.reddit_address)
        common.DumpToJson('jsonreddit.json',posts)
        if posts:
            print(f"Retrieved {len(posts)} posts")
            vc = VigilantClassifier(config_dir=self.config_dir, input=posts)
            vc_report = vc.GetReport()
            common.DumpToText(vc_report, os.path.join('export', 'vc_report.vigileye'))
        else:
            print("No posts were retrieved. Please check your Reddit credentials and network connection.")

if __name__ == '__main__':
    eor = EyesOnReddit()
    eor.main()