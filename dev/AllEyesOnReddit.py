import praw
from dotenv import load_dotenv
import os 
import Common as common
import argparse
from tqdm import tqdm
from VigilantClassifier import VigilantClassifier
import time
import CustomExceptions as ce
from SharedLogger import CustomLogger
import sys

# Initialize the logger
logger = CustomLogger().get_logger()

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="""
VigilEye app is made to track and evaluate social media. You are using the reddit branch of the app. This is meant to analyze users and subreddits. Usages can vary from marketing to security analysis. You adapt this app to your own usage purposes by making your own jinja2 prompt template and pass it to the app easily.
""")
parser.add_argument('--reddit', type=str, required=False, help='username or subreddit name. eg.: u/someUser , r/someSub')
parser.add_argument('--max_tokens', type=int, required=True, help='max tokens to be retrieved. The gpt4o-mini has context length of 128K tokens. 10% token safe zone will be applied')
parser.add_argument('--config', type=str, required=True, help='yaml config path')
args = parser.parse_args()

class AllEyesOnReddit:
    def __init__(self):
        # Load configuration
        self.config = common.OpenYaml(args.config, 'all_eyes_on_reddit')
        self.token_limit = int(args.max_tokens)
        self.reddit_address = args.reddit if args.reddit else self.config.get('reddit_', 'r/Turkey')
        logger.info(f"Analyzing Reddit source: {self.reddit_address}")
        
        # Configuration flags and fields
        self.use_images = self.config.get('use_images', False)
        self.other_fields = self.config.get('other_fields', [])
        self.main_field = self.config.get('main_field', 'title')
        self.user_field = self.config.get('author_field', 'author')
        self.config_dir = args.config

        # Initialize PRAW (Reddit API) client
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', "VigilEye/1.0 (by Ezel Bayraktar, Mert Ergün)")
        )

        # Determine if per-user classification is needed
        self.DO_PERUSER_CLASSIFY = True if self.user_field else False

        try:
            self.template = self.config['template']
            self.user_template = self.config['user_template']
            self.sub_template = self.config['sub_template']
        except KeyError as e:
            if not self.DO_PERUSER_CLASSIFY:
                raise ce.CustomError('You should provide a template unless you provided user and sub template!')
            else:
                raise ce.CustomError(f'Missing required template: {e}')
    
    def PostRetriever(self, source, post_type, limit=None, after=None):
        """
        Retrieves posts from a subreddit or user.

        Args:
            source (str): The source name (e.g., 'r/someSub' or 'u/someUser').
            post_type (str): 'subreddit' or 'user'.
            limit (int, optional): Number of posts to retrieve.
            after (str, optional): The post ID to start retrieving after.

        Returns:
            list: A list of post dictionaries.
        """
        source = source.split('/')[1]
        logger.debug(f"Retrieving {post_type} posts for: {source}")
        posts = []
        
        if post_type == 'subreddit':
            subreddit = self.reddit.subreddit(source)
            submissions = subreddit.hot(limit=limit, params={'after': after})
            for submission in submissions:
                post = {
                    'name': submission.name,
                    'title': submission.title,
                    'author': submission.author.name if submission.author else '[UNKNOWN]',
                    'created_utc': submission.created_utc,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'url': submission.url,
                    'selftext': submission.selftext,
                    'type': 'submission'
                }
                posts.append(post)

        elif post_type == 'user':
            user = self.reddit.redditor(source)
            
            # Retrieve submissions
            submissions = user.submissions.new(limit=limit, params={'after': after})
            for submission in submissions:
                post = {
                    'name': submission.name,
                    'title': submission.title,
                    'author': submission.author.name if submission.author else '[UNKNOWN]',
                    'created_utc': submission.created_utc,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'url': submission.url,
                    'selftext': submission.selftext,
                    'type': 'submission'
                }
                posts.append(post)
            
            # Uncomment below to retrieve comments as well
            # comments = user.comments.new(limit=limit, params={'after': after})
            # for comment in comments:
            #     post = {
            #         'name': comment.name,
            #         'author': comment.author.name if comment.author else '[UNKNOWN]',
            #         'created_utc': comment.created_utc,
            #         'score': comment.score,
            #         'body': comment.body,
            #         'type': 'comment'
            #     }
            #     posts.append(post)

        else:
            logger.error("Invalid post_type. Use 'subreddit' or 'user'.")
            raise ValueError("Invalid post_type. Use 'subreddit' or 'user'.")

        return posts

    def get_post_type(self, source):
        """
        Determines the type of the Reddit source.

        Args:
            source (str): The source string (e.g., 'u/someUser' or 'r/someSub').

        Returns:
            str: 'user' or 'subreddit'.

        Raises:
            ValueError: If the source format is invalid.
        """
        source = source.replace('\\', '/')

        if source.startswith('u/'):
            return 'user'
        elif source.startswith('r/'):
            return 'subreddit'
        else:
            logger.error("Invalid source format. Must start with 'u/' for users or 'r/' for subreddits.")
            raise ValueError("Invalid source format. Must start with 'u/' for users or 'r/' for subreddits.")

    def RedditHandler(self, limit, source: str):
        """
        Handles the retrieval of Reddit posts with token limits and batching.

        Args:
            limit (int): Maximum number of tokens to retrieve.
            source (str): The Reddit source (e.g., 'u/someUser' or 'r/someSub').

        Returns:
            list: A list of post dictionaries.
        """
        post_type = self.get_post_type(source=source)
        source = source.replace('\\', '/')

        posts = []
        batch_size = 100
        wait_time = 61  # seconds
        total_tokens = 0
        no_new_posts_count = 0
        max_attempts = 2
        threshold = limit * 0.95  # 95% of the limit

        try:
            with tqdm(total=limit, desc="Retrieving tokens", unit="tokens") as pbar:
                while total_tokens < threshold and no_new_posts_count < max_attempts:
                    after = posts[-1]['name'] if posts else None
                    batch = self.PostRetriever(source, post_type, batch_size, after)
                    
                    if not batch:
                        logger.warning('No posts retrieved or no new posts available.')
                        return posts

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
                            logger.info(f"Reached {int((total_tokens / limit) * 100)}% of the token limit. Stopping retrieval.")
                            break

                    if new_posts_added:
                        no_new_posts_count = 0  # Reset the counter if new posts were added
                    else:
                        no_new_posts_count += 1

                    if total_tokens >= threshold:
                        logger.info(f"Reached {int((total_tokens / limit) * 100)}% of the token limit. Stopping retrieval.")
                        break
                    
                    if no_new_posts_count < max_attempts:
                        logger.debug(f"No new posts added in this batch. Waiting for {wait_time} seconds before retrying...")
                        time.sleep(wait_time)

            logger.info(f"Retrieved {len(posts)} posts from {post_type} '{source}'.")
            logger.info(f"Total tokens: {total_tokens}")
            return posts
        except Exception as e:
            logger.error(f"An error occurred while retrieving posts: {e}")
            return posts  # Return any posts retrieved before the error

    def _per_user_classifier(self, reddit_address):
        """
        Handles per-user classification within a subreddit.

        Args:
            reddit_address (str): The subreddit address (e.g., 'r/someSub').

        Returns:
            str: The generated report.
        """
        posts = self.RedditHandler(self.token_limit, reddit_address)
        authors = list(set([post[self.user_field] for post in posts if post[self.user_field] != '[UNKNOWN]']))
        logger.debug(f"Found {len(authors)} unique authors.")

        user_reports = {}
        for author in authors:
            report = self._default(f'u/{author}', _template=self.user_template)
            user_reports[author] = report
            logger.debug(f'Author: {author} >> Report: {report}')

        vc = VigilantClassifier(
            config_dir=self.config_dir,
            template=self.sub_template, 
            main_field=self.main_field,
            other_fields=self.other_fields, 
            handle_pics=self.use_images,
            input__=user_reports,
            mode='peruser'
        )
        vc_report = vc.GetReport()
        return vc_report

    def _default(self, reddit_address, _template=None, _posts=None):
        """
        Handles default classification (subreddit level).

        Args:
            reddit_address (str): The subreddit address (e.g., 'r/someSub').
            _template (str, optional): Path to the Jinja2 template. Defaults to None.
            _posts (list, optional): List of posts to classify. Defaults to None.

        Returns:
            str: The generated report.
        """
        _template = self.template if _template is None else _template
        posts = _posts if _posts is not None else self.RedditHandler(self.token_limit, reddit_address)
        if len(posts) == 0:
            logger.warning('No posts retrieved for classification.')
            return 'User or subreddit has no posts!'

        logger.info(f"Retrieved {len(posts)} posts from {reddit_address}")

        vc = VigilantClassifier(
            config_dir=self.config_dir,
            template=_template, 
            main_field=self.main_field,
            other_fields=self.other_fields, 
            handle_pics=self.use_images,
            input__=posts,
            mode='default'
        )
        vc_report = vc.GetReport()
        return vc_report

    def main(self):
        """
        Main execution method for the AllEyesOnReddit class.
        Determines whether to perform per-user classification or default classification based on configuration.
        """
        if self.DO_PERUSER_CLASSIFY:
            if not self.reddit_address.startswith('r/'):
                logger.error("REDDIT PER USER CLASSIFIER USED BUT REDDIT ADDRESS IS NOT FOR SUBREDDIT")
                raise ce.CustomError("ERROR: REDDIT PER USER CLASSIFIER USED BUT REDDIT ADDRESS IS NOT FOR SUBREDDIT")
            
            report = self._per_user_classifier(self.reddit_address)
            print(report)
            export_path = os.path.join('export', 'vc_report_per_user.vigileye')
            common.DumpToText(report, export_path)
            logger.info(f"Per-user report saved to {export_path}")
        else:
            report = self._default(self.reddit_address)
            export_path = os.path.join('export', 'vc_report.vigileye')
            common.DumpToText(report, export_path)
            logger.info(f"Default report saved to {export_path}")

if __name__ == '__main__':
    try:
        eor = AllEyesOnReddit()
        eor.main()
    except ce.CustomError as ce_err:
        logger.error(f"CustomError: {ce_err}")
        sys.exit(1)
    except Exception as ex:
        logger.error(f"An unexpected error occurred: {ex}")
        sys.exit(1)
