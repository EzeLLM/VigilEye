from apify_client import ApifyClient

#
client = ApifyClient("######")

#
run_input = {
    "author": "apify",
    "customMapFunction": "(object) => { return {...object} }",
    "end": "2021-07-02",
    "geocode": "37.7764685,-122.4172004,10km",
    "geotaggedNear": "Los Angeles",
    "inReplyTo": "webexpo",
    "includeSearchTerms": False,
    "maxItems": 2500,
    "mentioning": "elonmusk",
    "minimumFavorites": 5,
    "minimumReplies": 5,
    "minimumRetweets": 5,
    "onlyImage": False,
    "onlyQuote": False,
    "onlyTwitterBlue": False,
    "onlyVerifiedUsers": False,
    "onlyVideo": False,
    "placeObjectId": "96683cc9126741d1",
    "sort": "Latest",
    "start": "2021-07-01",
    "twitterHandles": [
        "HighDavidBrown"
    ],
    "withinRadius": "15km"
}


run = client.actor("######").call(run_input=run_input)

#
for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    print(item)