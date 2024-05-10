import json
import requests
from pyyoutube import Api

# from logging.logger_setup import setup_logger # type: ignore
# logger = setup_logger(__name__, '../logging/logging.log')

# logger.info('Your log message')


def get_data(YOUTUBE_API_KEY, video_id, max_results, next_page_token):
    youtube_uri = f'https://www.googleapis.com/youtube/v3/commentThreads?key={YOUTUBE_API_KEY}&textFormat=plainText&' + \
        f'part=snippet&videoId={video_id}&maxResults={max_results}&pageToken={next_page_token}'
    
    # logger.info('Connecting by API')
    try: 
        response = requests.get(youtube_uri)
        response.raise_for_status()
        data = json.loads(response.text)
        # logger.info('Successful connection')
    except Exception as e:
        # logger.error('Failed connection: %s', e)
        return None
    return data


def get_text_of_comment(data):
    comms = set()
    for item in data['items']:
        comm = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comms.add(comm)
    return comms


def get_all_comments(YOUTUBE_API_KEY, query, count_video=10, limit=30, max_results=10, next_page_token=''):
    api = Api(api_key=YOUTUBE_API_KEY)
    video_by_keywords = api.search_by_keywords(q=query,
                                               search_type=["video"],
                                               count=count_video,
                                               limit=limit
    )
    video_ids = [x.id.videoId for x in video_by_keywords.items]

    comments_all = []
    for video_id in video_ids:
        try:
            # logger.info('Connecting to YouTube API for video ID: %s', video_id)
            data = get_data(YOUTUBE_API_KEY,
                            video_id=video_id,
                            max_results=max_results,
                            next_page_token=next_page_token
            )
            # logger.info('Successfully fetched data for video ID: %s', video_id)
            
            if 'items' in data:
                comment = list(get_text_of_comment(data))
                comments_all.append(comment)
            # else:
                # logger.info('No comments found for video ID: %s', video_id)
        except Exception as e:
            # logger.error('Failed to fetch data for video ID: %s. Error: %s', video_id, e)
            continue

    comments = sum(comments_all, [])
    return comments