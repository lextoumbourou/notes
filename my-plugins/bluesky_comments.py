from urllib.parse import urlparse, parse_qs

from pelican import signals


def _extract_youtube_id(url):
    parsed = urlparse(url)
    if parsed.hostname == 'youtu.be':
        return parsed.path.lstrip('/')
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        qs = parse_qs(parsed.query)
        if 'v' in qs:
            return qs['v'][0]
    return None


def add_bluesky_metadata(generator):
    for article in generator.articles:
        if not hasattr(article, 'bluesky_post'):
            article.bluesky_post = None
        if not hasattr(article, 'mastodon_post'):
            article.mastodon_post = None
        if not hasattr(article, 'youtube_video'):
            article.youtube_video = None
        article.youtube_video_id = (
            _extract_youtube_id(article.youtube_video) if article.youtube_video else None
        )


def register():
    signals.article_generator_finalized.connect(add_bluesky_metadata)
