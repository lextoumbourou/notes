from pelican import signals


def add_bluesky_metadata(generator):
    for article in generator.articles:
        if not hasattr(article, 'bluesky_post'):
            article.bluesky_post = None


def register():
    signals.article_generator_finalized.connect(add_bluesky_metadata)
