from pelican import signals
from collections import defaultdict


def initialize_backlinks(generator):
    generator.backlinks = defaultdict(list)


def analyze_backlinks(generator):
    for article in generator.articles:
        exact_url = f'"{article.url}"'
        for other_article in generator.articles:
            if article != other_article and exact_url in other_article.content:
                generator.backlinks[article].append(other_article)


def article_generator_write_article(generator, content):
    if content in generator.backlinks:
        backlinks = generator.backlinks[content]
        content._content += "<hr><h4>Backlinks</h4><ul>"
        for backlink in backlinks:
            content._content += (
                f"<li><a href='{backlink.url}'>{backlink.title}</a></li>"
            )
        content._content += "</ul>"


def register():
    signals.article_generator_init.connect(initialize_backlinks)
    signals.article_generator_pretaxonomy.connect(analyze_backlinks)
    signals.article_generator_write_article.connect(article_generator_write_article)
