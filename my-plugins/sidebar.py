"""Reuse recent-article lists while rendering a generator's pages."""

from functools import lru_cache

from pelican import signals


def initialize_generator(generator):
    # Generators are recreated for each run, including development rebuilds.
    # Keep the cache local so article edits cannot leave stale ordering behind.
    @lru_cache(maxsize=256)
    def sort_articles(articles):
        posts = [article for article in articles if article.category.name == "posts"]
        notes = [article for article in articles if article.category.name != "posts"]
        # Stable sorting preserves the template's posts-first order on ties.
        return tuple(sorted(posts + notes, key=lambda article: article.date, reverse=True))

    def recent_articles(articles):
        # Tag and category pages supply different collections. Key by their
        # contents and order, not the identity of the mutable input list.
        return sort_articles(tuple(articles))

    generator.env.filters["recent_articles"] = recent_articles


def register():
    signals.generator_init.connect(initialize_generator)
