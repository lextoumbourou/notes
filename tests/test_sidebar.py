from dataclasses import dataclass
from datetime import datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jinja2 import Environment
from pelican import signals


spec = importlib.util.spec_from_file_location(
    "site_sidebar", Path(__file__).resolve().parents[1] / "my-plugins" / "sidebar.py"
)
sidebar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sidebar)


@dataclass(eq=False)
class Article:
    url: str
    date: datetime
    category: object


def article(name, day=1, category="permanent"):
    return Article(name + ".html", datetime(2026, 9, day), SimpleNamespace(name=category))


class SidebarTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        sidebar.register()
        self.addCleanup(signals.generator_init.disconnect, sidebar.initialize_generator)
        signals.generator_init.send(SimpleNamespace(env=self.env))

    def test_matches_old_template_for_collections_ties_and_active_page(self):
        old_selection = '''
            {% set posts = articles | selectattr("category.name", "equalto", "posts") | list %}
            {% set notes = articles | rejectattr("category.name", "equalto", "posts") | list %}
            {% set recent = (posts + notes) | sort(attribute='date', reverse=True) %}
        '''
        new_selection = "{% set recent = articles | recent_articles %}"
        links = '''{%- for item in recent[:10] -%}
            <a href="/{{ item.url }}"{% if item.url == output_file %} class="active"{% endif %}>{{ item.url }}</a>
            {%- endfor -%}
            {% if recent | length > 10 %}See more{% endif %}'''
        old = self.env.from_string(old_selection + links)
        new = self.env.from_string(new_selection + links)
        items = [article("note"), article("post", category="posts"),
                 article("newer", day=2), article("other-post", category="posts")]
        items += [article(f"extra-{number}") for number in range(9)]
        with patch.object(sidebar, "sorted", wraps=sorted, create=True) as sort:
            for collection in [items, items[:2], [], list(reversed(items[:4]))]:
                for output_file in ["note.html", "post.html", "not-listed.html"]:
                    values = {"articles": list(collection), "output_file": output_file}
                    self.assertEqual(old.render(**values).strip(), new.render(**values).strip())
            # Changing the active page or passing a copy of the same list must
            # reuse prepared data, while distinct collections stay separate.
            self.assertEqual(sort.call_count, 4)

    def test_mutating_collection_does_not_reuse_wrong_cached_order(self):
        recent = self.env.filters["recent_articles"]
        first, second, newest = article("first"), article("second"), article("newest", day=3)
        items = [first, second]
        self.assertEqual(recent(items), (first, second))
        items.reverse()
        self.assertEqual(recent(items), (second, first))
        items.append(newest)
        self.assertEqual(recent(items), (newest, second, first))

    def test_rebuild_observes_edits_even_when_article_objects_are_reused(self):
        first, second = article("first", day=2), article("second")
        items = [first, second]
        previous_filter = self.env.filters["recent_articles"]
        self.assertEqual(previous_filter(items), (first, second))
        second.date = datetime(2026, 9, 3)
        rebuilt = Environment()
        signals.generator_init.send(SimpleNamespace(env=rebuilt))
        self.assertIsNot(rebuilt.filters["recent_articles"], previous_filter)
        self.assertEqual(rebuilt.filters["recent_articles"](items), (second, first))


if __name__ == "__main__":
    unittest.main()
