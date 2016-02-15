import sys
import time
import bz2
import logging
from xml.etree.cElementTree import iterparse

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


logging.basicConfig(level=logging.INFO)


def load_wiki(dump, max_items=None):
    found = 0
    for event, elem in iterparse(dump):
        if elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':

            found += 1

            title = elem.find(
                '{http://www.mediawiki.org/xml/export-0.10/}title').text
            page_id = elem.find(
                '{http://www.mediawiki.org/xml/export-0.10/}id').text
            output_text = None
            revision = elem.find(
                '{http://www.mediawiki.org/xml/export-0.10/}revision')
            if revision:
                text = revision.find(
                    '{http://www.mediawiki.org/xml/export-0.10/}text')
                output_text = text.text

            yield dict(
                _id=page_id, title=title, text=output_text,
                _index='wiki-test', _type='page')

            if max_items and found >= max_items:
                print("Found max item {0}".format(max_items))
                return

        elem.clear()


if __name__ == '__main__':
    try:
        file_path = sys.argv[1]
    except IndexError:
        logging.error('First argument should be file path.')
        sys.exit(1)

    try:
        max_items = sys.argv[2]
    except IndexError:
        max_items = None

    start = time.time()

    es = Elasticsearch()
    es.indices.delete('wiki-test', ignore=404)
    es.indices.create('wiki-test', {
        'settings': {
            'number_of_shards': 1
        },
        'mappings': {
            'page': {
                'properties': {
                    'title': {
                        'type': 'string'
                    },
                    'text': {
                        'type': 'string'
                    }
                }
            }
        }
    })

    print("Start index")
    bulk(es, load_wiki(bz2.BZ2File(file_path), max_items))
    print("Total time to index: {0}".format(time.time() - start))
