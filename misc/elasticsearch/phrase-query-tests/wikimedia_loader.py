import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import bz2
import json
from xml.etree.cElementTree import iterparse
import time

def load_wiki(max_items=None):
    dump = bz2.BZ2File('/Users/lex/code/elasticsearch-testing/enwiki-latest-pages-articles.xml.bz2')
    start = time.time()

    found = 0

    for event, elem in iterparse(dump):
        if elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':

            found += 1

            title = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
            page_id = elem.find('{http://www.mediawiki.org/xml/export-0.10/}id').text
            output_text = None
            revision = elem.find('{http://www.mediawiki.org/xml/export-0.10/}revision')
            if revision:
                text = revision.find('{http://www.mediawiki.org/xml/export-0.10/}text')
                output_text = text.text

            yield dict(_id=page_id, title=title, text=output_text, _index='wiki-test', _type='page')

            if max_items and found >= max_items:
                print("Found max item {0}".format(max_items))
                return


if __name__ == '__main__':
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
    bulk(es, load_wiki(1000))
    print("Total time to index: {0}".format(time.time() - start))
