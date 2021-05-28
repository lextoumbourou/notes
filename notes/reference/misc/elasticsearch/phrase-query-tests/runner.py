import time
import logging
import sys

import numpy as np
from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO)


def runner(es, query, time_in_secs=60):
    """
    Start by clearing the cache. Then, run query until we hit or exceed ``time_in_secs``.
    
    Displays a plot of query time and returns some stats about them.
    """
    results = []

    es.indices.clear_cache('wiki-test')
    
    time_start = time.time()

    while (time.time() - time_start) < time_in_secs:
        r = es.search(index='wiki-test', body=query)
        results.append(r['took'])
    
    results = sorted(results)

    return results


if __name__ == '__main__':
    try:
        time_in_secs = int(sys.argv[1])
    except IndexError, ValueError:
        logging.error('First argument should be time to run in secs.')
        sys.exit(1)

    es = Elasticsearch()

    logging.info('Test #1: Baseline phrase query.')
    query = {
        'query': {
            'match_phrase': {
                'text': 'running the risk'
            }
        },
        'size': 20
    }
    results = runner(es, query, time_in_secs)
    logging.info(results)
    logging.info('Median: {0}, Max: {1}, Min: {2}'.format(
        np.median(np.array(results)), max(results), min(results)))

    logging.info('Test #2: Term query (min should match 100%).')
    query = {
        'query': {
            'match': {
                'text': {
                    'query': 'running the risk',
                    'minimum_should_match': '100%'
                }
            }
        },
        'size': 20
    }
    results = runner(es, query, time_in_secs)
    logging.info(results)
    logging.info('Median: {0}, Max: {1}, Min: {2}'.format(
        np.median(np.array(results)), max(results), min(results)))

    logging.info('Test #3: phrase query less requests')
    query = {
        'query': {
            'match_phrase': {
                'text': 'running the risk'
            }
        },
        'size': 5
    }
    results = runner(es, query, time_in_secs)
    logging.info(results)
    logging.info('Median: {0}, Max: {1}, Min: {2}'.format(
        np.median(np.array(results)), max(results), min(results)))

    logging.info('Test #4: rescore phrase query.')
    query = {
        'query': {
            'match': {
                'text': 'running the risk'
            }
        },
        'rescore': {
           'window_size': 50,
           'query': {
               'rescore_query' : {
                   'match_phrase': {
                       'text': {
                           'query': 'running the risk',
                       }
                   }
               },
               'query_weight' : 0.7,
               'rescore_query_weight' : 1.2
            }
         }
    }
    results = runner(es, query, time_in_secs)
    logging.info(results)
    logging.info('Median: {0}, Max: {1}, Min: {2}'.format(
        np.median(np.array(results)), max(results), min(results)))
