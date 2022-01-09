import logging
import time

from SPARQLWrapper import SPARQLWrapper, JSON

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger('endpoint')


class Endpoint:

    def __init__(self, url):
        self.repository = SPARQLWrapper(url)
        self.repository.setReturnFormat(JSON)

    def get_all_predicates(self):
        self.repository.setQuery('SELECT DISTINCT ?p WHERE { ?s ?p ?o . }')
        results = self.repository.query().convert()
        bindings = results['results']['bindings']
        predicates = list()
        for row in bindings:
            predicates.append(row['p']['value'])

        return predicates

    def get_total_triples(self):
        self.repository.setQuery('SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o . }')

        result = self.repository.query().convert()
        binding = result['results']['bindings'][0]

        return int(binding['count']['value'])

    def send_ask_query(self, query):
        t1 = time.time()

        self.repository.setQuery(query)
        result = self.repository.query().convert()

        t2 = time.time()

        logger.info('ask query execute time = {}s, result = {}, query = {}'.format("%.03f" % ((t2 - t1) / 1000),
                                                                                   result['boolean'], query))

        return bool(result['boolean'])

    def query_cardinality(self, query):
        t1 = time.time()

        self.repository.setQuery(query)
        result = self.repository.query().convert()
        binding = result['results']['bindings'][0]

        t2 = time.time()

        logger.info('cardinality query execute time = {}s, result = {}, query = {}'.format("%.03f" % ((t2 - t1) / 1000),
                                                                                           binding['count']['value'],
                                                                                           query))

        return int(binding['count']['value'])

    def random_query_with_predicate(self, predicate, offset, limit=1):
        t1 = time.time()

        self.repository.setQuery(
            'SELECT * WHERE { ?s <' + predicate + '> ?o . } OFFSET ' + str(offset) + ' LIMIT ' + str(limit))
        results = self.repository.query().convert()
        bindings = results['results']['bindings']

        t2 = time.time()

        logger.info('random query execute time = {}s'.format("%.03f" % ((t2 - t1) / 1000)))

        if len(bindings) == 0:
            return None, None

        row = bindings[0]
        return row['s']['value'], row['o']['value']

    def query_with_predicate(self, predicate):
        self.repository.setQuery('SELECT * WHERE { ?s <' + predicate + '> ?o . }')
        results = self.repository.query().convert()
        bindings = results['results']['bindings']

        so_list = list()

        for row in bindings:
            so_list.append([row['s']['value'], row['o']['value']])

        return so_list

    def query_by_offset(self, offset, limit=1000):
        self.repository.setQuery('SELECT * WHERE { ?s ?p ?o . } OFFSET ' + str(offset) + ' LIMIT ' + str(limit))
        results = self.repository.query().convert()
        bindings = results['results']['bindings']

        spo_list = list()

        for binding in bindings:
            spo_list.append([binding['s']['value'], binding['p']['value'], binding['o']['value']])

        return spo_list

    def query_one_by_offset(self, query, offset):
        self.repository.setQuery(query + ' OFFSET ' + str(offset) + ' LIMIT 1')
        results = self.repository.query().convert()

        result = {}
        binding = results['results']['bindings'][0]
        for head in results['head']['vars']:
            result[head] = binding[head]['value']

        return result
