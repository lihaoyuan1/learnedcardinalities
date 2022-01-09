from endpoint import Endpoint


def generate_predicate():
    file_name = "../data/predicate.csv"
    repository = Endpoint('http://182.61.55.66:7200/repositories/lubm')

    predicates = repository.get_all_predicates()

    with open(file_name, 'w+') as f:
        for predicate in predicates:
            cardinality = repository.query_cardinality(
                'SELECT (COUNT(*) as ?count) WHERE { ?s <' + predicate + '> ?o . }')

            f.write(predicate + ',' + str(cardinality) + '\n')


if __name__ == "__main__":
    generate_predicate()
