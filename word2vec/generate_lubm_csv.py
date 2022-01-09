from prepare.endpoint import Endpoint

repository = Endpoint('http://182.61.55.66:7200/repositories/lubm')

offset = 0
limit = 10000

with open('../data/lubm100' + '.csv', 'w+') as f:
    while True:
        sentences = repository.query_by_offset(offset, limit)
        for s, p, o in sentences:
            f.write(s + ',' + p + ',' + o + '\n')

        offset += len(sentences)

        print(offset)

        if len(sentences) < limit:
            break
