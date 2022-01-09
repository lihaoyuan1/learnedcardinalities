import csv
import logging
import random
import urllib.error

import SPARQLWrapper.SPARQLExceptions
import numpy as np

from endpoint import Endpoint
from utils import normalization, standard_format

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger('train_data')


def load_predicates(file_name):
    predicates = list()
    with open(file_name, 'r') as f:
        rows = csv.reader(f)
        for row in rows:
            predicates.append(list(row))

    return predicates


def load_cache(file_name):
    query_cache = {}
    with open(file_name, 'r') as f:
        rows = csv.reader(f)
        for query, result in rows:
            query_cache[query] = int(result)

    return query_cache


def get_joins(bgps):
    joins = list()

    visited = list()
    for i, bgp in enumerate(bgps):
        name = bgp[1].split('#')[1]
        vertex = [name + '.s', name + '.o']
        if i > 0:
            joins.append([visited[random.randint(0, len(visited) - 1)], vertex[random.randint(0, 1)]])
        visited.extend(vertex)

    return joins


def get_variable_name(key, variable_dict, joins):
    if variable_dict.get(key) is not None:
        variable = variable_dict.get(key)
    else:
        variable = '?' + chr(ord('A') + len(set(variable_dict.values())))

        join_vertex = {key}
        for join in joins:
            if join[0] in join_vertex or join[1] in join_vertex:
                join_vertex.update(join)
        for k in join_vertex:
            variable_dict[k] = variable

    return variable


def get_cardinality_query(bgps, joins):
    query = 'SELECT (COUNT(*) AS ?count) WHERE { '

    variable_dict = {}
    for s, p, o in bgps:
        name = p.split('#')[1]
        if s is None:
            s = get_variable_name(name + '.s', variable_dict, joins)
        if o is None:
            o = get_variable_name(name + '.o', variable_dict, joins)
        query += standard_format(s) + ' ' + standard_format(p) + ' ' + standard_format(o) + ' . '

    query += '}'

    return query


def get_select_query(bgps, joins):
    query = 'SELECT * WHERE { '

    variable_dict = {}
    for s, p, o in bgps:
        name = p.split('#')[1]
        if s is None:
            s = get_variable_name(name + '.s', variable_dict, joins)
        if o is None:
            o = get_variable_name(name + '.o', variable_dict, joins)
        query += standard_format(s) + ' ' + standard_format(p) + ' ' + standard_format(o) + ' . '

    query += '}'

    return query, variable_dict


def query_cardinality(repository, cardinality_query, query_cache):
    if query_cache.get(cardinality_query) is not None:
        logger.info('命中缓存')
        return query_cache.get(cardinality_query), False

    result = repository.query_cardinality(cardinality_query)

    query_cache[cardinality_query] = result

    with open('../data/cache.csv', 'a') as f:
        f.write(cardinality_query + ',' + str(result) + '\n')

    return result, True


def random_binding_variable(bgps, joins, random_one, variable_dict):
    binding_num = 1 if len(joins) == 0 else random.randint(1, len(random_one) - len(joins))

    join_vertex = set()
    for join in joins:
        join_vertex.update(join)
    bindings = {}
    variable_keys = list(variable_dict.keys())
    random.shuffle(variable_keys)
    for key in variable_keys:
        if key not in join_vertex:
            bindings[key] = random_one.get(variable_dict.get(key)[1:])
        if len(bindings) == binding_num:
            break

    for bgp in bgps:
        name = bgp[1].split('#')[1]
        if bindings.get(name + '.s') is not None:
            bgp[0] = bindings.get(name + '.s')
        if bindings.get(name + '.o') is not None:
            bgp[2] = bindings.get(name + '.o')


def get_csv_format(bgps, joins, cardinality):
    csv_str = ''
    for i, bgp in enumerate(bgps):
        if i > 0:
            csv_str += '|'
        csv_str += bgp[1].split('#')[1]
    csv_str += ','
    for i, join in enumerate(joins):
        if i > 0:
            csv_str += '|'
        csv_str += join[0] + '=' + join[1]
    csv_str += ','
    for i, bgp in enumerate(bgps):
        if i > 0:
            csv_str += '|'
        if bgp[0] is not None:
            csv_str += bgp[1].split('#')[1] + '.s=' + bgp[0]
        if bgp[2] is not None:
            csv_str += bgp[1].split('#')[1] + '.o=' + bgp[2]
    csv_str += ','
    csv_str += str(cardinality)
    return csv_str


def random_sparql():
    # sparql连接对象
    repository = Endpoint('http://182.61.55.66:7200/repositories/lubm')
    # 获得所有谓词
    predicates = load_predicates('../data/predicate.csv')
    # 读取缓存
    query_cache = load_cache('../data/cache.csv')

    # 统计每个谓词的概率
    probability = normalization(np.array([int(cardinality) for predicate, cardinality in predicates]))

    # 生成指定数量的随机查询
    cnt = 78508
    while cnt > 0:
        # 生成包含join_num个连接的查询
        join_num = random.randint(0, 2)
        # 保证生成的查询有效
        while True:
            try:
                # 随机选择join_num+1个谓词
                random_predicates = [predicates[i] for i in
                                     np.random.choice(len(predicates), size=join_num + 1, replace=False, p=probability)]
                # 生成不绑定任何值的子查询
                bgps = list([[None, predicate, None] for predicate, _ in random_predicates])
                # 生成连接节点
                joins = get_joins(bgps)
                # 检查组合是否有返回值
                cardinality_query = get_cardinality_query(bgps, joins)
                cardinality, is_new = query_cardinality(repository, cardinality_query, query_cache)
                if cardinality > 0:
                    # 将不绑定任何属性的查询也写入训练集
                    if is_new:
                        with open('../data/train1.sparql', 'a') as train_sparql, open('../data/train1.csv',
                                                                                     'a') as train_csv:
                            train_sparql.write(get_cardinality_query(bgps, joins) + '\n')
                            train_csv.write(get_csv_format(bgps, joins, cardinality) + '\n')
                        cnt = cnt - 1
                    select_query, variable_dict = get_select_query(bgps, joins)
                    random_one = repository.query_one_by_offset(select_query, random.randint(0, cardinality - 1))
                    random_binding_variable(bgps, joins, random_one, variable_dict)
                    cardinality_query = get_cardinality_query(bgps, joins)
                    cardinality = repository.query_cardinality(cardinality_query)
                    with open('../data/train.sparql', 'a') as train_sparql, open('../data/train.csv', 'a') as train_csv:
                        train_sparql.write(cardinality_query + '\n')
                        train_csv.write(get_csv_format(bgps, joins, cardinality) + '\n')
                    break
            except (urllib.error.HTTPError, SPARQLWrapper.SPARQLExceptions.EndPointInternalError):
                logger.warn('端点异常')

        cnt = cnt - 1
        print(cnt)


if __name__ == "__main__":
    random_sparql()
