import argparse
import json
import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas
import pandas as pd
import requests
from pandas import DataFrame

from thesis.database.blazegraph import BlazeGraph
from thesis.experiments.utils import Degrees, Paths
from thesis.models import select_query_rewriter_by_model

logger = logging.getLogger('main')
SPARQL_TRIPLE_REGEX = re.compile("\s*[\?\<]?[\w:\/\.#_]+\>?\s[\?\<]?[\w:\/\.#_]+\>?\s[\?\<]?[\w:\/\.#_]+\>?\s\.")
INDEX_COLUMNS = [
    'dataset',
    'r_degree',
    'r_technique',
    'query',
    'round',
]


def run_query(dataset_name, query_id, r_degree, r_technique, database, n_rounds, use_optimizations=True):
    # Select query
    if r_degree == Degrees.Minimal:
        query_fn_ori = Paths.get_query_fn(query_id)
    else:
        query_fn_ori = Paths.get_query_fn(query_id, r_technique)

    with open(query_fn_ori) as f:
        query_ori = f.read()

    results = []
    qr = None

    if r_degree == Degrees.Minimal:
        logger.debug(f"Initializing QueryRewriter for {r_degree}/{r_technique}/{query_id}")
        rewriter_klass = select_query_rewriter_by_model(r_technique)
        qr = rewriter_klass(db=database, use_optimizations=use_optimizations)

    for round_no in range(1, n_rounds + 1):
        logger.info(f"Running {dataset_name}/{r_degree}/{r_technique}/{query_id}/{round_no}")

        db_results = None
        query = query_ori
        query_fn = query_fn_ori

        # Start rewriting
        start_rewrite = datetime.now()
        if r_degree == Degrees.Minimal:
            logger.debug("Rewriting query: {}".format(query.replace("\n", " ")))
            query = qr.rewrite(query)
            end_rewrite = datetime.now()
            query_fn = Paths.get_result_fn(r_degree, r_technique, query_id, round_no=round_no, suffix="rq")
            with open(query_fn, "w") as f:
                f.write(str(query))
            time_rewrite = (end_rewrite - start_rewrite).total_seconds()
        else:
            time_rewrite = 0

        # Start querying
        start_querying = datetime.now()
        try:
            db_results = database.query(query, raw=True)
            end_querying = datetime.now()
            time_query = (end_querying - start_querying).total_seconds()
            logger.info(f"Round {round_no} finished in {time_rewrite + time_query}")
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            logger.error(f"Request of round {round_no} timed out: {e}")
            time_query = -1

        n_results = None
        if time_query > 0:
            n_results = database.exp_get_count(db_results, raw=True)
            logger.debug(f"Retrieved {n_results} results")

        results.append(dict(
            round=round_no,
            time_query=time_query,
            time_rewrite=time_rewrite,
            query_file=query_fn,
            n_results=n_results,
        ))

    return results


def run_queries(dataset_name, config, r_degree, r_technique, queries):
    if os.path.exists(Paths.result_csv()):
        df_results = pd.read_csv(Paths.result_csv())
    else:
        df_results = DataFrame(columns=[
            'dataset',
            'r_degree',
            'r_technique',
            'query',
            'round',
            'time_query',
            'time_rewrite',
            'n_results',
            'query_file',
            'use_optimizations',
        ])
    df_results = df_results.set_index(INDEX_COLUMNS)

    database = BlazeGraph(port=config["port"], timeout=config["timeout"])

    for query_id in queries:
        results = run_query(dataset_name, query_id, r_degree, r_technique, database, config["rounds"],
                            use_optimizations=config['use_optimizations'])

        for result_dict in results:
            row_key = (dataset_name, r_degree, r_technique, query_id, result_dict["round"])
            col_key = ["time_query", "time_rewrite", "query_file", "n_results", "use_optimizations"]
            df_results.loc[row_key, col_key] = [
                result_dict["time_query"],
                result_dict["time_rewrite"],
                result_dict["query_file"],
                result_dict["n_results"],
                config['use_optimizations'],
            ]

        df_results.to_csv(Paths.result_csv())


def apply_precision(row, gold_standards):
    n_results = row['n_results']
    query_id = row["query"]

    if np.isnan(n_results):
        return None

    if int(n_results) == 0:
        return None

    n_results_gold = gold_standards[query_id]
    precision = n_results_gold / int(n_results)
    precision = precision if precision < 1.0 else 1.0
    return precision


def apply_recall(row, gold_standards):
    n_results = row['n_results']
    query_id = row["query"]

    if np.isnan(n_results):
        return None

    n_results_gold = gold_standards[query_id]

    if n_results_gold == 0:
        return None

    recall = int(n_results) / n_results_gold
    recall = recall if recall < 1.0 else 1.0
    return recall


def buf_count_newlines_gen(fname):
    def _make_gen(reader):
        b = reader(2 ** 16)
        while b:
            yield b
            b = reader(2 ** 16)

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count


def apply_query_triple_count(row):
    query_file = row['query_file']
    with open(query_file) as f:
        query = f.read()
    return len(SPARQL_TRIPLE_REGEX.findall(query))


def apply_graph_triple_count(row):
    r_degree = row["r_degree"]
    r_technique = row["r_technique"]

    graph_fn = Paths.get_graph_fn(r_degree, r_technique)

    return buf_count_newlines_gen(graph_fn) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--degree', required=True)
    parser.add_argument('--technique', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--time', action="store_true")
    parser.add_argument('--add-precision', action="store_true")
    parser.add_argument('--add-stats', action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset
    Paths.set_dataset(dataset_name)

    with open(Paths.config()) as f:
        config = json.load(f)

    # Ugly because I call the script using bash and I don't know a better way
    queries = args.queries.split(",")
    for query_id in queries:
        if query_id not in config["queries"]:
            raise ValueError(f"Query {query_id} not available")

    # Configure logging
    fmt = logging.Formatter('%(asctime)s - %(module)s:%(lineno)d:%(funcName)s - %(levelname)s: %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(Paths.log_file())
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)

    logger.debug("Starting evaluation")

    if args.time:
        logger.info("Running '--time'")
        os.makedirs(Paths.result_dir(), exist_ok=True)
        run_queries(dataset_name, config, args.degree, args.technique, queries)

    if args.add_precision:
        logger.info("Running '--add-precision'")
        df_results = pandas.read_csv(Paths.result_csv())

        # Compute precision
        gold_standards = {}
        for query_id in config["queries"]:
            gs_file = Paths.get_gold_standard_file(query_id)
            df_gold = pandas.read_csv(gs_file)
            gold_standards[query_id] = len(df_gold)
            logger.debug(f"Read gold standard for {query_id} from {gs_file}: {len(df_gold)} records")

        logger.debug("Computing precision")
        df_results["precision"] = df_results.apply(
            lambda x: apply_precision(x, gold_standards),
            axis=1
        )
        df_results["recall"] = df_results.apply(
            lambda x: apply_recall(x, gold_standards),
            axis=1
        )
        logger.debug(f"Writing results to {Paths.result_csv()}")
        df_results.to_csv(Paths.result_csv(), index=False)

    if args.add_stats:
        logger.info("Running '--add-stats'")
        df_results = pandas.read_csv(Paths.result_csv())

        df_results["query_triple_count"] = df_results.apply(apply_query_triple_count, axis=1)
        df_results["graph_triple_count"] = df_results.apply(apply_graph_triple_count, axis=1)
        logger.debug(f"Writing results to {Paths.result_csv()}")
        df_results.to_csv(Paths.result_csv(), index=False)


if __name__ == '__main__':
    main()
