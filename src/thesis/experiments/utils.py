import itertools
import logging
import os

from thesis.models import Models

MODELS = (
    Models.Lossy,
    Models.Standard,
    Models.NAryRelation,
    Models.Singleton,
)


class Degrees:
    No = "none"
    Minimal = "minimal"
    Complete = "complete"


DEGREES = (
    Degrees.No,
    Degrees.Minimal,
    Degrees.Complete,
)

logger = logging.getLogger('main')


def get_graph_name(dataset_name, r_degree, r_technique):
    if r_degree == Degrees.No or r_technique == Models.Lossy:
        r_degree = Degrees.No
        r_technique = Models.Lossy
    return f"{dataset_name}_{r_degree}_{r_technique}"


class Paths:
    dataset = None

    @staticmethod
    def set_dataset(dataset):
        if dataset is None:
            raise ValueError("Dataset name must be str, not None")
        Paths.dataset = dataset

    @staticmethod
    def experiments_dir():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../experiments"))

    @staticmethod
    def config():
        return os.path.join(Paths.experiments_dir(), "config.json")

    @staticmethod
    def query_dir():
        return os.path.join(Paths.experiments_dir(), "queries")

    @staticmethod
    def graph_dir():
        return os.path.join(Paths.experiments_dir(), "graphs")

    @staticmethod
    def get_query_fn(query_id, technique=Models.Lossy):
        if technique == Models.Lossy:
            query_fn = f"{query_id}.rq"
        else:
            query_fn = f"{query_id}_{technique}.rq"
            # return Paths.get_result_fn("complete", technique, query_id, round_no=1, suffix="rq")
        return os.path.join(Paths.query_dir(), query_fn)

    @staticmethod
    def get_query(query_id, technique=Models.Lossy):
        query_fn = Paths.get_query_fn(query_id, technique)
        with open(query_fn) as f:
            query = f.read().strip()
        return query

    @staticmethod
    def dataset_dir():
        return os.path.join(Paths.experiments_dir(), Paths.dataset)

    @staticmethod
    def result_dir():
        return os.path.join(Paths.dataset_dir(), "results")

    @staticmethod
    def relation_dir():
        return os.path.join(Paths.dataset_dir(), "relations")

    @staticmethod
    def result_csv():
        return os.path.join(Paths.dataset_dir(), "results.csv")

    @staticmethod
    def main_relation():
        return os.path.join(Paths.relation_dir(), "main.csv")

    @staticmethod
    def binary_relation():
        return os.path.join(Paths.relation_dir(), "binary.csv")

    @staticmethod
    def get_result_fn(r_degree, r_technique, query_id, round_no=None, suffix="csv"):
        fn = os.path.join(Paths.result_dir(), f"{r_degree}-{r_technique}-{query_id}-{{round}}.{suffix}")
        if round_no:
            fn = fn.format(round=round_no)
        return fn

    @staticmethod
    def log_file():
        return os.path.join(Paths.dataset_dir(), "debug.log")

    @staticmethod
    def gold_standard_dir():
        return os.path.join(Paths.dataset_dir(), "gold_standard")

    @staticmethod
    def get_gold_standard_file(query_id):
        return os.path.join(Paths.gold_standard_dir(), f"{query_id}.result.csv")

    @staticmethod
    def parameter_fn():
        return os.path.join(Paths.dataset_dir(), "parameters.json")

    @staticmethod
    def get_graph_fn(degree, technique, suffix="ttl"):
        graph_name = get_graph_name(Paths.dataset, degree, technique)
        return os.path.join(Paths.graph_dir(), f"{graph_name}.{suffix}")


def get_combinations(degrees=DEGREES, techniques=MODELS, queries=None):
    add_lossy = False

    degrees = set(degrees)
    if Degrees.No in degrees:
        add_lossy = True
        degrees.remove(Degrees.No)

    techniques = set(techniques)
    if Models.Lossy in techniques:
        add_lossy = True
        techniques.remove(Models.Lossy)

    if not queries:
        queries = [None]

    configurations = set(itertools.product(degrees, techniques, queries))

    if add_lossy:
        configurations.update(set(itertools.product([Degrees.No], [Models.Lossy], queries)))

    configurations = sorted(configurations)
    logger.debug(f"Computed configurations: {', '.join(str(x) for x in configurations)}")
    return configurations
