import math
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesis.graph import format_graph_name, PREFIX_ENTITY, \
    PREFIX_REL, PREFIX_INTERNAL
from thesis.models import Models

RDF_PROPERTY = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Property>"

PRED_ID = f"<{PREFIX_INTERNAL}id>"
PRED_BELONGS_TO = f"<{PREFIX_INTERNAL}belongsTo>"
INT_TYPE = f"<{PREFIX_INTERNAL}type>"
INT_STATEMENT = f"<{PREFIX_INTERNAL}Statement>"
INT_SUBJECT = f"<{PREFIX_INTERNAL}subject>"
INT_PREDICATE = f"<{PREFIX_INTERNAL}predicate>"
INT_OBJECT = f"<{PREFIX_INTERNAL}object>"

SINGLETON_PROPERTY = f"<{PREFIX_INTERNAL}singletonPropertyOf>"
STATEMENT_PROPERTY = f"<{PREFIX_INTERNAL}statementProperty>"
VALUE_PROPERTY = f"<{PREFIX_INTERNAL}valueProperty>"

LITERAL_PATTERNS = [
    re.compile(r"\d+"),
    re.compile(r"\d{4}-\d{2}-\d{2}"),
]


def get_singleton_predicate(original_predicate, idx):
    return f"<{PREFIX_INTERNAL}{original_predicate[:-1].split('/')[-1]}-{idx}>"


def get_statement_property(original_predicate):
    return f"<{PREFIX_INTERNAL}{original_predicate[:-1].split('/')[-1]}_s>"


def get_value_property(original_predicate):
    return f"<{PREFIX_INTERNAL}{original_predicate[:-1].split('/')[-1]}_v>"


def is_value_allowed(value):
    return not isinstance(value, float) or not math.isnan(value)


def format_entity(entity: str):
    entity = entity.replace(" ", "_")
    return f"<{PREFIX_ENTITY}{entity}>"


def format_relation(relation: str):
    return f"<{PREFIX_REL}{relation}>"


def format_int(value: int):
    return f"\"{value}\"^^<http://www.w3.org/2001/XMLSchema#integer>"


def get_object(value):
    if isinstance(value, str):
        for pattern in LITERAL_PATTERNS:
            if pattern.match(value):
                return f"\"{value}\""
        return format_entity(value)
    elif isinstance(value, np.integer):
        return format_int(value)
    raise ValueError(f"Value {value} (type {type(value)}) not recognized")


class GraphFactory:
    def __init__(self, graph_name=None, complete_reification=False, no_reification=False):
        self.graph_name = graph_name
        if complete_reification and no_reification:
            raise ValueError
        self.complete_reification = complete_reification
        self.no_reification = no_reification
        self.triples = set()
        # Blank nodes
        self.bnode_count = 0
        # Multiple CSV support
        self.offset = 0
        # Setup
        self.setup()

    def setup(self):
        self.triples.add((PRED_ID, INT_TYPE, RDF_PROPERTY))

    def reify_tuple(self, t, predicate: str):
        raise NotImplementedError

    def get_join_id(self, value):
        return value + self.offset

    def read_csv_files(self, *filenames):
        for fn in filenames:
            self.read_csv(fn)

    def get_bnode(self):
        self.bnode_count += 1
        return f"_:B{self.bnode_count + self.offset}"

    def read_csv(self, fn):
        df = pd.read_csv(fn)

        if len(df) == 0:
            return

        subject_column = df.columns[0]
        attr_columns = df.columns[1:]
        df_fact_count = df.groupby([subject_column]).nunique(dropna=False).prod(axis=1)
        df_group_size = df.groupby([subject_column]).size()

        if not self.complete_reification and not self.no_reification:
            indexes_star = np.where(df_group_size == df_fact_count)[0]
            indexes_star = df_group_size.index[indexes_star]
            indexes_reification = np.where(df_group_size != df_fact_count)[0]
            indexes_reification = df_group_size.index[indexes_reification]
        elif self.no_reification:
            indexes_star = df_group_size.index
            indexes_reification = []
        else:
            indexes_star = []
            indexes_reification = df_group_size.index

        # Create "belongsTo" triples
        relation_bnode = self.get_bnode()
        for col in attr_columns:
            nq_pred = format_relation(col)
            self.triples.add((nq_pred, PRED_BELONGS_TO, relation_bnode))

        # Star decomposition
        for col in attr_columns:
            tuples = df[df[subject_column].isin(indexes_star)][[subject_column, col]].to_records(
                index=False)
            nq_pred = format_relation(col)
            self.triples.update(
                (format_entity(t[0]), nq_pred, get_object(t[1])) for t in tqdm(tuples) if is_value_allowed(t[1]))

        # Reification
        for col in attr_columns:
            tuples = df[df[subject_column].isin(indexes_reification)][[subject_column, col]] \
                .to_records(index=True)
            nq_pred = format_relation(col)
            for t in tqdm(tuples):
                if is_value_allowed(t[2]):
                    self.triples.update(self.reify_tuple(t, nq_pred))

        self.offset += len(df)

    def serialize(self, filename, output_format="nquads"):
        if output_format == "nquads":
            if not self.graph_name:
                raise ValueError(f"Graph name must not be empty")
            graph = f"<{format_graph_name(self.graph_name)}>"
            with open(filename, "w") as f:
                for s, p, o in tqdm(self.triples):
                    f.write(f"{s} {p} {o} {graph} .\n")
        elif output_format == "turtle":
            with open(filename, "w") as f:
                for s, p, o in tqdm(self.triples):
                    f.write(f"{s} {p} {o} .\n")
        else:
            raise ValueError("Format not supported")


class SingletonGraphFactory(GraphFactory):
    def setup(self):
        super().setup()
        self.triples.add((SINGLETON_PROPERTY, INT_TYPE, RDF_PROPERTY))

    def reify_tuple(self, t, predicate):
        join_id = self.get_join_id(t[0])
        pred_singleton = get_singleton_predicate(predicate, join_id)
        return {
            (format_entity(t[1]), pred_singleton, get_object(t[2])),
            (pred_singleton, PRED_ID, format_int(join_id)),
            (pred_singleton, SINGLETON_PROPERTY, predicate),
        }


class StandardGraphFactory(GraphFactory):
    def reify_tuple(self, t, predicate):
        stmt = self.get_bnode()
        join_id = self.get_join_id(t[0])
        return {
            (stmt, PRED_ID, format_int(join_id)),
            (stmt, INT_SUBJECT, format_entity(t[1])),
            (stmt, INT_PREDICATE, predicate),
            (stmt, INT_OBJECT, get_object(t[2])),
            (stmt, INT_TYPE, INT_STATEMENT),
        }


class NAryRelationGraphFactory(GraphFactory):
    def setup(self):
        super().setup()
        self.triples.add((STATEMENT_PROPERTY, INT_TYPE, RDF_PROPERTY))
        self.triples.add((VALUE_PROPERTY, INT_TYPE, RDF_PROPERTY))

    def reify_tuple(self, t, predicate):
        stmt = self.get_bnode()
        stmt_prop = get_statement_property(predicate)
        value_prop = get_value_property(predicate)
        join_id = self.get_join_id(t[0])
        return {
            (format_entity(t[1]), stmt_prop, stmt),
            (stmt, value_prop, get_object(t[2])),
            (stmt, PRED_ID, format_int(join_id)),

            (stmt_prop, STATEMENT_PROPERTY, predicate),
            (value_prop, VALUE_PROPERTY, predicate),
        }


def get_graph_factory_by_technique(technique: str):
    if technique == Models.Singleton:
        return SingletonGraphFactory
    elif technique == Models.Standard:
        return StandardGraphFactory
    elif technique == Models.Lossy:
        return GraphFactory
    elif technique == Models.NAryRelation:
        return NAryRelationGraphFactory
    else:
        raise ValueError("Technique not supported")
