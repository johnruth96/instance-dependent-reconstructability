import argparse
import itertools
import json
import os
import pprint
import random
import string

import pandas as pd

from thesis.decomposition.graphfactory import get_graph_factory_by_technique
from thesis.experiments.utils import get_combinations, get_graph_name, Degrees, Paths
from thesis.graph import BGP
from thesis.models import select_query_rewriter_by_model, Models
from thesis.rewriting import parse_select_query


def generate_dataset(n_entity_reification, n_entity_star, n_tuples_per_ent, n_columns_entity, n_columns_literal):
    def sample():
        max_value = (n_entity_reification + n_entity_star) * n_tuples_per_ent * (n_columns_entity + n_columns_literal)
        return random.randint(1, max_value)

    # Main relation
    subject_column = "x"
    columns_entity = [string.ascii_lowercase[i] for i in range(n_columns_entity)]
    columns_literals = [string.ascii_lowercase[n_columns_entity + i] for i in range(n_columns_literal)]
    columns = [subject_column] + columns_entity + columns_literals

    # Additional binary relations
    column_binary = string.ascii_lowercase[n_columns_entity + n_columns_literal + 1]
    columns_binary = [subject_column, column_binary]

    rows_main = []
    rows_binary = []
    n_generation_steps = (n_columns_entity + n_columns_literal) * (n_tuples_per_ent - 1) + 1
    for n in range(n_entity_star):
        subject = f"{subject_column}{n}"
        for i in range(n_generation_steps):
            line = [subject]
            if i == 0:
                line.extend(f"{name.upper()}{sample()}" for name in columns_entity)
                line.extend(str(sample()) for _ in columns_literals)
                rows_main.append(line)
            else:
                rows_binary.append([subject, sample()])

    for n in range(n_entity_star, n_entity_star + n_entity_reification):
        subject = f"{subject_column}{n}"
        for i in range(n_tuples_per_ent):
            line = [subject]
            line.extend(f"{name.upper()}{sample()}" for name in columns_entity)
            line.extend(str(sample()) for _ in columns_literals)
            rows_main.append(line)

    df_main = pd.DataFrame(rows_main, columns=columns)
    df_binary = pd.DataFrame(rows_binary, columns=columns_binary)
    return df_main, df_binary


def generate_gold_standard(csv_table, output_dir=None):
    df = pd.read_csv(csv_table)
    print(f"Read dataset with {len(df)} rows")
    if output_dir:
        print(f"Writing data to:", output_dir)

    # Q1
    result_q1 = df[['x', 'a', 'b']]
    if output_dir:
        result_q1.to_csv(os.path.join(output_dir, 'Q1.result.csv'), index=False)

    # Q2
    result_q2 = pd.merge(df, df, on='a')
    result_q2 = result_q2[['x_x', 'x_y', 'a', 'c_x', 'c_y']]
    result_q2 = result_q2.rename(columns={'x_x': 'x1', 'x_y': 'x2', 'c_x': 'c1', 'c_y': 'c2'})
    result_q2 = result_q2[result_q2['x1'] != result_q2['x2']]
    if output_dir:
        result_q2.to_csv(os.path.join(output_dir, 'Q2.result.csv'), index=False)

    # Q3
    result_q3 = df[['x', 'c']]
    if output_dir:
        result_q3.to_csv(os.path.join(output_dir, 'Q3.result.csv'), index=False)

    # Q4
    result_q4 = pd.merge(df, df, on='a')
    result_q4 = result_q4[['x_x', 'x_y', 'a']]
    result_q4 = result_q4.rename(columns={'x_x': 'x1', 'x_y': 'x2'})
    result_q4 = result_q4[result_q4['x1'] != result_q4['x2']]
    if output_dir:
        result_q4.to_csv(os.path.join(output_dir, 'Q4.result.csv'), index=False)


def rewrite_queries(config):
    techniques = (Models.Singleton, Models.Standard, Models.NAryRelation)
    combinations = itertools.product(config["queries"], techniques)

    for query_id, technique in combinations:
        print("Rewriting query", query_id, "for", technique)
        query_str = Paths.get_query(query_id)

        rewriter_klass = select_query_rewriter_by_model(technique)
        rewriter = rewriter_klass(db=None)

        query_bgp, variables = parse_select_query(query_str)
        final_bgq = BGP()

        for subject in query_bgp.subjects:
            bgp = BGP((s, p, o) for (s, p, o) in query_bgp if s == subject)
            rewritten_triples = rewriter.reify_triples(bgp)
            final_bgq.update(rewritten_triples)

        final_query = rewriter.construct_select_query(
            variables,
            {final_bgq},
            query_bgp.filters,
        )

        query_fn = Paths.get_query_fn(query_id, technique)
        with open(query_fn, "w") as f:
            print("Creating", query_fn, "...")
            f.write(final_query)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--n-ent', type=int, help="Number of entities", metavar="N")
    parser.add_argument('-r', '--r-rei', type=float, help="Ratio of entities with reification",
                        metavar="RATIO")
    parser.add_argument('-t', '--n-tup', type=int, help="Number of tuples per reified entity", metavar="N")
    parser.add_argument('-c', '--n-col', type=int, help="Number of columns/attributes")
    parser.add_argument('-l', '--r-lit', type=float, help="Ratio of columns/attributes with literals")

    parser.add_argument('-d', '--dataset', help="Dataset name")
    parser.add_argument('--queries', action="store_true", help="Rewrite queries")
    parser.add_argument('--relations', action="store_true", help="Generate relations")
    parser.add_argument('--graph', action="store_true", help="Generate graphs")
    parser.add_argument('--gold-standard', action="store_true", help="Generate Gold Standard")

    args = parser.parse_args()

    with open(Paths.config()) as f:
        config = json.load(f)

    # Queries
    if args.queries:
        os.makedirs(Paths.query_dir(), exist_ok=True)
        rewrite_queries(config)

    if args.relations or args.graph or args.gold_standard:
        # Paths and config
        dataset_name = args.dataset
        Paths.set_dataset(dataset_name)
        os.makedirs(Paths.dataset_dir(), exist_ok=True)

    # Relation
    if args.relations and not os.path.exists(Paths.relation_dir()):
        os.makedirs(Paths.relation_dir(), exist_ok=True)

        _n_entity_reified = int(args.n_ent * args.r_rei)
        _n_tuples_reified = _n_entity_reified * args.n_tup
        _n_tuples_star = args.n_ent - _n_entity_reified

        params = dict(
            n_entity_total=args.n_ent,
            n_entity_reified=_n_entity_reified,  # n_entities * ratio of reification
            n_entity_star=args.n_ent - _n_entity_reified,
            n_tuples_total=_n_tuples_star + _n_tuples_reified,
            n_tuples_reified=_n_tuples_reified,
            n_tuples_star=_n_tuples_star,
            n_attributes=args.n_col,
            n_attributes_entities=args.n_col - int(args.n_col * args.r_lit),
            n_attributes_literals=int(args.n_col * args.r_lit),
        )

        with open(Paths.parameter_fn(), "w") as f:
            print("Writing params to", Paths.parameter_fn())
            json.dump(params, f)

        # Generate dataset
        df_main, df_binary = generate_dataset(
            _n_entity_reified,
            params["n_entity_star"],
            args.n_tup,
            params["n_attributes_entities"],
            params["n_attributes_literals"],
        )

        print("Parameters:")
        pprint.pprint(params)

        # Write dataset
        print("Writing main dataset to", Paths.main_relation())
        df_main.to_csv(Paths.main_relation(), index=False)
        print("Writing binary dataset to", Paths.binary_relation())
        df_binary.to_csv(Paths.binary_relation(), index=False)
    elif args.relations and os.path.exists(Paths.relation_dir()):
        print("Datasets in", Paths.relation_dir(), "already exist. Skipping.")

    # Graphs
    if args.graph:
        os.makedirs(Paths.graph_dir(), exist_ok=True)

        for degree, technique, _ in get_combinations():
            graph_name = get_graph_name(dataset_name, degree, technique)
            output_file = Paths.get_graph_fn(degree, technique)

            factory_cls = get_graph_factory_by_technique(technique)
            factory = factory_cls(
                complete_reification=degree == Degrees.Complete,
                no_reification=degree == Degrees.No,
            )

            print(f"Creating graph {graph_name}")
            factory.read_csv_files(Paths.main_relation(), Paths.binary_relation())
            factory.serialize(output_file, output_format="turtle")

    # Gold Standard
    if args.gold_standard:
        os.makedirs(Paths.gold_standard_dir(), exist_ok=True)
        generate_gold_standard(Paths.main_relation(), Paths.gold_standard_dir())


if __name__ == '__main__':
    main()
