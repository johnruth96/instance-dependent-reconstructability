import argparse

from thesis.decomposition.graphfactory import get_graph_factory_by_technique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_files', nargs="+")
    parser.add_argument('-o', '--output')
    parser.add_argument('-t', '--technique')
    parser.add_argument('-n', '--name')
    parser.add_argument('-c', '--complete', action="store_true")
    args = parser.parse_args()

    factory_cls = get_graph_factory_by_technique(args.technique)
    factory = factory_cls(graph_name=args.name, complete_reification=args.complete)

    if args.output.endswith("ttl"):
        fmt = "turtle"
    elif args.output.endswith("trig"):
        fmt = "trig"
    elif args.output.endswith("nq"):
        fmt = "nquads"
    else:
        raise ValueError("Format not supported")

    for fn in args.csv_files:
        factory.read_csv(fn)

    factory.serialize(args.output, output_format=fmt)


if __name__ == '__main__':
    main()
