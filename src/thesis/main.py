import argparse

from database.blazegraph import BlazeGraph
from database.rdflib import RDFLibDB
from thesis.models import select_query_rewriter_by_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', metavar='FILE')
    parser.add_argument('--model')
    parser.add_argument('--query', metavar='QUERY_FILE')
    parser.add_argument('--db', default="rdflib")
    parser.add_argument('--name', default=None)
    args = parser.parse_args()

    if args.db == "blazegraph":
        db = BlazeGraph(port=8080)
    elif args.db == "rdflib":
        db = RDFLibDB()
    else:
        raise ValueError("Database does not exist")

    if args.load:
        db.load(args.load)

    if args.query:
        with open(args.query) as f:
            query = f.read().strip()

        # Rewrite query
        rewriter_klass = select_query_rewriter_by_model(args.model)
        qr = rewriter_klass(db=db, graph_name=args.name)
        rewritten_query = qr.rewrite(query)

        print(20 * "=" + " ORIGINAL " + 20 * "=")
        print(query)
        print(20 * "=" + " REWRITTEN " + 20 * "=")
        print(rewritten_query)
        print(51 * "=")

        # Query DB
        r = db.query(rewritten_query)
        for idx, row in enumerate(r):
            print(idx, row)

    db.close()


if __name__ == '__main__':
    main()
