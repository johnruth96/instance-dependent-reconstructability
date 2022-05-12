from rdflib import Graph

from database.base import GraphDatabase


class RDFLibDB(GraphDatabase):
    @staticmethod
    def serialize_results(results, filename):
        results.serialize(filename)

    def __init__(self, *args, **kwargs):
        self.graph = Graph()

    def query_single(self, query, raw=False):
        result = self.graph.query(query)
        return result

    def load(self, filename, format='ttl'):
        with open(filename) as f:
            self.graph.parse(f, format=format)

    def dump(self, format='ttl'):
        return self.graph.serialize(format=format)

    def dumps(self, filename, format='ttl'):
        with open(filename, "w") as f:
            f.write(self.dump(format))

    def close(self):
        pass
