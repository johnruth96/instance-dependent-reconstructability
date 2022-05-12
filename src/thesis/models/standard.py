from rdflib import Variable

from thesis.graph import InternalNS, BGP
from thesis.rewriting import QueryRewriter


class StandardQueryRewriter(QueryRewriter):
    LAST_ID = 0

    def get_subject_predicate_pairs(self):
        query_reification = f"""
        PREFIX thi: <http://thesis.de/internal/>
        SELECT ?subject ?predicate
        {self.get_from_clause()}
        WHERE {{
            _:blank thi:type thi:Statement .
            _:blank thi:subject ?subject .
            _:blank thi:predicate ?predicate .
        }}"""
        return self.db.query(query_reification.strip())

    @staticmethod
    def _get_next_id():
        StandardQueryRewriter.LAST_ID = StandardQueryRewriter.LAST_ID + 1
        return StandardQueryRewriter.LAST_ID

    @staticmethod
    def get_id_node():
        return Variable(f"id_{StandardQueryRewriter._get_next_id()}")  # BNode()

    @staticmethod
    def get_stmt_node():
        return Variable(f"stmt_{StandardQueryRewriter._get_next_id()}")  # BNode()

    @staticmethod
    def reify_triples(triples):
        new_triples = BGP()

        id_node = StandardQueryRewriter.get_id_node()
        for s, p, o in triples:
            stmt_node = StandardQueryRewriter.get_stmt_node()
            new_triples.add((stmt_node, InternalNS.type, InternalNS.Statement))
            new_triples.add((stmt_node, InternalNS.subject, s))
            new_triples.add((stmt_node, InternalNS.predicate, p))
            new_triples.add((stmt_node, InternalNS.object, o))

            new_triples.add_filter(f'isBlank(?{stmt_node})')

            if len(triples) > 1:
                new_triples.add((stmt_node, InternalNS.id, id_node))

        if len(triples) > 1:
            new_triples.add_filter(f'isLiteral(?{id_node})')

        return new_triples
