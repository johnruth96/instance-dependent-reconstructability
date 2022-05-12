from rdflib import Variable

from thesis.graph import InternalNS, internal, RelationNS, BGP
from thesis.rewriting import QueryRewriter


def statement_predicate(pred):
    if isinstance(pred, Variable):
        return Variable(f"{str(pred)}_s")
    else:
        name = str(pred)[len(str(RelationNS)):]
        return internal(f"{name}_s")


def value_predicate(pred):
    if isinstance(pred, Variable):
        return Variable(f"{str(pred)}_v")
    else:
        name = str(pred)[len(str(RelationNS)):]
        return internal(f"{name}_v")


class NAryRelationQueryRewriter(QueryRewriter):
    LAST_ID = 0

    @staticmethod
    def _get_next_id():
        NAryRelationQueryRewriter.LAST_ID = NAryRelationQueryRewriter.LAST_ID + 1
        return NAryRelationQueryRewriter.LAST_ID

    @staticmethod
    def get_id_node():
        return Variable(f"id_{NAryRelationQueryRewriter._get_next_id()}")  # BNode()

    @staticmethod
    def get_stmt_node():
        return Variable(f"stmt_{NAryRelationQueryRewriter._get_next_id()}")  # BNode()

    def get_subject_predicate_pairs(self):
        query_reification = f"""
        PREFIX thi: <http://thesis.de/internal/>
        SELECT ?subject ?predicate
        {self.get_from_clause()}
        WHERE {{
            ?subject ?p_v _:blank .
            ?p_v thi:statementProperty ?predicate .
        }}"""
        return self.db.query(query_reification.strip())

    @staticmethod
    def reify_triples(triples: BGP) -> BGP:
        new_triples = BGP()
        predicates = set(p for _, p, _ in triples)

        # Replace edge with reified edge and connection to join id variable
        id_node = NAryRelationQueryRewriter.get_id_node()
        for s, p, o in triples:
            stmt_node = NAryRelationQueryRewriter.get_stmt_node()
            new_triples.add((s, statement_predicate(p), stmt_node))
            new_triples.add((stmt_node, value_predicate(p), o))

            new_triples.add_filter(f'isBlank(?{stmt_node})')

            if len(triples) > 1:
                new_triples.add((stmt_node, InternalNS.id, id_node))

        if len(triples) > 1:
            new_triples.add_filter(f'isLiteral(?{id_node})')

        # Add statement- and valueProperty links
        for pred in predicates:
            new_triples.add((
                statement_predicate(pred),
                InternalNS.statementProperty,
                pred
            ))
            new_triples.add((
                value_predicate(pred),
                InternalNS.valueProperty,
                pred
            ))

        return new_triples
