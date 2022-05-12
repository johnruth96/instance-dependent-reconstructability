from rdflib import Variable

from thesis.graph import PREFIX_REL, InternalNS, internal, RelationNS, BGP
from thesis.rewriting import QueryRewriter


def singleton_predicate(pred, rid):
    name = str(pred)[len(str(RelationNS)):]
    return internal(f"{name}-{rid}")


class SingletonQueryRewriter(QueryRewriter):
    LAST_ID = 0

    @staticmethod
    def _get_next_id():
        SingletonQueryRewriter.LAST_ID = SingletonQueryRewriter.LAST_ID + 1
        return SingletonQueryRewriter.LAST_ID

    @staticmethod
    def get_id_node():
        return Variable(f"id_{SingletonQueryRewriter._get_next_id()}")  # BNode()

    def get_subject_predicate_pairs(self):
        query_reification = f"""
        PREFIX thi: <http://thesis.de/internal/>
        SELECT ?subject ?predicate
        {self.get_from_clause()}
        WHERE {{
            ?subject ?singletonProperty ?o .
            ?singletonProperty thi:singletonPropertyOf ?predicate .
        }}"""
        return self.db.query(query_reification.strip())

    @staticmethod
    def reify_triples(triples):
        def get_pred_var(predicate, pid):
            pred_name = str(predicate) if isinstance(predicate, Variable) else str(predicate)[len(PREFIX_REL):]
            return Variable(pred_name + "_" + str(pid))

        new_triples = BGP()
        predicates = set(p for _, p, _ in triples)
        predicate_id = SingletonQueryRewriter._get_next_id()

        # Replace predicate with singleton property variable
        for s, p, o in triples:
            new_triples.add((s, get_pred_var(p, predicate_id), o))

        # Add singletonPropertyOf edge to original predicate
        for pred in predicates:
            new_triples.add((
                get_pred_var(pred, predicate_id),
                InternalNS.singletonPropertyOf,
                pred
            ))

        # Add a common Variable ?id for the join
        if len(triples) > 1:
            id_node = SingletonQueryRewriter.get_id_node()
            for pred in predicates:
                new_triples.add((
                    get_pred_var(pred, predicate_id),
                    InternalNS.id,
                    id_node
                ))
            new_triples.add_filter(f'isLiteral(?{id_node})')

        return new_triples
