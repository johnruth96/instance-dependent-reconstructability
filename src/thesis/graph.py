from itertools import permutations
from typing import Iterable

from rdflib import Namespace, Graph, RDF, Variable, BNode, ConjunctiveGraph, URIRef, Literal

PREFIX_ENTITY = "http://thesis.de/entity/"
PREFIX_REL = "http://thesis.de/relation/"
PREFIX_INTERNAL = "http://thesis.de/internal/"
PREFIX_GRAPHS = "http://thesis.de/graph/"

EntityNS = Namespace(PREFIX_ENTITY)
RelationNS = Namespace(PREFIX_REL)
InternalNS = Namespace(PREFIX_INTERNAL)
GraphNS = Namespace(PREFIX_GRAPHS)


def format_graph_name(name):
    return f"{PREFIX_GRAPHS}{name}"


def setup_graph(name=None):
    if name:
        g = ConjunctiveGraph(identifier=format_graph_name(name))
    else:
        g = Graph()
    g.namespace_manager.bind('graph', GraphNS)
    g.namespace_manager.bind('the', EntityNS)
    g.namespace_manager.bind('thr', RelationNS)
    g.namespace_manager.bind('thi', InternalNS)
    g.add((InternalNS.belongsTo, RDF.type, RDF.Property))
    return g


def slugify(value):
    return value.replace(' ', '_').strip()


def entity(value):
    return EntityNS[slugify(value)]


def predicate(value: str):
    return RelationNS[value]


def internal(value):
    return InternalNS[value]


def format_term(term):
    if isinstance(term, Variable):
        return f"?{term}"
    elif isinstance(term, BNode):
        return f"_:{term}"
    else:
        return f"<{term}>"


class BGP(set):
    def __init__(self, iterable=(), filters=None):
        super().__init__(iterable)
        if filters is None:
            self.filters = getattr(iterable, 'filters') if hasattr(iterable, 'filters') else set()
        else:
            self.filters = filters

    def add_filter(self, condition: str):
        if condition.startswith("("):
            self.filters.add(condition)
        else:
            self.filters.add(f"({condition})")

    @property
    def subjects(self):
        already_yield = set()
        for x, _, _ in self:
            if x not in already_yield:
                already_yield.add(x)
                yield x

    @property
    def predicates(self):
        already_yield = set()
        for _, x, _ in self:
            if x not in already_yield:
                already_yield.add(x)
                yield x

    @property
    def objects(self):
        already_yield = set()
        for _, _, x in self:
            if x not in already_yield:
                already_yield.add(x)
                yield x

    def to_sparql(self, with_brackets=False):
        block = "{\n" if with_brackets else ""
        for s, p, o in self:
            block += f"\t{format_term(s)} {format_term(p)} {format_term(o)} .\n"
        if self.filters:
            for cond in self.filters:
                block += f"\tFILTER {cond} .\n"
        block += "}" if with_brackets else ""
        return block

    def union(self, *s: Iterable):
        ns = BGP(self, filters=self.filters.copy())
        for iterable in s:
            for item in iterable:
                ns.add(item)
            if hasattr(iterable, 'filters'):
                for cond in iterable.filters:
                    ns.add_filter(cond)
        return ns

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        output = "BGP("
        output += ", ".join(f"{format_term(s)} {format_term(p)} {format_term(o)}" for s, p, o in self)
        output += ", ".join(f"{cond}" for cond in self.filters)
        output += ")"
        return output


def is_iso_under_mapping(pat1, pat2, h):
    for s, p, o in pat1:
        if not (h(s), h(p), h(o)) in pat2:
            return False

    return True


def mapping_func(x, lookup_dict):
    if isinstance(x, URIRef) or isinstance(x, Literal):
        return x

    if x in lookup_dict:
        return lookup_dict[x]

    return None


def is_isomorphic(bgp1: BGP, bgp2: BGP):
    if len(bgp1) != len(bgp2):
        return False

    vars_bgp1 = list({term for t in bgp1 for term in t if isinstance(term, Variable)})
    vars_bgp2 = list({term for t in bgp2 for term in t if isinstance(term, Variable)})

    if len(vars_bgp1) != len(vars_bgp2):
        return False

    for p in permutations(vars_bgp1):
        lookup_dict = dict(zip(p, vars_bgp2))
        if is_iso_under_mapping(bgp1, bgp2, lambda x: mapping_func(x, lookup_dict)):
            return True

    return False
