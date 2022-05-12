import functools
import itertools
import logging
import os
import re
from collections import defaultdict
from itertools import product
from typing import Tuple, Set, List, Generator, Union

from rdflib import Variable, BNode
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.parserutils import CompValue

from thesis.graph import EntityNS, RelationNS, BGP, format_graph_name, is_isomorphic, InternalNS

logger = logging.getLogger('main')

PROJECTION_PATTERN = re.compile(r"SELECT\s(.*?)\n")
FILTER_PATTERN = re.compile(r"FILTER\s?(.*?)\s?\.")


def parse_select_query(query_str) -> Tuple[BGP, str]:
    """
    Create a pair (BGP, Projection Str) of the query string.

    The BGP contains the triples and filter conditions.

    :param query_str: SPARQL Select Query
    :return: Pair (BGP, String of the projection)
    """

    def _get_triples(p: CompValue):
        if p.name == 'BGP':
            return p.triples
        else:
            return _get_triples(p.p)

    query_obj = prepareQuery(query_str, initNs=dict(thr=RelationNS, the=EntityNS))
    triples = _get_triples(query_obj.algebra.p)
    projection_str = PROJECTION_PATTERN.search(query_str).group(1)

    bgp = BGP(triples)
    filter_conditions = FILTER_PATTERN.findall(query_str)
    for cond in filter_conditions:
        bgp.add_filter(cond)

    return bgp, projection_str


def iter_partition(collection) -> Generator[List[List[str]], None, None]:
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in iter_partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def is_var(pred: str):
    return pred.startswith("?") or not pred.startswith("http://") or pred.startswith("$")


def combine_bgps(*bgps) -> Set[BGP]:
    # Caution: product() creates tuples
    cart_prod = product(*bgps)
    return set(functools.reduce(lambda x1, x2: x1.union(x2), x) for x in cart_prod)


class QueryRewriter:
    template_no_filter = os.path.join(os.path.dirname(__file__), "queries/query_template.txt")
    template_filter = os.path.join(os.path.dirname(__file__), "queries/query_template_filter.txt")
    template_co_predicates = os.path.join(os.path.dirname(__file__), "queries/query_co_predicates.txt")
    template_subjects = os.path.join(os.path.dirname(__file__), "queries/query_subjects.txt")

    def get_from_clause(self):
        if self.graph_name:
            return f"FROM <{format_graph_name(self.graph_name)}>"
        else:
            return ""

    def __init__(self, db=None, graph_name=None, use_optimizations=True):
        """
        Initialize the Query Rewriter

        :param db:
        """
        logger.debug(f"Initiliazing QueryRewriter with use_optimizations={str(use_optimizations)}")
        self.db = db
        self.graph_name = graph_name
        self.use_optimizations = use_optimizations
        self.co_predicate_sets = []
        self.rel_id_by_predicate = dict()
        self.predicate_mixed = set()
        self.predicate_only_star = set()
        self.predicate_only_reification = set()
        self.reification_by_subject = dict()

        with open(self.template_no_filter) as f:
            self.query_template_no_filter = f.read().strip()
        with open(self.template_filter) as f:
            self.query_template_filter = f.read().strip()
        if db:
            self.prepare_co_predicates()
            if use_optimizations:
                self.prepare_index()

    def get_subject_predicate_pairs(self):
        raise NotImplementedError

    def prepare_co_predicates(self):
        """
        Retrieve all co-predicates from the DB, grouped by their relation.
        """
        # 0) Retrieve co-predicates (ID of co-predicate set is its index in the list)
        with open(self.template_co_predicates) as f:
            query = f.read().strip()
            query = query.format(from_clause=self.get_from_clause())
        predicate_relation = self.db.query(query)
        predicates_by_relation = defaultdict(set)
        for pred, rel in predicate_relation:
            predicates_by_relation[rel].add(str(pred))
        self.co_predicate_sets = list(predicates_by_relation.values())
        if not self.co_predicate_sets:
            raise ValueError("No co-predicates retrieved")
        logger.debug(f"Retrieved {len(self.co_predicate_sets)} relations")
        logger.debug(f"Co-predicate sets: {', '.join(str(x) for x in self.co_predicate_sets)}")
        # 1) Build index: Predicate -> Relation ID
        for idx, co_predicates in enumerate(self.co_predicate_sets):
            self.rel_id_by_predicate.update({pred: idx for pred in co_predicates})

    def prepare_index(self):
        with open(self.template_subjects) as f:
            query = f.read().strip()
            query = query.format(from_clause=self.get_from_clause())

        sub_pred_star = self.db.query(query)
        sub_pred_reification = self.get_subject_predicate_pairs()

        # Index 2): Subject -> is reified
        self.reification_by_subject = {str(s): False for s, _ in sub_pred_star}
        for subject, _ in sub_pred_reification:
            self.reification_by_subject[str(subject)] = True

        # Indexes 3): Predicate reification state
        predicates_star = set(str(p) for _, p in sub_pred_star)
        predicates_reification = set(str(p) for _, p in sub_pred_reification)

        self.predicate_mixed = predicates_star.intersection(predicates_reification)
        self.predicate_only_star = predicates_star.difference(predicates_reification)
        self.predicate_only_reification = predicates_reification.difference(predicates_star)

        n_predicates = len(self.rel_id_by_predicate.keys())
        if len(self.predicate_only_star) + len(self.predicate_only_reification) + len(
                self.predicate_mixed) != n_predicates:
            logger.critical("Predicate partitioning error.")
            logger.debug(f"Star-only: {', '.join(str(p) for p in self.predicate_only_star)}")
            logger.debug(f"Reification-only: {', '.join(str(p) for p in self.predicate_only_reification)}")
            logger.debug(f"Mixed: {', '.join(str(p) for p in self.predicate_mixed)}")
            logger.debug(f"n_predicates: {n_predicates}")
            raise ValueError("Predicate initialization error.")

    def _partition_has_heterogeneous_part(self, partition):
        for part in partition:
            relation_ids = {self.rel_id_by_predicate[pred] for pred in part if not is_var(pred)}
            if len(relation_ids) >= 2:
                return True
        return False

    def _partition_splits_co_predicates(self, partition):
        for co_predicates in self.co_predicate_sets:
            part_ids = {idx for idx, part in enumerate(partition) if co_predicates.intersection(set(part))}
            if len(part_ids) >= 2:
                return True
        return False

    def _partition_has_oversized_part(self, partition):
        for part in partition:
            first_constant_pred = next((pred for pred in part if not is_var(pred)), None)
            if first_constant_pred:
                relation_id = self.rel_id_by_predicate[first_constant_pred]
                if len(part) > len(self.co_predicate_sets[relation_id]):
                    return True
        return False

    def iter_predicate_partitions(self, predicates: set):
        """
        Generator for the possible partitions of the predicates in the query.
        Exclude partitions which are not possible, e.g.,
        1. A part contains two predicates from two different relations (need to be apart)
        2. Two parts contain two predicates from the same relation (need to be together)
        3. A part contains more predicates than the relation has attributes

        :param predicates: Set of predicates for a subject
        :return: Partition of the predicate set
        """
        for partition in iter_partition(list(predicates)):
            # Check (1): a part contains two predicates which are not co-predicates
            if self._partition_has_heterogeneous_part(partition):
                continue

            # Check (2): Two parts contain two co-predicates
            if self._partition_splits_co_predicates(partition):
                continue

            # Check (3): A part has more predicates than the relation has attributes
            if self._partition_has_oversized_part(partition):
                continue

            yield partition
        return

    @staticmethod
    def reify_triples(bgp: BGP) -> BGP:
        """
        Reify a BGP
        """
        raise NotImplementedError

    def exclude_internal_matches(self, triples: BGP) -> BGP:
        """
        Add constraints for a flat BGP so that it does not match internal IRIs
        """
        predicates = set(t[1] for t in triples if isinstance(t[1], Variable))
        for pred in predicates:
            triples.add_filter(f"!STRSTARTS(STR(?{pred}), '{InternalNS}')")
        return triples

    def rewrite_select(self, query_str, skip_union=False):
        """
        Rewriting function for SELECT queries.
        """
        query_bgp, projection_str = parse_select_query(query_str)
        bgps_of_query: List[Set[BGP]] = list()

        # 1) Iterate over subjects (star BGPs)
        for subject in query_bgp.subjects:
            logger.debug(f"Creating BGP for subject: {subject}")
            bgps_of_subject: Set[BGP] = set()
            star_bgp = BGP((s, p, o) for s, p, o in query_bgp if s == subject)

            # 1) Create possible partitions of predicates (iterator/generator)
            predicates = {str(pred) for pred in star_bgp.predicates}
            partitions = self.iter_predicate_partitions(predicates)

            # 2) Iterate over predicate partitions
            for partition in partitions:
                logger.debug(f"Creating BGP for partition: {', '.join(str(x) for x in partition)}")
                bgps_of_parts: List[Set[BGP]] = list()

                # 3) Generate the BGPs for each part
                for part in partition:
                    entity_centric_bgp = BGP((s, p, o) for s, p, o in star_bgp if str(p) in part)

                    # Generate reified BGP
                    reified_bgp = self.reify_triples(entity_centric_bgp)

                    # Add constraints for variable predicate + variable subject/object
                    has_var_subject = isinstance(subject, Variable) or isinstance(subject, BNode)
                    has_var_predicate = any(isinstance(p, Variable) for p in entity_centric_bgp.predicates)
                    has_var_object = any(
                        isinstance(o, Variable) or isinstance(o, BNode) for o in entity_centric_bgp.objects
                    )
                    if has_var_predicate and (has_var_subject or has_var_object):
                        entity_centric_bgp = self.exclude_internal_matches(entity_centric_bgp)

                    # Optimization: Only select the required BGPs
                    if self.use_optimizations:
                        # TODO: Add subject-level index
                        constant_predicates = set(p for p in part if not is_var(p))
                        constant_predicate = str(constant_predicates.pop())
                        if constant_predicate in self.predicate_only_star:
                            bgps_of_parts.append({entity_centric_bgp})
                        elif constant_predicate in self.predicate_only_reification:
                            bgps_of_parts.append({reified_bgp})
                        else:
                            bgps_of_parts.append({entity_centric_bgp, reified_bgp})
                    else:
                        bgps_of_parts.append({entity_centric_bgp, reified_bgp})

                # 4) Generate and add BGPs for partition using the Cartesian product
                bgps_of_partition: Set[BGP] = combine_bgps(*bgps_of_parts)
                bgps_of_subject.update(bgps_of_partition)

            # 6) Add all possible BGPs of the subject to the list of BGPs per subject
            bgps_of_query.append(bgps_of_subject)

        # 2.6 Create final set of BGPs for the query using the Cartesian product
        bgps_final = list(combine_bgps(*bgps_of_query))

        # 2.7 OPTIMIZATIONS
        if self.use_optimizations:
            bgps_final = self.optimize_prune_duplicate_bgps(bgps_final)

        # 3) Write new query
        new_query = self.construct_select_query(projection_str, bgps_final, query_bgp.filters, skip_union=skip_union)
        return new_query

    def construct_select_query(self, projection_str: str, bgps: Set[BGP], filters, skip_union=False) -> Union[
        str, List[str]]:
        query_template = self.query_template_filter if filters else self.query_template_no_filter
        filter_block = " && ".join(filters)

        if skip_union:
            result = []
            for bgp in bgps:
                new_query = query_template.format(
                    projection=projection_str,
                    where_block=bgp.to_sparql().strip(),
                    filter_block=filter_block,
                    from_clause=self.get_from_clause(),
                ).strip()
                result.append(new_query)
        else:
            with_brackets = len(bgps) > 1
            where_block = " UNION ".join(bgp.to_sparql(with_brackets) for bgp in bgps)
            result = query_template.format(
                projection=projection_str,
                where_block=where_block.strip(),
                filter_block=filter_block,
                from_clause=self.get_from_clause(),
            ).strip()

        return result

    def rewrite(self, query_str, skip_union=False):
        """
        Assumptions: Query string is written in SPARQL and consists only of a BGP and FILTER constraints.

        :param skip_union:
        :param query_str: SPARQL query
        :return: SPARQL query in a rewritten form
        """
        return self.rewrite_select(query_str.strip(), skip_union=skip_union)

    def optimize_prune_duplicate_bgps(self, final_bgps):
        for bgp1, bgp2 in itertools.combinations(final_bgps, 2):
            if is_isomorphic(bgp1, bgp2):
                final_bgps.remove(bgp2)
        return final_bgps
