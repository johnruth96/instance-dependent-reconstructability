from thesis.graph import BGP
from thesis.rewriting import QueryRewriter


# class LossyQueryRewriter(QueryRewriter):
#     @staticmethod
#     def reify_triples(bgp: BGP) -> BGP:
#         return bgp
#
#     def exclude_internal_matches(self, bgp: BGP) -> BGP:
#         return bgp
#
#     def rewrite(self, query_str):
#         return query_str
