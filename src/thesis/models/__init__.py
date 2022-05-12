# from thesis.models.lossy import LossyQueryRewriter
from thesis.models.nary import NAryRelationQueryRewriter
from thesis.models.singleton import SingletonQueryRewriter
from thesis.models.standard import StandardQueryRewriter


class MetaModel(type):
    def __contains__(cls, item):
        return item in (
            cls.Lossy,
            cls.Singleton,
            cls.NAryRelation,
            cls.Standard,
        )


class BaseModels(metaclass=MetaModel):
    pass


class Models(BaseModels):
    Lossy = "lossy"
    Singleton = "singleton"
    NAryRelation = "nary"
    Standard = "standard"


REWRITERS = {
    # Models.Lossy: LossyQueryRewriter,
    Models.Singleton: SingletonQueryRewriter,
    Models.NAryRelation: NAryRelationQueryRewriter,
    Models.Standard: StandardQueryRewriter,
}

# TRANSFORMER_FUNCTION = {
#     Models.Lossy: transform_lossy,
#     Models.Singleton: transform_singleton,
#     Models.NAryRelation: transform_nary_relation,
#     Models.Standard: transform_standard,
# }


def select_query_rewriter_by_model(model):
    return REWRITERS.get(model)


# def select_transform_func_by_model(model):
#     return TRANSFORMER_FUNCTION.get(model)
