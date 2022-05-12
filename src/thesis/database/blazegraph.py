import json
import logging

import requests

from thesis.database.base import GraphDatabase

logger = logging.getLogger('main')


class BlazeGraph(GraphDatabase):
    def __init__(self, **kwargs):
        self.port = kwargs["port"]
        self.timeout = kwargs.get("timeout", 0)
        self.base_url = f"http://localhost:{self.port}/blazegraph/sparql"

    def query_single(self, query: str, raw=False):
        logger.debug("Running query: {}".format(query.replace("\n", " ")))

        r = requests.get(self.base_url, timeout=self.timeout, params=dict(
            query=query,
            format="json",
        ))

        if r.status_code == 200:
            if raw:
                return r.text
            else:
                result_dict = json.loads(r.text)
                variables = result_dict["head"]["vars"]
                bindings = result_dict["results"]["bindings"]
                results = [tuple(binding[v]["value"] for v in variables) for binding in bindings]
                return results
        else:
            raise ValueError(f"ERROR {r.status_code}: {r.reason}. {r.text}")

    def load(self, filename, **kwargs):
        pass

    def dump(self, format='ttl'):
        raise NotImplementedError

    def dumps(self, filename, format='ttl'):
        with open(filename, "w") as f:
            f.write(self.dump(format))

    @staticmethod
    def serialize_results(results, filename):
        """
        Input is JSON format
        """
        result_dict = json.loads(results)
        variables = result_dict["head"]["vars"]
        bindings = result_dict["results"]["bindings"]

        with open(filename, "w") as f:
            f.write(",".join(variables) + "\n")
            for binding in bindings:
                f.write(",".join(binding[v]["value"] for v in variables) + "\n")

    def exp_get_count_single(self, result, raw=True):
        if not raw:
            raise NotImplementedError

        return int(json.loads(result)["results"]["bindings"][0]["count"]["value"])
