from typing import Union, List


class GraphDatabase:
    def query_single(self, query, raw=False):
        raise NotImplementedError

    def query(self, query: Union[str, List[str]], raw=False):
        if isinstance(query, list):
            results = []
            for q in query:
                results.append(self.query_single(q, raw=raw))
            return results
        else:
            return self.query_single(query, raw=raw)

    def load(self, filename, format='ttl'):
        raise NotImplementedError

    def dump(self, format='ttl'):
        raise NotImplementedError

    def dumps(self, filename, format='ttl'):
        raise NotImplementedError

    def close(self):
        pass

    @staticmethod
    def serialize_results(results, filename):
        raise NotImplementedError

    # Experiments only
    def exp_get_count_single(self, result, raw=True):
        raise NotImplementedError

    # Experiments only
    def exp_get_count(self, result_obj, raw=True):
        if not raw:
            raise NotImplementedError

        if isinstance(result_obj, str):
            result_obj = [result_obj]

        count = 0
        for result in result_obj:
            count += self.exp_get_count_single(result)
        return count
