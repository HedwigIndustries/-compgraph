import typing as tp
from . import external_sort
from . import operations as ops


class Graph:
    """Computational graph implementation"""

    def __init__(self, op: ops.Operation, prev_node: tp.Union['Graph', None] = None) -> None:
        self._op = op
        self._join_graph: tp.Union['Graph', None] = None
        self._prev_node = prev_node

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        return Graph(ops.ReadIterFactory(name))

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        return Graph(ops.Read(filename, parser))

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        return Graph(ops.Map(mapper), self)

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        return Graph(ops.Reduce(reducer, keys=keys), self)

    # fix
    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        return Graph(external_sort.ExternalSort(keys=keys), self)

    # fix
    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        graph = Graph(ops.Join(joiner, keys=keys), self)
        graph._join_graph = join_graph
        return graph

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        if self._join_graph is not None and self._prev_node is not None:
            yield from self._op(self._prev_node.run(**kwargs), self._join_graph.run(**kwargs))
        elif self._prev_node is None:
            yield from self._op(**kwargs)
        elif self._prev_node is not None:
            yield from self._op(self._prev_node.run(**kwargs))
