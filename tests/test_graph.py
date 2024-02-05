from compgraph import external_sort
from compgraph import operations as ops
from compgraph.graph import Graph


def test_graph_from_file() -> None:
    pass


def test_graph_from_iter() -> None:
    rows = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'testing out stuff'}
    ]
    expected = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'testing out stuff'}
    ]
    graph = Graph.graph_from_iter('test_from_iter')
    assert isinstance(graph._op, ops.ReadIterFactory)
    assert graph._op.name == 'test_from_iter'
    assert graph._prev_node is None
    assert list(graph.run(test_from_iter=lambda: iter(rows))) == expected


def test_map() -> None:
    rows = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'testing out stuff'}
    ]
    mapper = ops.DummyMapper()
    expected = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'testing out stuff'}
    ]

    graph_before = Graph.graph_from_iter('test_map')
    graph_after = graph_before.map(mapper)
    assert isinstance(graph_after._op, ops.Map)
    assert graph_after._op.mapper == mapper
    assert graph_after._prev_node == graph_before
    assert list(graph_after.run(test_map=lambda: iter(rows))) == expected


def test_reduce() -> None:
    rows = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 1, 'text': 'testing out stuff'}
    ]
    reducer = ops.FirstReducer()
    keys = ['test_id']
    expected = [
        {'test_id': 1, 'text': 'one two three'}
    ]

    graph_before = Graph.graph_from_iter('test_reduce')
    graph_after = graph_before.reduce(reducer, keys)
    assert isinstance(graph_after._op, ops.Reduce)
    assert graph_after._op.reducer == reducer
    assert graph_after._prev_node == graph_before
    assert list(graph_after.run(test_reduce=lambda: iter(rows))) == expected


def test_sort() -> None:
    rows = [
        {'test_id': 1, 'text': 'testing out stuff'},
        {'test_id': 2, 'text': 'one two three'}
    ]
    keys = ['text']
    expected = [
        {'test_id': 2, 'text': 'one two three'},
        {'test_id': 1, 'text': 'testing out stuff'}
    ]

    graph_before = Graph.graph_from_iter('test_sort')
    graph_after = graph_before.sort(keys)
    assert isinstance(graph_after._op, external_sort.ExternalSort)
    assert graph_after._prev_node == graph_before
    assert list(graph_after.run(test_sort=lambda: iter(rows))) == expected


def test_join() -> None:
    rows_a = [
        {'test_id': 1, 'text': 'testing out stuff'},
        {'test_id': 2, 'text': 'one two three'}
    ]
    rows_b = [
        {'test_id': 1, 'comment': 'every day'},
        {'test_id': 2, 'comment': 'four five six'}
    ]
    joiner = ops.InnerJoiner()
    keys = ['test_id']

    expected = [
        {'test_id': 1, 'text': 'testing out stuff', 'comment': 'every day'},
        {'test_id': 2, 'text': 'one two three', 'comment': 'four five six'},
    ]
    graph_a_before = Graph.graph_from_iter('test_sort_a')
    graph_b_before = Graph.graph_from_iter('test_sort_b')
    graph_b_after = graph_b_before.join(joiner, graph_a_before, keys)
    assert isinstance(graph_b_after._op, ops.Join)
    assert graph_b_after._op.joiner == joiner
    assert graph_b_after._prev_node == graph_b_before
    assert graph_b_after._join_graph == graph_a_before
    assert list(graph_b_after.run(test_sort_a=lambda: iter(rows_a), test_sort_b=lambda: iter(rows_b))) == expected
