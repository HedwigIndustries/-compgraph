import json
from itertools import islice, cycle
from operator import itemgetter
from pathlib import Path

import pytest
from click.testing import CliRunner
from compgraph import algorithms
from compgraph import operations as ops
# from examples.run_inverted_index_graph import run_inverted_index_graph
# from examples.run_pmi_graph import run_pmi_graph
# from examples.run_word_count import run_word_count
# from examples.run_yandex_maps_graph import run_yandex_maps_graph
from examples import run_word_count, run_inverted_index_graph, run_pmi_graph, run_yandex_maps_graph
from pytest import approx


def test_multiple_call_tf_idf() -> None:
    graph = algorithms.inverted_index_graph('texts', doc_column='doc_id', text_column='text', result_column='tf_idf')

    rows1 = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!!'}
    ]

    expected1 = [
        {'doc_id': 1, 'text': 'hello', 'tf_idf': approx(0.1351, 0.001)},
        {'doc_id': 1, 'text': 'world', 'tf_idf': approx(0.1351, 0.001)},
        {'doc_id': 2, 'text': 'little', 'tf_idf': approx(0.4054, 0.001)},
        {'doc_id': 3, 'text': 'little', 'tf_idf': approx(0.4054, 0.001)},
        {'doc_id': 4, 'text': 'hello', 'tf_idf': approx(0.1013, 0.001)},
        {'doc_id': 4, 'text': 'little', 'tf_idf': approx(0.2027, 0.001)},
        {'doc_id': 5, 'text': 'hello', 'tf_idf': approx(0.2703, 0.001)},
        {'doc_id': 5, 'text': 'world', 'tf_idf': approx(0.1351, 0.001)},
        {'doc_id': 6, 'text': 'world', 'tf_idf': approx(0.3243, 0.001)}
    ]

    result1 = graph.run(texts=lambda: iter(rows1))
    assert sorted(result1, key=itemgetter('doc_id', 'text')) == expected1

    rows2 = [
        {'doc_id': 1, 'text': 'hi!*%!@^'},
        {'doc_id': 2, 'text': 'aboba!*%!@^'},
        {'doc_id': 3, 'text': 'aboba AbObA !*%!@^ aboba'},
        {'doc_id': 4, 'text': 'aboba?!*%!@^ HI aBoBa BaObAB'},
        {'doc_id': 5, 'text': 'hi HI!*%!@^ baobab...'},
        {'doc_id': 6, 'text': '!*%!@^baobab? baobab... BAOBAB!!! BaoBaB!!*%!@^!! hi!!!*%!@^!!!'}
    ]

    expected2 = [
        {'doc_id': 1, 'text': 'hi', 'tf_idf': approx(0.40546510, 0.001)},
        {'doc_id': 2, 'text': 'aboba', 'tf_idf': approx(0.693147, 0.001)},
        {'doc_id': 3, 'text': 'aboba', 'tf_idf': approx(0.693147, 0.001)},
        {'doc_id': 4, 'text': 'aboba', 'tf_idf': approx(0.34657, 0.001)},
        {'doc_id': 4, 'text': 'baobab', 'tf_idf': approx(0.17328, 0.001)},
        {'doc_id': 4, 'text': 'hi', 'tf_idf': approx(0.10136, 0.001)},
        {'doc_id': 5, 'text': 'baobab', 'tf_idf': approx(0.23104, 0.001)},
        {'doc_id': 5, 'text': 'hi', 'tf_idf': approx(0.27031, 0.001)},
        {'doc_id': 6, 'text': 'baobab', 'tf_idf': approx(0.554517, 0.001)}
    ]

    result2 = graph.run(texts=lambda: iter(rows2))
    assert sorted(result2, key=itemgetter('doc_id', 'text')) == expected2


def test_multiple_call_pmi() -> None:
    graph = algorithms.pmi_graph('texts', doc_column='doc_id', text_column='text', result_column='pmi')

    docs1 = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!! HELLO!!!!!!!'}
    ]

    expected1 = [
        {'doc_id': 3, 'text': 'little', 'pmi': approx(0.9555, 0.001)},
        {'doc_id': 4, 'text': 'little', 'pmi': approx(0.9555, 0.001)},
        {'doc_id': 5, 'text': 'hello', 'pmi': approx(1.1786, 0.001)},
        {'doc_id': 6, 'text': 'world', 'pmi': approx(0.7731, 0.001)},
        {'doc_id': 6, 'text': 'hello', 'pmi': approx(0.0800, 0.001)}
    ]

    result1 = graph.run(texts=lambda: iter(docs1))

    assert list(result1) == expected1

    docs2 = [
        {'doc_id': 1, 'text': 'hi!*%!@^'},
        {'doc_id': 2, 'text': 'aboba!*%!@^'},
        {'doc_id': 3, 'text': 'aboba !!*%!@^!AbObA !*%!@^ aboba'},
        {'doc_id': 4, 'text': 'aboba?!*%!@^ HI aBoBa BaObAB'},
        {'doc_id': 5, 'text': 'hi HI!*%!@^ baobab...'},
        {'doc_id': 6, 'text': '!*%!@^baobab?!!*%!@^! baobab... BAOBAB!!! BaoBaB!!*%!@^!! hi!!!*%!@^!!!'}
    ]

    expected2 = [
        {'doc_id': 3, 'text': 'aboba', 'pmi': approx(0.5877, 0.001)},
        {'doc_id': 4, 'text': 'aboba', 'pmi': approx(0.58778, 0.001)},
        {'doc_id': 6, 'text': 'baobab', 'pmi': approx(0.81093, 0.001)}
    ]

    result2 = graph.run(texts=lambda: iter(docs2))

    assert list(result2) == expected2


def test_multiple_call_yandex_maps() -> None:
    graph = algorithms.yandex_maps_graph(
        'travel_time', 'edge_length',
        enter_time_column='enter_time', leave_time_column='leave_time', edge_id_column='edge_id',
        start_coord_column='start', end_coord_column='end',
        weekday_result_column='weekday', hour_result_column='hour', speed_result_column='speed'
    )

    lengths1 = [
        {'start': [37.84870228730142, 55.73853974696249], 'end': [37.8490418381989, 55.73832445777953],
         'edge_id': 8414926848168493057}
    ]
    times1 = [
        {'leave_time': '20171020T112238.723000', 'enter_time': '20171020T112237.427000',
         'edge_id': 8414926848168493057},
        {'leave_time': '20171011T145553.040000', 'enter_time': '20171011T145551.957000',
         'edge_id': 8414926848168493057},
        {'leave_time': '20171020T090548.939000', 'enter_time': '20171020T090547.463000',
         'edge_id': 8414926848168493057},
        {'leave_time': '20171024T144101.879000', 'enter_time': '20171024T144059.102000',
         'edge_id': 8414926848168493057}
    ]

    expected1 = [
        {'weekday': 'Fri', 'hour': 9, 'speed': approx(78.1070, 0.001)},
        {'weekday': 'Fri', 'hour': 11, 'speed': approx(88.9552, 0.001)},
        {'weekday': 'Tue', 'hour': 14, 'speed': approx(41.5145, 0.001)},
        {'weekday': 'Wed', 'hour': 14, 'speed': approx(106.4505, 0.001)}
    ]

    result1 = graph.run(travel_time=lambda: islice(cycle(iter(times1)), len(times1)),
                        edge_length=lambda: iter(lengths1))

    assert sorted(result1, key=itemgetter('weekday', 'hour')) == expected1

    lengths2 = [
        {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824],
         'edge_id': 5342768494149337085}
    ]
    times2 = [
        {'leave_time': '20171022T131828.330000', 'enter_time': '20171022T131820.842000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171014T134826.836000', 'enter_time': '20171014T134825.215000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171010T060609.897000', 'enter_time': '20171010T060608.344000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171027T082600.201000', 'enter_time': '20171027T082557.571000',
         'edge_id': 5342768494149337085}
    ]

    expected2 = [
        {'weekday': 'Fri', 'hour': 8, 'speed': approx(62.2322, 0.001)},
        {'weekday': 'Sat', 'hour': 13, 'speed': approx(100.9690, 0.001)},
        {'weekday': 'Sun', 'hour': 13, 'speed': approx(21.8577, 0.001)},
        {'weekday': 'Tue', 'hour': 6, 'speed': approx(105.3901, 0.001)}
    ]
    result2 = graph.run(travel_time=lambda: islice(cycle(iter(times2)), len(times2)),
                        edge_length=lambda: iter(lengths2))

    assert sorted(result2, key=itemgetter('weekday', 'hour')) == expected2


@pytest.fixture(scope='session')
def word_count_input_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp('data') / 'input.txt'
    rows = [
        {'doc_id': 1, 'text': 'hi!*%!@^'},
        {'doc_id': 2, 'text': 'aboba!*%!@^'},
        {'doc_id': 3, 'text': 'aboba AbObA !*%!@^ aboba'},
        {'doc_id': 4, 'text': 'aboba?!*%!@^ HI aBoBa BaObAB'},
        {'doc_id': 5, 'text': 'hi HI!*%!@^ baobab...'},
        {'doc_id': 6, 'text': '!*%!@^baobab? baobab... BAOBAB!!! BaoBaB!!*%!@^!! hi!!!*%!@^!!!'}
    ]
    with open(input_path, 'w') as file:
        for row in rows:
            print(json.dumps(row), file=file)
    return input_path


@pytest.fixture(scope='session')
def output_file(tmp_path_factory) -> Path:  # type: ignore
    output_path = tmp_path_factory.mktemp('data') / 'output.txt'
    return output_path


def test_word_count_from_file(word_count_input_file: Path, output_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(run_word_count.main, [word_count_input_file.as_posix(), output_file.as_posix()])
    assert result.exit_code == 0

    expected = [
        {'count': 5, 'text': 'hi'},
        {'count': 6, 'text': 'aboba'},
        {'count': 6, 'text': 'baobab'}
    ]

    output = ops.Read(output_file.as_posix(), lambda line: json.loads(line))()
    assert list(output) == expected


@pytest.fixture(scope='session')
def tf_idf_input_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp('data') / 'input.txt'
    rows = [
        {'doc_id': 1, 'text': 'hi!*%!@^'},
        {'doc_id': 2, 'text': 'aboba!*%!@^'},
        {'doc_id': 3, 'text': 'aboba AbObA !*%!@^ aboba'},
        {'doc_id': 4, 'text': 'aboba?!*%!@^ HI aBoBa BaObAB'},
        {'doc_id': 5, 'text': 'hi HI!*%!@^ baobab...'},
        {'doc_id': 6, 'text': '!*%!@^baobab? baobab... BAOBAB!!! BaoBaB!!*%!@^!! hi!!!*%!@^!!!'}
    ]
    with open(input_path, 'w') as file:
        for row in rows:
            print(json.dumps(row), file=file)
    return input_path


def test_tf_idf_from_file(tf_idf_input_file: Path, output_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(run_inverted_index_graph.main, [tf_idf_input_file.as_posix(), output_file.as_posix()])
    assert result.exit_code == 0

    expected = [
        {'doc_id': 1, 'text': 'hi', 'tf_idf': approx(0.40546510, 0.001)},
        {'doc_id': 2, 'text': 'aboba', 'tf_idf': approx(0.693147, 0.001)},
        {'doc_id': 3, 'text': 'aboba', 'tf_idf': approx(0.693147, 0.001)},
        {'doc_id': 4, 'text': 'aboba', 'tf_idf': approx(0.34657, 0.001)},
        {'doc_id': 4, 'text': 'baobab', 'tf_idf': approx(0.17328, 0.001)},
        {'doc_id': 4, 'text': 'hi', 'tf_idf': approx(0.10136, 0.001)},
        {'doc_id': 5, 'text': 'baobab', 'tf_idf': approx(0.23104, 0.001)},
        {'doc_id': 5, 'text': 'hi', 'tf_idf': approx(0.27031, 0.001)},
        {'doc_id': 6, 'text': 'baobab', 'tf_idf': approx(0.554517, 0.001)}
    ]

    output = ops.Read(output_file.as_posix(), lambda line: json.loads(line))()
    assert sorted(output, key=itemgetter('doc_id', 'text')) == expected


@pytest.fixture(scope='session')
def pmi_input_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp('data') / 'input.txt'
    rows = [
        {'doc_id': 1, 'text': 'hi!*%!@^'},
        {'doc_id': 2, 'text': 'aboba!*%!@^'},
        {'doc_id': 3, 'text': 'aboba !!*%!@^!AbObA !*%!@^ aboba'},
        {'doc_id': 4, 'text': 'aboba?!*%!@^ HI aBoBa BaObAB'},
        {'doc_id': 5, 'text': 'hi HI!*%!@^ baobab...'},
        {'doc_id': 6, 'text': '!*%!@^baobab?!!*%!@^! baobab... BAOBAB!!! BaoBaB!!*%!@^!! hi!!!*%!@^!!!'}
    ]
    with open(input_path, 'w') as file:
        for row in rows:
            print(json.dumps(row), file=file)
    return input_path


def test_pmi_from_file(pmi_input_file: Path, output_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(run_pmi_graph.main, [pmi_input_file.as_posix(), output_file.as_posix()])
    assert result.exit_code == 0

    expected = [
        {'doc_id': 3, 'text': 'aboba', 'pmi': approx(0.5877, 0.001)},
        {'doc_id': 4, 'text': 'aboba', 'pmi': approx(0.58778, 0.001)},
        {'doc_id': 6, 'text': 'baobab', 'pmi': approx(0.81093, 0.001)}
    ]

    output = ops.Read(output_file.as_posix(), lambda line: json.loads(line))()
    assert list(output) == expected


@pytest.fixture(scope='session')
def time_maps_input_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp('data') / 'input.txt'
    rows = [
        {'leave_time': '20171022T131828.330000', 'enter_time': '20171022T131820.842000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171014T134826.836000', 'enter_time': '20171014T134825.215000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171010T060609.897000', 'enter_time': '20171010T060608.344000',
         'edge_id': 5342768494149337085},
        {'leave_time': '20171027T082600.201000', 'enter_time': '20171027T082557.571000',
         'edge_id': 5342768494149337085}
    ]
    with open(input_path, 'w') as file:
        for row in rows:
            print(json.dumps(row), file=file)
    return input_path


@pytest.fixture(scope='session')
def len_maps_input_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp('data') / 'input.txt'
    rows = [
        {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824],
         'edge_id': 5342768494149337085}
    ]
    with open(input_path, 'w') as file:
        for row in rows:
            print(json.dumps(row), file=file)
    return input_path


def test_yandex_maps_from_file(time_maps_input_file: Path, len_maps_input_file: Path, output_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(run_yandex_maps_graph.main,
                           [time_maps_input_file.as_posix(), len_maps_input_file.as_posix(), output_file.as_posix()])
    assert result.exit_code == 0

    expected = [
        {'weekday': 'Fri', 'hour': 8, 'speed': approx(62.2322, 0.001)},
        {'weekday': 'Sat', 'hour': 13, 'speed': approx(100.9690, 0.001)},
        {'weekday': 'Sun', 'hour': 13, 'speed': approx(21.8577, 0.001)},
        {'weekday': 'Tue', 'hour': 6, 'speed': approx(105.3901, 0.001)}
    ]

    output = ops.Read(output_file.as_posix(), lambda line: json.loads(line))()
    assert sorted(output, key=itemgetter('weekday', 'hour')) == expected
