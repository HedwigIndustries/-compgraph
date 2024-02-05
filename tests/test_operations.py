import copy
import dataclasses
import math
import typing as tp

import pytest
from compgraph import operations as ops
from pytest import approx


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_ground_truth_items: tuple[int, ...] = (0,)


MAP_CASES = [
    MapCase(
        mapper=ops.Calculate(operation=lambda row: row['count'] ** 2, result='square'),
        data=[
            {'test_id': 1, 'count': 5},
            {'test_id': 2, 'count': 10}
        ],
        ground_truth=[
            {'test_id': 1, 'count': 5, 'square': 25},
            {'test_id': 2, 'count': 10, 'square': 100}
        ],
        cmp_keys=('test_id', 'count')
    ),
    MapCase(
        mapper=ops.Calculate(operation=lambda row: math.log2(row['count']), result='log2'),
        data=[
            {'test_id': 1, 'count': 16},
            {'test_id': 2, 'count': 256}
        ],
        ground_truth=[
            {'test_id': 1, 'count': 16, 'log2': 4},
            {'test_id': 2, 'count': 256, 'log2': 8}
        ],
        cmp_keys=('test_id', 'count')
    ),
    MapCase(
        mapper=ops.Calculate(operation=lambda row: row['sum'] / row['count'], result='divide'),
        data=[
            {'test_id': 1, 'sum': 16, 'count': 2},
            {'test_id': 2, 'sum': 10000, 'count': 20000}
        ],
        ground_truth=[
            {'test_id': 1, 'sum': 16, 'count': 2, 'divide': 8},
            {'test_id': 2, 'sum': 10000, 'count': 20000, 'divide': 0.5}
        ],
        cmp_keys=('test_id', 'count')
    ),
    MapCase(
        mapper=ops.CalculateTime('enter_time', '%Y%m%dT%H%M%S.%f', 'weekday', 'hour'),
        data=[
            {'leave_time': '20171020T112238.723000', 'enter_time': '20171020T112237.427000',
             'edge_id': 8414926848168493057},
            {'leave_time': '20171011T145553.040000', 'enter_time': '20171011T145551.957000',
             'edge_id': 8414926848168493057},
            {'leave_time': '20171020T090548.939000', 'enter_time': '20171020T090547.463000',
             'edge_id': 8414926848168493057},
            {'leave_time': '20171024T144101.879000', 'enter_time': '20171024T144059.102000',
             'edge_id': 8414926848168493057},
            {'leave_time': '20171022T131828.330000', 'enter_time': '20171022T131820.842000',
             'edge_id': 5342768494149337085},
            {'leave_time': '20171014T134826.836000', 'enter_time': '20171014T134825.215000',
             'edge_id': 5342768494149337085},
            {'leave_time': '20171010T060609.897000', 'enter_time': '20171010T060608.344000',
             'edge_id': 5342768494149337085},
            {'leave_time': '20171027T082600.201000', 'enter_time': '20171027T082557.571000',
             'edge_id': 5342768494149337085}
        ],
        ground_truth=[
            {'leave_time': '20171020T112238.723000', 'enter_time': '20171020T112237.427000',
             'edge_id': 8414926848168493057, 'weekday': 'Fri', 'hour': 11},
            {'leave_time': '20171011T145553.040000', 'enter_time': '20171011T145551.957000',
             'edge_id': 8414926848168493057, 'weekday': 'Wed', 'hour': 14},
            {'leave_time': '20171020T090548.939000', 'enter_time': '20171020T090547.463000',
             'edge_id': 8414926848168493057, 'weekday': 'Fri', 'hour': 9},
            {'leave_time': '20171024T144101.879000', 'enter_time': '20171024T144059.102000',
             'edge_id': 8414926848168493057, 'weekday': 'Tue', 'hour': 14},
            {'leave_time': '20171022T131828.330000', 'enter_time': '20171022T131820.842000',
             'edge_id': 5342768494149337085, 'weekday': 'Sun', 'hour': 13},
            {'leave_time': '20171014T134826.836000', 'enter_time': '20171014T134825.215000',
             'edge_id': 5342768494149337085, 'weekday': 'Sat', 'hour': 13},
            {'leave_time': '20171010T060609.897000', 'enter_time': '20171010T060608.344000',
             'edge_id': 5342768494149337085, 'weekday': 'Tue', 'hour': 6},
            {'leave_time': '20171027T082600.201000', 'enter_time': '20171027T082557.571000',
             'edge_id': 5342768494149337085, 'weekday': 'Fri', 'hour': 8}
        ],
        cmp_keys=('edge_id', 'weekday', 'hour')
    ),
    MapCase(
        mapper=ops.CalculateLength('start', 'end', 'length_column'),
        data=[
            {'start': [37.84870228730142, 55.73853974696249], 'end': [37.8490418381989, 55.73832445777953],
             'edge_id': 8414926848168493057},
            {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824],
             'edge_id': 5342768494149337085},
            {'start': [37.56963176652789, 55.846845586784184], 'end': [37.57018438540399, 55.8469259692356],
             'edge_id': 5123042926973124604},
            {'start': [37.41463478654623, 55.654487907886505], 'end': [37.41442892700434, 55.654839486815035],
             'edge_id': 5726148664276615162},
            {'start': [37.584684155881405, 55.78285809606314], 'end': [37.58415022864938, 55.78177368734032],
             'edge_id': 451916977441439743},
            {'start': [37.736429711803794, 55.62696328852326], 'end': [37.736344216391444, 55.626937723718584],
             'edge_id': 7639557040160407543},
            {'start': [37.83196756616235, 55.76662947423756], 'end': [37.83191015012562, 55.766647034324706],
             'edge_id': 1293255682152955894},
        ],
        ground_truth=[
            {'start': [37.84870228730142, 55.73853974696249], 'end': [37.8490418381989, 55.73832445777953],
             'edge_id': 8414926848168493057, 'length_column': approx(0.03202, 0.001)},
            {'start': [37.524768467992544, 55.88785375468433], 'end': [37.52415172755718, 55.88807155843824],
             'edge_id': 5342768494149337085, 'length_column': approx(0.04546, 0.001)},
            {'start': [37.56963176652789, 55.846845586784184], 'end': [37.57018438540399, 55.8469259692356],
             'edge_id': 5123042926973124604, 'length_column': approx(0.03564, 0.001)},
            {'start': [37.41463478654623, 55.654487907886505], 'end': [37.41442892700434, 55.654839486815035],
             'edge_id': 5726148664276615162, 'length_column': approx(0.04118, 0.001)},
            {'start': [37.584684155881405, 55.78285809606314], 'end': [37.58415022864938, 55.78177368734032],
             'edge_id': 451916977441439743, 'length_column': approx(0.12515, 0.001)},
            {'start': [37.736429711803794, 55.62696328852326], 'end': [37.736344216391444, 55.626937723718584],
             'edge_id': 7639557040160407543, 'length_column': approx(0.00607, 0.001)},
            {'start': [37.83196756616235, 55.76662947423756], 'end': [37.83191015012562, 55.766647034324706],
             'edge_id': 1293255682152955894, 'length_column': approx(0.00409, 0.001)},
        ],
        cmp_keys=('edge_id', 'length_column')
    )
]


@pytest.mark.parametrize('case', MAP_CASES)
def test_new_mappers(case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(case.data[case.mapper_item])
    mapper_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.mapper_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    mapper_result = case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_result, key=key_func) == sorted(mapper_ground_truth_rows, key=key_func)

    result = ops.Map(case.mapper)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(result, key=key_func) == sorted(case.ground_truth, key=key_func)


@dataclasses.dataclass
class ReduceCase:
    reducer: ops.Reducer
    reducer_keys: tuple[str, ...]
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    reduce_data_items: tuple[int, ...] = (0,)
    reduce_ground_truth_items: tuple[int, ...] = (0,)


REDUCE_CASES = [
    ReduceCase(
        reducer=ops.CalculateSpeed('length_column', 'enter_time', 'leave_time', '%Y%m%dT%H%M%S.%f', 'speed'),
        reducer_keys=('weekday', 'hour'),
        data=[
            {'leave_time': '20171020T112238.723000', 'enter_time': '20171020T112237.427000',
             'edge_id': 8414926848168493057, 'weekday': 'Fri', 'hour': 11,
             'length_column': 0.03202},
            {'leave_time': '20171011T145553.040000', 'enter_time': '20171011T145551.957000',
             'edge_id': 8414926848168493057, 'weekday': 'Wed', 'hour': 14,
             'length_column': 0.03202},
            {'leave_time': '20171020T090548.939000', 'enter_time': '20171020T090547.463000',
             'edge_id': 8414926848168493057, 'weekday': 'Fri', 'hour': 9,
             'length_column': 0.03202},
            {'leave_time': '20171024T144101.879000', 'enter_time': '20171024T144059.102000',
             'edge_id': 8414926848168493057, 'weekday': 'Tue', 'hour': 14,
             'length_column': 0.03202},
            {'leave_time': '20171022T131828.330000', 'enter_time': '20171022T131820.842000',
             'edge_id': 5342768494149337085, 'weekday': 'Sun', 'hour': 13,
             'length_column': 0.04546},
            {'leave_time': '20171014T134826.836000', 'enter_time': '20171014T134825.215000',
             'edge_id': 5342768494149337085, 'weekday': 'Sat', 'hour': 13,
             'length_column': 0.04546},
            {'leave_time': '20171010T060609.897000', 'enter_time': '20171010T060608.344000',
             'edge_id': 5342768494149337085, 'weekday': 'Tue', 'hour': 6,
             'length_column': 0.04546},
            {'leave_time': '20171027T082600.201000', 'enter_time': '20171027T082557.571000',
             'edge_id': 5342768494149337085, 'weekday': 'Fri', 'hour': 8,
             'length_column': 0.04546}
        ],
        ground_truth=[
            {'weekday': 'Fri', 'hour': 11, 'speed': approx(88.94444, 0.001)},
            {'weekday': 'Wed', 'hour': 14, 'speed': approx(106.43767, 0.001)},
            {'weekday': 'Fri', 'hour': 9, 'speed': approx(78.09756, 0.001)},
            {'weekday': 'Tue', 'hour': 14, 'speed': approx(41.50954, 0.001)},
            {'weekday': 'Sun', 'hour': 13, 'speed': approx(21.8577, 0.001)},
            {'weekday': 'Sat', 'hour': 13, 'speed': approx(100.95990, 0.001)},
            {'weekday': 'Tue', 'hour': 6, 'speed': approx(105.38055, 0.001)},
            {'weekday': 'Fri', 'hour': 8, 'speed': approx(62.2266, 0.001)}
        ],
        cmp_keys=('weekday', 'hour', 'speed')
    )
]


@pytest.mark.parametrize('case', REDUCE_CASES)
def test_new_reducer(case: ReduceCase) -> None:
    reducer_data_rows = [copy.deepcopy(case.data[i]) for i in case.reduce_data_items]
    reducer_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.reduce_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    reducer_result = case.reducer(case.reducer_keys, iter(reducer_data_rows))
    assert isinstance(reducer_result, tp.Iterator)
    assert sorted(reducer_result, key=key_func) == sorted(reducer_ground_truth_rows, key=key_func)

    result = ops.Reduce(case.reducer, case.reducer_keys)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(result, key=key_func) == sorted(case.ground_truth, key=key_func)
