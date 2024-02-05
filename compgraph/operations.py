import heapq
import re
import typing as tp
from abc import abstractmethod, ABC
from copy import deepcopy
from datetime import datetime
from itertools import groupby, chain
from math import radians, sin, cos, sqrt, asin

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper(row)


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for _, group_rows in groupby(rows, key=lambda r: [r[k] for k in self.keys]):
            yield from self.reducer(tuple(self.keys), group_rows)


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable,
                 rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass

    def general_join(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        checked: bool = False
        rows_b_list: list[TRow] = list(rows_b)
        common_keys: set[tp.Any] = set()
        for row_a in rows_a:
            for row_b in rows_b_list:
                merged: TRow = {}
                for key, value in chain(row_a.items(), row_b.items()):
                    common_keys = set(row_a.keys()) & set(row_b.keys()) - set(keys) if not checked else common_keys
                    checked = True

                    if key in common_keys:
                        if key in row_a.keys() and key + self._a_suffix not in merged.keys():
                            merged[key + self._a_suffix] = value
                        elif key in row_b.keys():
                            merged[key + self._b_suffix] = value
                    else:
                        merged[key] = value

                yield merged


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        rows_a = rows
        rows_b = args[0]

        group_a, group_b = (groupby(rows_a, key=lambda r: [r[k] for k in self.keys]),
                            groupby(rows_b, key=lambda r: [r[k] for k in self.keys]))

        _none: tuple[None, list[tp.Any]] = (None, [])
        key_a, value_a = next(group_a, _none)
        key_b, value_b = next(group_b, _none)

        while (key_a is not None) and (key_b is not None):

            if key_a == key_b:
                yield from self.joiner(self.keys, value_a, value_b)
                key_a, value_a = next(group_a, _none)
                key_b, value_b = next(group_b, _none)

            elif key_a < key_b:
                yield from self.joiner(self.keys, value_a, [])
                key_a, value_a = next(group_a, _none)

            else:
                yield from self.joiner(self.keys, [], value_b)
                key_b, value_b = next(group_b, _none)

        while key_a is not None:
            yield from self.joiner(self.keys, value_a, [])
            key_a, value_a = next(group_a, _none)

        while key_b is not None:
            yield from self.joiner(self.keys, [], value_b)
            key_b, value_b = next(group_b, _none)


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = re.sub(r'([^\w\s]|_)+', '', row[self.column])
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str = r'\s+') -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        idx_start: int = 0
        for ptr in re.finditer(self.separator, row[self.column]):
            yield self._cut(row, idx_start, ptr.start())
            idx_start = ptr.end()

        yield self._cut(row, idx_start)

    def _cut(self, row: TRow, start: int, end: int | None = None) -> TRow:
        result: TRow = deepcopy(row)
        result[self.column] = row[self.column][start:] if end is None else row[self.column][start:end]
        return result


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        calc_res: float = 1
        for column in self.columns:
            calc_res *= row[column]
        row[self.result_column] = calc_res
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {column: row[column] for column in self.columns}


class Calculate(Mapper):
    """Calculate some operation for row"""

    def __init__(self, operation: tp.Callable[[TRow], tp.Any], result: str) -> None:
        self.operation = operation
        self.result = result

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result] = self.operation(row)
        yield row


class CalculateTime(Mapper):
    """Calculate time by week, hour for enter/leave time"""

    def __init__(self, enter_time: str, dt_format: str, weekday_result: str, hour_result: str) -> None:
        self.enter_time = enter_time
        self.dt_format = dt_format
        self.weekday_result = weekday_result
        self.hour_result = hour_result
        self.weekdays: list[str] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __call__(self, row: TRow) -> TRowsGenerator:
        dt: datetime = datetime.strptime(row[self.enter_time], self.dt_format)
        row[self.weekday_result] = self.weekdays[dt.weekday()]
        row[self.hour_result] = dt.hour
        yield row


class CalculateLength(Mapper):
    """Calculate length of route with using haversine distance"""

    def __init__(self, start_point: str, end_point: str, result_column: str) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.result_column = result_column
        self._r: int = 6373

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.result_column not in row:
            row[self.result_column] = self._calc_len(map(radians,
                                                         row[self.start_point] + row[self.end_point]))
            yield row
        else:
            yield row

    def _calc_len(self, args: tp.Iterable[float]) -> float:
        lon_start, lat_start, lon_end, lat_end = args
        return 2 * self._r * \
            asin(
                sqrt(
                    sin((lat_end - lat_start) / 2) ** 2 +
                    cos(lat_start) * cos(lat_end) * sin((lon_end - lon_start) / 2) ** 2))


class CalculateSpeed(Reducer):
    """Calculate average speed for route"""

    def __init__(self, length_column: str, enter_column: str, leave_column: str, dt_format: str,
                 result_column: str) -> None:
        self.length_column = length_column
        self.enter_column = enter_column
        self.leave_column = leave_column
        self.dt_format = dt_format
        self.result_column = result_column
        self.length_total: float = 0.0
        self.time_total: float = 0.0

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        first_row: TRow = next(iter(rows))
        group_dict: TRow = {key: first_row[key] for key in group_key}
        self.length_total = 0.0
        self.time_total = 0.0
        self._calc_speed(first_row)
        for row in rows:
            self._calc_speed(row)
        group_dict[self.result_column] = self.length_total / self.time_total
        yield group_dict

    def _calc_speed(self, row: TRow) -> None:
        td = datetime.strptime(row[self.leave_column], self.dt_format) - \
             datetime.strptime(row[self.enter_column], self.dt_format)
        self.time_total += (td.seconds + td.microseconds * 10 ** (-6)) / 3600
        self.length_total += row[self.length_column]


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        yield from heapq.nlargest(self.n, rows, key=lambda r: r[self.column_max])


def _calc_stats(row: TRow, key: str | None, word_stats: dict[str, int | float], step: int | float = 1) -> tp.Any:
    default_column: str = "ROWS_COUNT"
    item: tp.Any = row[key] if isinstance(key, str) else default_column
    word_stats[item] = word_stats.get(item, 0) + step
    return item


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        word_stats: dict[str, int | float] = {}
        first_row: TRow = next(iter(rows))
        _calc_stats(first_row, self.words_column, word_stats)
        total_words: int = 1

        group_dict: TRow = {key: first_row[key] for key in group_key}

        for row in rows:
            total_words += 1
            _calc_stats(row, self.words_column, word_stats)

        for word, value in word_stats.items():
            yield {self.words_column: word, self.result_column: (value / total_words)} | group_dict


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        key: str | None = group_key[0] if len(group_key) > 0 else None

        word_stats: dict[str, int | float] = {}
        first_row: TRow = next(iter(rows))
        _calc_stats(first_row, key, word_stats)

        group_dict: TRow = {key: first_row[key] for key in group_key}
        for row in rows:
            _calc_stats(row, key, word_stats)

        for _, value in word_stats.items():
            group_dict[self.column] = value
            yield group_dict


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        key: str | None = group_key[0] if len(group_key) > 0 else None

        word_stats: dict[str, int | float] = {}
        first_row: TRow = next(iter(rows))
        _calc_stats(first_row, key, word_stats, first_row[self.column])

        group_dict: TRow = {key: first_row[key] for key in group_key}

        for row in rows:
            _calc_stats(row, key, word_stats, row[self.column])

        for _, value in word_stats.items():
            group_dict[self.column] = value
            yield group_dict


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        yield from self.general_join(keys, rows_a, rows_b)


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        list_b = list(rows_b)
        if isinstance(rows_a, list):
            yield from list_b
        elif not list_b:
            yield from rows_a
        else:
            yield from self.general_join(keys, rows_a, list_b)


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        if isinstance(rows_b, list):
            yield from rows_a
        else:
            yield from self.general_join(keys, rows_a, rows_b)


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        if isinstance(rows_a, list):
            yield from rows_b
        else:
            yield from self.general_join(keys, rows_a, rows_b)
