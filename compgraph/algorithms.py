import json
import math

from . import Graph
from . import operations as ops


def _graph_from(input_stream_name: str, from_file: bool) -> 'Graph':
    return Graph.graph_from_file(
        input_stream_name,
        lambda line: json.loads(line)) if from_file else Graph.graph_from_iter(input_stream_name)


def _split_graph(graph: 'Graph', text_column: str) -> 'Graph':
    return graph.map(ops.FilterPunctuation(text_column)) \
        .map(ops.LowerCase(text_column)) \
        .map(ops.Split(text_column))


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count',
                     from_file: bool = False) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    graph = _graph_from(input_stream_name, from_file)
    return _split_graph(graph, text_column) \
        .sort(keys=[text_column]) \
        .reduce(ops.Count(count_column), keys=[text_column]) \
        .sort(keys=[count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf', from_file: bool = False) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""

    graph = _graph_from(input_stream_name, from_file)
    split_words = _split_graph(graph, text_column)

    column_docs_count: str = 'docs_count'
    count_docs = graph \
        .reduce(ops.FirstReducer(), keys=[doc_column]) \
        .reduce(ops.Count(column_docs_count), [])

    column_total: str = 'total'
    column_idf: str = 'idf'
    idf = split_words \
        .sort(keys=[doc_column, text_column]) \
        .reduce(ops.FirstReducer(), keys=[doc_column, text_column]) \
        .sort(keys=[text_column]) \
        .reduce(ops.Count(column_total), keys=[text_column]) \
        .join(ops.InnerJoiner(), count_docs, keys=[]) \
        .map(ops.Calculate(lambda row: math.log(row[column_docs_count] / row[column_total]), column_idf))

    column_tf: str = 'tf'
    tf = split_words \
        .reduce(ops.TermFrequency(text_column, column_tf), keys=[doc_column]) \
        .sort(keys=[text_column])

    result_graph = tf.join(ops.InnerJoiner(), idf, keys=[text_column]) \
        .map(ops.Product([column_idf, column_tf], result_column)) \
        .map(ops.Project([doc_column, text_column, result_column])) \
        .reduce(ops.TopN(result_column, 3), keys=[text_column])

    return result_graph


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi', from_file: bool = False) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    graph = _graph_from(input_stream_name, from_file)

    count_column: str = 'count'
    words_with_correct_len = _split_graph(graph, text_column) \
        .map(ops.Filter(lambda row: len(row[text_column]) > 4)) \
        .sort(keys=[doc_column, text_column])

    words_with_correct_count = words_with_correct_len \
        .reduce(ops.Count(count_column), keys=[doc_column, text_column]) \
        .map(ops.Filter(lambda row: row[count_column] > 1))

    words_satisfying_cond = words_with_correct_len \
        .join(ops.InnerJoiner(), words_with_correct_count, keys=[doc_column, text_column])

    def calc_freq(freq_column: str, keys: list[str]) -> 'Graph':
        return words_satisfying_cond \
            .reduce(ops.TermFrequency(text_column, freq_column), keys=keys) \
            .sort(keys=[text_column])

    freq_only_column: str = 'freq_only_doc_graph'
    freq_only_doc_graph = calc_freq(freq_only_column, keys=[doc_column])

    freq_all_column: str = 'freq_all_docs_graph'
    freq_all_docs_graph = calc_freq(freq_all_column, keys=[])

    return freq_only_doc_graph.join(ops.InnerJoiner(), freq_all_docs_graph, keys=[text_column]) \
        .map(ops.Calculate(lambda row: math.log(row[freq_only_column] / row[freq_all_column]), result_column)) \
        .map(ops.Project([doc_column, text_column, result_column])) \
        .sort(keys=[doc_column]) \
        .reduce(ops.TopN(result_column, 10), keys=[doc_column])


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start',
                      end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed', from_file: bool = False) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    time_format: str = '%Y%m%dT%H%M%S.%f'

    time_graph = _graph_from(input_stream_name_time, from_file)
    time = time_graph \
        .map(ops.CalculateTime(enter_time_column,
                               time_format,
                               weekday_result_column,
                               hour_result_column)) \
        .sort(keys=[edge_id_column])

    length_graph = _graph_from(input_stream_name_length, from_file)
    length_column: str = 'length_column'
    length = length_graph.map(ops.CalculateLength(start_coord_column,
                                                  end_coord_column,
                                                  length_column)) \
        .sort(keys=[edge_id_column])

    return time.join(ops.InnerJoiner(), length, keys=[edge_id_column]) \
        .sort(keys=[weekday_result_column, hour_result_column]) \
        .reduce(ops.CalculateSpeed(length_column,
                                   enter_time_column,
                                   leave_time_column,
                                   time_format,
                                   speed_result_column),
                keys=[weekday_result_column, hour_result_column])
