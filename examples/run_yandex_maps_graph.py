import json

import click
from compgraph import algorithms


@click.command()
@click.argument('input_time_filepath', type=str)
@click.argument('input_len_filepath', type=str)
@click.argument('output_filepath', type=str)
def main(input_time_filepath: str, input_len_filepath: str, output_filepath: str) -> None:
    graph = algorithms.yandex_maps_graph(input_stream_name_time=input_time_filepath,
                                         input_stream_name_length=input_len_filepath,
                                         enter_time_column='enter_time',
                                         leave_time_column='leave_time',
                                         edge_id_column='edge_id',
                                         start_coord_column='start',
                                         end_coord_column='end',
                                         weekday_result_column='weekday',
                                         hour_result_column='hour',
                                         speed_result_column='speed',
                                         from_file=True)

    result = graph.run()
    with open(output_filepath, 'w') as out:
        for row in result:
            print(json.dumps(row), file=out)


if __name__ == '__main__':
    main()
