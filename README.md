# Compute Graph Library

Computational graphs make it possible to separate the description of a sequence of operations from their execution.
Thanks to this, you can both run operations in another environment (for example, describe a graph in a Python
interpreter, and then select a video card), and independently and in parallel launch multiple machines of a computing
cluster to process a large array of input data in an adequate finite time (for example, so the client works for the
Spark distributed computer system.

### Getting Started

Start with cloning repository with ssh-key.

### Prerequisites ans Dependencies

Check pyproject.toml.
P.S. You should install pytest

## Installing

To use library you should install it, use:

```bash
pip install -e compgraph
```

## Example usage

This is what a graph that counts the number of words in documents might look like (although who am I trying to fool,
this is what it is - see compgraph/algorithms.py):

```python
graph = Graph.graph_from_iter('texts')
.map(operations.FilterPunctuation('text'))
.map(operations.LowerCase('text'))
.map(operations.Split('text'))
.sort(['text'])
.reduce(operations.Count('count'), ['text'])
.sort(['count', 'text'])
```

## Algorithms

```word_count``` - counting the number of occurrences in the table for each word.
```tf-idf``` - sort documents for each word by tf-idf metric.
```pmi``` - sort words for each document according to the pmi metric.
```yandex_maps``` - calculation of the speed of movement in the city from an hour and for a week.

## Running the tests

To run all tests, use:

```bash
pytest compgraph
```

```tests/correctness``` - author's tests of various parts of the problem. Designed to help you write algorithms
correctly.
```tests/memory``` - author's tests for memory used. They check for memory leaks, excessive materialization of
generators, etc.
```tests/test_graph.py``` - author's tests for graph methods.
```tests/test_examples.py``` - author's tests, which check graph multiple calls.
```tests/test_operations.py``` - author's tests, which check some mappers and reducers.

### Scripts

To run algorithms, you can use scripts, which you can find in folder: ```compgrapf.examples```

### Authors

- **Rustam Kadyrov** - [HedwigIndustries](https://github.com/HedwigIndustries)
