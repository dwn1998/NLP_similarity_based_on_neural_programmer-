"""
This question generator currently generates only the following questions for the specified scenarios

- Single column experiment:
    - sum
    - count
    - greater [number] sum


Each function will return a tuple in the format: (question_string, answer)
"""

import random


def generate_single_column_table_sum_all(table: [[int]]):
    return 'sum', generate_single_column_table_sum_all_answer(table)


def generate_single_column_table_count_all(table: [[int]]):
    return 'count', generate_single_column_table_count_all_answer(table)


def generate_single_column_table_sum_greater(table: [[int]], min_limit: int, max_limit: int):
    pivot = random.randint(min_limit, max_limit)
    return 'greater %d sum' % pivot, generate_single_column_table_sum_greater_answer(table, pivot)


def generate_single_column_table_count_greater(table: [[int]], min_limit: int, max_limit: int):
    pivot = random.randint(min_limit, max_limit)
    return 'greater %d count' % pivot,generate_single_column_table_count_greater_answer(table, pivot)


def generate_single_column_table_sum_all_answer(table: [[int]]):
    return sum(table[0]), True


def generate_single_column_table_count_all_answer(table: [[int]]):
    return len(table[0]), True


def generate_single_column_table_sum_greater_answer(table: [[int]], pivot: int):
    dif = sum(table[0])-pivot
    if dif>0:
        return 1.0,True
    else:
        return 0.0,True
    # return sum(list(map(lambda a: a if a > pivot else 0, table[0]))), True

def generate_single_column_table_count_greater_answer(table: [[int]], pivot: int):
    dif = len(table[0])-pivot
    if dif>0:
        return 1.0,True
    else:
        return 0.0,True
    # return len(list(map(lambda a: a if a < pivot else 0, table[0])))


QUESTION_GENERATION_FUNCTIONS = [
    generate_single_column_table_sum_all,
    generate_single_column_table_count_all,
    generate_single_column_table_sum_greater,
    generate_single_column_table_count_greater,
]
