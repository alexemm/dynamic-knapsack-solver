from pprint import pprint
from typing import List, Union, Tuple, Set, FrozenSet


def decide(weights: List[int], profits: List[float], capacity: int, minimum_profit: float) -> bool:
    return solve(weights, profits, capacity)[len(weights) - 1][capacity] >= minimum_profit


def solve(weights: List[int], profits: List[float], capacity: int) -> List[List[float]]:
    assert len(weights) == len(profits)
    n: int = len(weights)  # items are numbered 0 to n-1
    f: List[List[float]] = [[0 if weights[0] > c else profits[0] for c in range(capacity + 1)]]
    for j in range(1, n):
        f.append([f[j - 1][c] if weights[j] > c else max(f[j - 1][c], f[j - 1][c - weights[j]] + profits[j]) for c in
                  range(capacity + 1)])
    return f


def get_solution_matrix_with_indices(solution: List[List[float]]) -> List[Tuple[str, List[Union[float, int]]]]:
    n: int = len(solution[0])
    ret: List[Tuple[Union[str, int], List[Union[float, int]]]] = [["i/c"] + list(range(n))]
    for i, item in enumerate(solution):
        ret.append([str(i + 1)] + item)
    return ret


def get_packed_items_rek(solution: List[List[float]], weights: List[int], profits: List[float], item: int, c: int,
                         items: FrozenSet[int], solution_set: Set[FrozenSet[int]]) -> Set[FrozenSet[int]]:
    if item == 0 or c == 0:
        if c != 0:
            items |= frozenset({item + 1})
        return solution_set | {items}
    if solution[item][c] - profits[item] == solution[item - 1][c - weights[item]]:
        solution_set |= get_packed_items_rek(solution, weights, profits, item - 1, c - weights[item],
                                             items | frozenset({item + 1}), solution_set)

    if solution[item][c] == solution[item - 1][c]:
        solution_set |= get_packed_items_rek(solution, weights, profits, item - 1, c, items, solution_set)
    return solution_set


def get_packed_items(weights: List[int], profits: List[float], solution: List[List[float]]) -> Set[FrozenSet[int]]:
    number_items: int = len(solution)
    n: int = len(solution[0])
    return get_packed_items_rek(solution, weights, profits, number_items - 1, n - 1, frozenset(), set())


# w = (2, 4, 3)
# p = (10, 17, 14)
w = [1, 4, 1, 3, 2, 5]
p = [4, 1, 5, 2, 2, 7]
b = 10  # data from our example
solution = solve(w, p, b)
print("Table from Calculation")
pprint(get_solution_matrix_with_indices(solution))
print("Solutions:")
print(get_packed_items(w, p, solution))