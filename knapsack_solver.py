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
    """
    Recursive function for calculating the packed items
    :param solution: The matrix of the solved knapsack instance
    :param weights: The list of the weight of the items
    :param profits: The list of the profit of the items
    :param item: The index of the item
    :param c: The current capacity of the baggage
    :param items: The set of the currently packed items
    :param solution_set: The set of all the solutions to the knapsack instance
    :return: The solution set with all the item combinations used
    """
    if item == 0 or c == 0:
        if c != 0 and solution[item][c] != 0:
            # When we need to pack in the first item
            items |= frozenset({item + 1})
        return solution_set | {items}
    if solution[item][c] - profits[item] == solution[item - 1][c - weights[item]]:
        # When we pack the item
        solution_set |= get_packed_items_rek(solution, weights, profits, item - 1, c - weights[item],
                                             items | frozenset({item + 1}), solution_set)
    if solution[item][c] == solution[item - 1][c]:
        # When we do not pack the item, because the maximal profit would stay the same
        solution_set |= get_packed_items_rek(solution, weights, profits, item - 1, c, items, solution_set)
    return solution_set


def get_packed_items(weights: List[int], profits: List[float], solution: List[List[float]]) -> Set[FrozenSet[int]]:
    number_items: int = len(solution)
    n: int = len(solution[0])
    return get_packed_items_rek(solution, weights, profits, number_items - 1, n - 1, frozenset(), set())


def get_profit(solution: FrozenSet[int], profit: List[float]):
    return sum(profit[item - 1] for item in solution)


def get_weights(solution: FrozenSet[int], weights: List[int]):
    return sum(weights[item - 1] for item in solution)


w = [1, 4, 1, 3, 2, 5]
p = [4, 1, 5, 2, 2, 7]
w = [4,2,6,3,5,1]
p = [5,4,10,2,9,3]
b = 10  # data from our example
solution = solve(w, p, b)
print("Table from Calculation")
pprint(get_solution_matrix_with_indices(solution))
print("Solutions:")
for packed_items in get_packed_items(w, p, solution):
    print(f"{packed_items}: profit={get_profit(packed_items, p)}, weight={get_weights(packed_items, w)}")

