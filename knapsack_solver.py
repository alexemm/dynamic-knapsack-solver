from pprint import pprint
from typing import List, Union, Tuple, Set, FrozenSet


def decide(weights: List[int], profits: List[float], capacity: int, minimum_profit: float) -> bool:
    """
    Deciding algorithm which returns if there exists a knapsack item allocation with a given minimum profit
    :param weights: Weights of the items
    :param profits: Profits of the items
    :param capacity: The capacity of the knapsack
    :param minimum_profit: Given minimum profit
    :return: True, if there exists a knapsack item allocation. False, if not
    """
    return solve(weights, profits, capacity)[len(weights) - 1][capacity] >= minimum_profit


def solve(weights: List[int], profits: List[float], capacity: int) -> List[List[float]]:
    """
    Solves given knapsack instance with dynamic programming and returns solution matrix.
    :param weights: Weights of items
    :param profits: Profits of items
    :param capacity: Capacity of knapsack instance
    :return: Solution matrix as list of lists of floats representing the maximum possible profit in the cell
    """
    assert len(weights) == len(profits)
    n: int = len(weights)  # items are numbered 0 to n-1
    f: List[List[float]] = [[0 if weights[0] > c else profits[0] for c in range(capacity + 1)]]
    for j in range(1, n):
        f.append([f[j - 1][c] if weights[j] > c else max(f[j - 1][c], f[j - 1][c - weights[j]] + profits[j]) for c in
                  range(capacity + 1)])
    return f


def get_solution_matrix_with_indices(solution: List[List[float]]) -> List[List[Union[float, int, str]]]:
    """
    Returns the matrix as a table with indices
    :param solution: The calculated table from the knapsack instance
    :return: The matrix as a table with indices
    """
    n: int = len(solution[0])
    ret: List[List[Union[float, int, str]]] = [["i/c"] + list(range(n))]
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
        if solution[item][c] != 0:
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
    """
    Returns the possible solution from a solved knapsack instance.
    :param weights: Weights of items
    :param profits: Profits of items
    :param solution: Solution matrix of the solved knapsack instance.
    :return: Set of item allocation of given knapsack instance
    """
    number_items: int = len(solution)
    n: int = len(solution[0])
    return get_packed_items_rek(solution, weights, profits, number_items - 1, n - 1, frozenset(), set())


def get_profit(solution: FrozenSet[int], profit: List[float]):
    """
    Calculates and returns total profit of given knapsack solution
    :param solution: Knapsack solution consisting of packed items
    :param profit: profit of items
    :return: Total profit of given knapsack solution
    """
    return sum(profit[item - 1] for item in solution)


def get_weights(solution: FrozenSet[int], weights: List[int]):
    """
    Calculates and returns total weight of given knapsack solution
    :param solution: Knapsack solution consisting of packed items
    :param weights: profit of items
    :return: Total weight of given knapsack solution
    """
    return sum(weights[item - 1] for item in solution)


def calculate_and_print_solutions(weights: List[int], profits: List[float], capacity: int) -> None:
    """
    Calculates the solution of the knapsack instance and prints out the table and packed items.
    :param weights: Weights of items
    :param profits: Profits of items
    :param capacity: Capacity of knapsack instance
    :return: None
    """
    solution = solve(weights, profits, capacity)
    print("Table from Calculation")
    pprint(get_solution_matrix_with_indices(solution))
    print("Solutions:")
    for packed_items in get_packed_items(weights, profits, solution):
        print(f"{packed_items}: profit={get_profit(packed_items, profits)}, weight={get_weights(packed_items, weights)}")


#w = [1, 4, 1, 3, 2, 5]
#p = [4, 1, 5, 2, 2, 7]
w = [3, 4, 1, 1, 2, 5]
p = [2, 1, 4, 5, 2, 7]
#w = [4,2,6,3,5,1]
#p = [5,4,10,2,9,3]
#w = [2,4,3]
#p = [10,17,14]
b = 10  # data from our example
calculate_and_print_solutions(w, p, b)
