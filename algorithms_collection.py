"""
algorithms_collection.py

A single-file collection of algorithm implementations for running in VS Code.

How to use:
1. Save this file as algorithms_collection.py in VS Code.
2. Open a terminal in VS Code and run: python algorithms_collection.py
3. Follow the menu to run example cases for each algorithm. You can also import functions from this file into other scripts.

Implemented algorithms:
- linear_search
- binary_search (iterative)
- merge_sort
- quick_sort
- strassen_matrix_multiply (pads matrices to power-of-two)
- fractional_knapsack (greedy)
- graph_coloring_greedy (Welsh-Powell style)
- eight_queens_backtracking
- held_karp_tsp (dynamic programming solution for small n)
- is_valid_parentheses

Notes:
- Strassen's algorithm uses recursion and expects square matrices; the implementation pads to the next power of two.
- Held-Karp TSP is exponential (O(n^2 * 2^n)) and suitable for n up to ~15.
- Fractional knapsack is the greedy variant (optimal for fractional knapsack but not 0/1 knapsack).

"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math
import itertools

# ------------------ Linear Search ------------------

def linear_search(arr: List[int], target: int) -> int:
    """Return index of target in arr or -1 if not found."""
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1

# ------------------ Binary Search ------------------

def binary_search(arr: List[int], target: int) -> int:
    """Iterative binary search on sorted arr. Returns index or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# ------------------ Merge Sort ------------------

def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    res = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    res.extend(left[i:])
    res.extend(right[j:])
    return res

# ------------------ Quick Sort ------------------

def quick_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    _quick_sort_inplace(a, 0, len(a) - 1)
    return a


def _quick_sort_inplace(a: List[int], lo: int, hi: int) -> None:
    if lo < hi:
        p = _partition(a, lo, hi)
        _quick_sort_inplace(a, lo, p - 1)
        _quick_sort_inplace(a, p + 1, hi)


def _partition(a: List[int], lo: int, hi: int) -> int:
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i

# ------------------ Strassen's Matrix Multiplication ------------------

import copy

Matrix = List[List[float]]

def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _pad_matrix(A: Matrix, size: int) -> Matrix:
    n = len(A)
    m = len(A[0]) if A else 0
    B = [[0.0] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            B[i][j] = float(A[i][j])
    return B


def _add(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def _sub(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def _split(A: Matrix) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    n = len(A)
    mid = n // 2
    A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
    A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
    A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
    A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
    return A11, A12, A21, A22


def _join(A11: Matrix, A12: Matrix, A21: Matrix, A22: Matrix) -> Matrix:
    top = [a + b for a, b in zip(A11, A12)]
    bottom = [a + b for a, b in zip(A21, A22)]
    return top + bottom


def strassen_matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    # Validate shapes
    n = len(A)
    p = len(A[0]) if A else 0
    p2 = len(B)
    m = len(B[0]) if B else 0
    if p != p2:
        raise ValueError("Incompatible matrix sizes for multiplication")
    size = max(n, p, m)
    s = _next_power_of_two(size)
    A_pad = _pad_matrix(A, s)
    B_pad = _pad_matrix(B, s)
    C_pad = _strassen_recursive(A_pad, B_pad)
    # Unpad to original size n x m
    C = [[C_pad[i][j] for j in range(m)] for i in range(n)]
    return C


def _strassen_recursive(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    M1 = _strassen_recursive(_add(A11, A22), _add(B11, B22))
    M2 = _strassen_recursive(_add(A21, A22), B11)
    M3 = _strassen_recursive(A11, _sub(B12, B22))
    M4 = _strassen_recursive(A22, _sub(B21, B11))
    M5 = _strassen_recursive(_add(A11, A12), B22)
    M6 = _strassen_recursive(_sub(A21, A11), _add(B11, B12))
    M7 = _strassen_recursive(_sub(A12, A22), _add(B21, B22))
    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)
    return _join(C11, C12, C21, C22)

# ------------------ Fractional Knapsack (Greedy) ------------------

def fractional_knapsack(values: List[float], weights: List[float], capacity: float) -> Tuple[float, List[Tuple[int, float]]]:
    """Returns (max_value, list of (index, fraction_taken))."""
    items = []
    for i, (v, w) in enumerate(zip(values, weights)):
        if w <= 0:
            raise ValueError("Weights must be positive")
        items.append((i, v / w, v, w))
    items.sort(key=lambda x: x[1], reverse=True)
    total_value = 0.0
    taken = []
    for i, ratio, v, w in items:
        if capacity <= 0:
            break
        amt = min(w, capacity)
        frac = amt / w
        total_value += v * frac
        taken.append((i, frac))
        capacity -= amt
    return total_value, taken

# ------------------ Graph Coloring (Greedy - Welsh-Powell style) ------------------

def graph_coloring_greedy(adj_list: Dict[int, List[int]]) -> Dict[int, int]:
    """Returns a color assignment dict node -> color (colors are integers starting at 0)."""
    # Sort vertices by decreasing degree
    nodes_sorted = sorted(adj_list.keys(), key=lambda x: -len(adj_list[x]))
    color_of: Dict[int, Optional[int]] = {v: None for v in adj_list}
    for v in nodes_sorted:
        used = set(color_of[u] for u in adj_list[v] if color_of[u] is not None)
        color = 0
        while color in used:
            color += 1
        color_of[v] = color
    return color_of

# ------------------ 8-Queens (Backtracking) ------------------

def eight_queens_backtracking() -> List[List[int]]:
    """Return list of solutions, each solution is a list of column indices (0..7) for each row."""
    N = 8
    solutions: List[List[int]] = []
    cols = set()
    diag1 = set()
    diag2 = set()
    board = [-1] * N

    def place(row: int):
        if row == N:
            solutions.append(board[:])
            return
        for c in range(N):
            if c in cols or (row - c) in diag1 or (row + c) in diag2:
                continue
            cols.add(c); diag1.add(row - c); diag2.add(row + c)
            board[row] = c
            place(row + 1)
            cols.remove(c); diag1.remove(row - c); diag2.remove(row + c)
            board[row] = -1

    place(0)
    return solutions

# ------------------ TSP - Held-Karp (Dynamic Programming) ------------------

def held_karp_tsp(dist: List[List[float]]) -> Tuple[float, List[int]]:
    """Return (min_cost, path) for metric complete graph using Held-Karp DP.
    dist is an n x n matrix. Node 0 is considered the start and end."""
    n = len(dist)
    if n == 0:
        return 0.0, []
    # DP[mask][j] = min cost to reach set mask ending at j
    # mask includes starting node 0
    Nmask = 1 << n
    dp = [[math.inf] * n for _ in range(Nmask)]
    parent = [[-1] * n for _ in range(Nmask)]
    dp[1][0] = 0.0
    for mask in range(1, Nmask):
        for j in range(n):
            if not (mask & (1 << j)): continue
            if j == 0 and mask != 1: continue  # don't revisit 0 in middle
            pmask = mask ^ (1 << j)
            if pmask == 0 and j != 0: continue
            if pmask == 0 and j == 0: continue
            if pmask == 1 and j != 0:
                # Coming from 0 directly
                cost = dp[1][0] + dist[0][j]
                if cost < dp[mask][j]:
                    dp[mask][j] = cost
                    parent[mask][j] = 0
                continue
            # general
            for k in range(n):
                if not (pmask & (1 << k)): continue
                if dp[pmask][k] + dist[k][j] < dp[mask][j]:
                    dp[mask][j] = dp[pmask][k] + dist[k][j]
                    parent[mask][j] = k
    # close tour
    full = (1 << n) - 1
    best_cost = math.inf
    last = -1
    for j in range(1, n):
        cost = dp[full][j] + dist[j][0]
        if cost < best_cost:
            best_cost = cost
            last = j
    if last == -1:
        return math.inf, []
    # Reconstruct path
    path = [0]
    mask = full
    cur = last
    stack = []
    while cur != 0:
        stack.append(cur)
        prev = parent[mask][cur]
        mask = mask ^ (1 << cur)
        cur = prev
    stack.reverse()
    path.extend(stack)
    path.append(0)
    return best_cost, path

# ------------------ Valid Parentheses ------------------

def is_valid_parentheses(s: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: List[str] = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            continue
    return len(stack) == 0

# ------------------ Helper / Demo ------------------

# ------------------ Interactive Menu ------------------

def run_menu():
    while True:
        print("===== Algorithm Menu =====")
        print("1. Linear Search")
        print("2. Binary Search")
        print("3. Merge Sort")
        print("4. Quick Sort")
        print("5. Strassen Matrix Multiplication")
        print("6. Fractional Knapsack")
        print("7. Graph Coloring (Greedy)")
        print("8. 8-Queens Problem")
        print("9. Traveling Salesperson Problem (TSP) - DP Held-Karp")
        print("10. Valid Parentheses Check")
        print("0. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            arr = list(map(int, input("Enter array elements: ").split()))
            target = int(input("Enter target to search: "))
            print("Index:", linear_search(arr, target))

        elif choice == "2":
            arr = list(map(int, input("Enter sorted array elements: ").split()))
            target = int(input("Enter target to search: "))
            print("Index:", binary_search(arr, target))

        elif choice == "3":
            arr = list(map(int, input("Enter array elements: ").split()))
            print("Sorted:", merge_sort(arr))

        elif choice == "4":
            arr = list(map(int, input("Enter array elements: ").split()))
            print("Sorted:", quick_sort(arr))

        elif choice == "5":
            n = int(input("Enter size n of square matrices: "))
            print("Enter matrix A:")
            A = [list(map(float, input().split())) for _ in range(n)]
            print("Enter matrix B:")
            B = [list(map(float, input().split())) for _ in range(n)]
            print("Result:")
            print(strassen_matrix_multiply(A, B))

        elif choice == "6":
            values = list(map(float, input("Enter values: ").split()))
            weights = list(map(float, input("Enter weights: ").split()))
            capacity = float(input("Enter capacity: "))
            print(fractional_knapsack(values, weights, capacity))

        elif choice == "7":
            n = int(input("Enter number of nodes: "))
            adj = {}
            for i in range(n):
                adj[i] = list(map(int, input(f"Neighbors of {i}: ").split()))
            print(graph_coloring_greedy(adj))

        elif choice == "8":
            sols = eight_queens_backtracking()
            print(f"Total solutions: {len(sols)}")
            print("First solution:", sols[0])

        elif choice == "9":
            n = int(input("Enter number of nodes: "))
            print("Enter distance matrix:")
            dist = [list(map(float, input().split())) for _ in range(n)]
            print(held_karp_tsp(dist))

        elif choice == "10":
            s = input("Enter parentheses string: ")
            print(is_valid_parentheses(s))

        elif choice == "0":
            print("Exiting...")
            break

        else:
            print("Invalid choice! Try again.")


if __name__ == "__main__":
    run_menu()
