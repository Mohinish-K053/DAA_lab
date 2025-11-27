# Algorithms Implementation – Reference Guide

This README contains **clean, separate code snippets** for each algorithm implemented in your project.
You can **copy & paste** any individual algorithm into your VS Code files.

---

# 📌 1. Linear Search

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
def linear_search(arr, target):
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1
arr = list(map(int, input("Enter array elements: ").split()))
target = int(input("Enter target to search: "))
print("Index:", linear_search(arr, target))
```

---

# 📌 2. Binary Search (Iterative)

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
def binary_search(arr, target):
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
arr = list(map(int, input("Enter sorted array elements: ").split()))
target = int(input("Enter target to search: "))
print("Index:", binary_search(arr, target))
```

---

# 📌 3. Merge Sort

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
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
arr = list(map(int, input("Enter array elements: ").split()))
print("Sorted:", merge_sort(arr))
```

---

# 📌 4. Quick Sort (Divide & Conquer)

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
def quick_sort(arr):
    a = arr[:]
    quick_sort_inplace(a, 0, len(a) - 1)
    return a

def quick_sort_inplace(a, lo, hi):
    if lo < hi:
        p = partition(a, lo, hi)
        quick_sort_inplace(a, lo, p - 1)
        quick_sort_inplace(a, p + 1, hi)

def partition(a, lo, hi):
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i
arr = list(map(int, input("Enter array elements: ").split()))
print("Sorted:", quick_sort(arr))
```

---

# 📌 5. Strassen's Matrix Multiplication (Divide & Conquer)

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
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
n = int(input("Enter size n of square matrices: "))
print("Enter matrix A:")
A = [list(map(float, input().split())) for _ in range(n)]
print("Enter matrix B:")
B = [list(map(float, input().split())) for _ in range(n)]
print("Result:")
print(strassen_matrix_multiply(A, B))
```

---

# 📌 6. Fractional Knapsack (Greedy Algorithm)

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
def fractional_knapsack(values, weights, capacity):
    items = []
    for i, (v, w) in enumerate(zip(values, weights)):
        items.append((i, v / w, v, w))
    items.sort(key=lambda x: x[1], reverse=True)

    total_value = 0
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
values = list(map(float, input("Enter values: ").split()))
weights = list(map(float, input("Enter weights: ").split()))
capacity = float(input("Enter capacity: "))
print(fractional_knapsack(values, weights, capacity))
```

---

# 📌 7. Graph Coloring (Greedy – Welsh Powell)

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
def graph_coloring_greedy(adj_list):
    nodes_sorted = sorted(adj_list, key=lambda x: -len(adj_list[x]))
    color_of = {v: None for v in adj_list}

    for v in nodes_sorted:
        used = set(color_of[u] for u in adj_list[v] if color_of[u] is not None)
        color = 0
        while color in used:
            color += 1
        color_of[v] = color

    return color_of
n = int(input("Enter number of nodes: "))
adj = {}
for i in range(n):
    adj[i] = list(map(int, input(f"Neighbors of {i}: ").split()))
print(graph_coloring_greedy(adj))
```

---

# 📌 8. 8‑Queens Problem (Backtracking)

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
def eight_queens_backtracking():
    N = 8
    solutions = []
    cols = set()
    diag1 = set()
    diag2 = set()
    board = [-1] * N

    def place(row):
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

    place(0)
    return solutions
sols = eight_queens_backtracking()
print(f"Total solutions: {len(sols)}")
print("First solution:", sols[0])
```

---

# 📌 9. Traveling Salesperson Problem — Held‑Karp DP

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
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
n = int(input("Enter number of nodes: "))
print("Enter distance matrix:")
dist = [list(map(float, input().split())) for _ in range(n)]
print(held_karp_tsp(dist))
```

---

# 📌 10. Valid Parentheses

```python
from typing import List, Tuple, Optional, Dict
import math
import itertools
def is_valid_parentheses(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0
s = input("Enter parentheses string: ")
print(is_valid_parentheses(s))
```

---

# ✅ Notes

* All code is Python 3 compatible.
* You may copy each snippet directly into VS Code.
* Full complex algorithms (Strassen + TSP) should be copied from the master Python file.

---

If you want a **PDF, DOCX, or printable version** of this README, tell me and I’ll generate it!
