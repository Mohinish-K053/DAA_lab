# Algorithms Implementation – Reference Guide

This README contains **clean, separate code snippets** for each algorithm implemented in your project.
You can **copy & paste** any individual algorithm into your VS Code files.

---

# 📌 1. Linear Search

```python
def linear_search(arr, target):
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1
```

---

# 📌 2. Binary Search (Iterative)

```python
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
```

---

# 📌 3. Merge Sort

```python
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
```

---

# 📌 4. Quick Sort (Divide & Conquer)

```python
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
```

---

# 📌 5. Strassen's Matrix Multiplication (Divide & Conquer)

```python
# Helper functions omitted for brevity
# Full version exists in the main python file

def strassen_matrix_multiply(A, B):
    # Validates sizes, pads matrices to power of 2
    # Performs recursive Strassen multiplication
    # Returns the result trimmed to original size
    pass
```

(Use the full version from your main code.)

---

# 📌 6. Fractional Knapsack (Greedy Algorithm)

```python
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
```

---

# 📌 7. Graph Coloring (Greedy – Welsh Powell)

```python
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
```

---

# 📌 8. 8‑Queens Problem (Backtracking)

```python
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
```

---

# 📌 9. Traveling Salesperson Problem — Held‑Karp DP

```python
# Simplified summary function
# Full version in main code due to complexity

def held_karp_tsp(dist):
    # Returns (min_cost, optimal_path)
    # Uses DP: O(n^2 * 2^n)
    pass
```

(Use full version from main file.)

---

# 📌 10. Valid Parentheses

```python
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
```

---

# ✅ Notes

* All code is Python 3 compatible.
* You may copy each snippet directly into VS Code.
* Full complex algorithms (Strassen + TSP) should be copied from the master Python file.

---

If you want a **PDF, DOCX, or printable version** of this README, tell me and I’ll generate it!
