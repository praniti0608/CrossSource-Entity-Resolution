import heapq
from array import array
from typing import List, Tuple


_UINT32_ARRAY_PROBE = array("I")


class BoundedHitCounter:
    __slots__ = ("counts", "touched", "size")

    def __init__(self, size: int):
        if _UINT32_ARRAY_PROBE.itemsize != 4:
            raise RuntimeError("array('I') itemsize is not 4 bytes.")

        self.size = size
        self.counts = array("I", [0]) * size
        self.touched: List[int] = []

    def add(self, row_id: int, increment: int = 1) -> None:
        if self.counts[row_id] == 0:
            self.touched.append(row_id)
        self.counts[row_id] += increment

    def top_k(self, k: int) -> List[Tuple[int, int]]:
        if k <= 0 or not self.touched:
            return []

        heap: List[Tuple[int, int, int]] = []

        for row_id in self.touched:
            count = self.counts[row_id]
            item = (count, -row_id, row_id)

            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)

        heap.sort(key=lambda item: (item[0], item[1]), reverse=True)

        return [
            (row_id, count)
            for count, _, row_id in heap
        ]

    def reset(self) -> None:
        for row_id in self.touched:
            self.counts[row_id] = 0
        self.touched.clear()


# Test 1 — tie-breaking
h = BoundedHitCounter(100)

for r in [3, 1, 7, 3, 1, 7, 9, 9, 5]:
    h.add(r)

assert h.top_k(3) == [
    (1, 2),
    (3, 2),
    (7, 2),
]

print("Test 1 PASSED")


# Test 2 — 60,000 distinct touched rows
h2 = BoundedHitCounter(60000)

for r in range(60000):
    h2.add(r)

assert h2.top_k(5) == [
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
]

print("Test 2 PASSED")


# Test 3 — reset
h2.reset()

assert len(h2.touched) == 0
assert h2.top_k(5) == []

print("Test 3 PASSED")

print("ALL HIT COUNTER TESTS PASSED")