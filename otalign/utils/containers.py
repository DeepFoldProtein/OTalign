import heapq
from typing import Callable, Generic, List, Optional, Tuple, TypeVar


T = TypeVar("T")


class TopKHeap(Generic[T]):
    """A fixed-size heap that maintains the top-K largest items seen in a stream.

    This implementation uses a min-heap of size at most `k`. Inserting a new
    item runs in O(log k) time.

    Attributes:
        _k (int): Maximum number of items to retain.
        _key (Callable[[T], float]): Function to extract comparison key.
        _heap (List[Tuple[float, T]]): Internal heap storing (key, item) tuples.
    """

    def __init__(self, k: int, key: Optional[Callable[[T], float]] = None):
        """Initializes the TopKHeap.

        Args:
            k: Number of top elements to keep. Must be > 0.
            key: Optional function to extract a numeric key from items.
                If None, the items themselves are compared.

        Raises:
            ValueError: If `k` is not positive.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        self._k = k
        self._key: Callable[[T], float] = key or (lambda x: x)  # type: ignore
        self._heap: List[Tuple[float, T]] = []

    def push(self, item: T) -> None:
        """Inserts an item from the stream, evicting the smallest if necessary.

        If the heap has fewer than `k` items, the new item is always added.
        Otherwise, the item is only added if its key is larger than the current
        smallest key in the heap (the root), in which case the root is replaced.

        Args:
            item: The item to insert.
        """
        item_key = self._key(item)
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, (item_key, item))
        else:
            # Only replace if the new item outranks the current smallest
            if item_key > self._heap[0][0]:
                heapq.heapreplace(self._heap, (item_key, item))

    def topk(self) -> List[T]:
        """Retrieves the current top-K items, sorted largest first.

        Returns:
            A list of the top-K items in descending order of key.
        """
        # Sort by key descending and extract the items
        return [item for _, item in sorted(self._heap, reverse=True)]

    def kth(self) -> Optional[T]:
        """Peeks at the K-th largest item seen so far.

        Returns:
            The K-th largest item (i.e., the smallest in the heap), or
            `None` if fewer than `k` items have been pushed.
        """
        if len(self._heap) < self._k:
            return None
        return self._heap[0][1]

    def __len__(self) -> int:
        """Returns the number of items currently stored (≤ k)."""
        return len(self._heap)


if __name__ == "__main__":
    # Example usage
    data_stream = [5, 1, 3, 10, 2, 8, 6]
    top3 = TopKHeap(k=3)

    for x in data_stream:
        top3.push(x)

    print(top3.topk())  # Output: [10, 8, 6]
    print(top3.kth())  # Output: 6

    # With a key function (e.g., keep top-2 longest strings):
    words = ["apple", "fig", "banana", "cherry", "date"]
    top2 = TopKHeap[str](2, key=len)
    for w in words:
        top2.push(w)
    print(top2.topk())  # Output: ['banana', 'cherry']
