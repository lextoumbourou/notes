from collections import namedtuple


def knapsack(capacity, items):
    """Return maximum value of the items with specified capacity."""
    if not items or not capacity:
        return 0

    item = items.pop()

    if (item.weight > capacity):
        return knapsack(capacity, items)

    capacity_with_item = capacity - item.weight

    with_item = item.value + knapsack(capacity_with_item, items)
    without_item = knapsack(capacity, items)

    return max(with_item, without_item)


def test_knapsack():
    Item = namedtuple('Item', ['weight', 'value'])

    items = [
        Item(weight=10, value=110),
        Item(weight=20, value=100),
        Item(weight=30, value=120)]
    capacity = 50
    assert knapsack(capacity, items) == 230


if __name__ == '__main__':
    test_knapsack()
