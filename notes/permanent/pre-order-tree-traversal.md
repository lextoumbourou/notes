---
title: Pre-order Tree Traversal
date: 2025-02-16 00:00
modified: 2026-09-26 08:55
status: draft
---

**Pre-order Tree Traversal** is a depth-first way of visiting every node in a tree, where each node is visited before its children: visit the root, then traverse the left subtree, then the right subtree.

For a [Binary Tree](binary-tree.md):

```python
def pre_order(node):
    if node is None:
        return
    visit(node)
    pre_order(node.left)
    pre_order(node.right)
```

Since the root always comes first, pre-order traversal is useful for copying or serialising a tree. Compare with in-order (left, root, right) and post-order (left, right, root) traversal.
