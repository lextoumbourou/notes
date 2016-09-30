# Week 4: Trees! (including Binary Search Trees and Tries)

## Core Trees

* Trees can naturally represent data in the real world eg family tree, decision tree etc.
* Organisation of tree can define type of tree:
  * Root most important -> heap
  * Organised by char frequency -> Huffman Tree
  * Organised by node ordering -> search trees

## Core: Defining Trees

* Immutable properties of tree:
  * Must have only 1 parent node (node with no other parents).
  * Child are any nodes that aren't the parent.
  * Leaf nodes are any nodes that don't have children.
  * No cycles in a tree.
  * Each node can only have one (or less if parent) parents.

## Core: Binary Trees

* Generic Tree: a parent can have any number of children.
* Binary Tree: a parent can have at most 2 children.
  * Each nodes need: a value, a parent, a left child (could be null) and a right child (could be null).

* Example code:

  ```
  public class BinaryTree<E> {
    TreeNode<E> root;
  }

  private class TreeNode<E> {
    private E value;
    private TreeNode<E> parent;
    private TreeNode<E> left;
    private TreeNode<E> right;

    public TreeNode(E val, TreeNode<E> par) {
      this.value = val;
      this.parent = par;
      this.left = null;
      this.right = null;
    }

    public TreeNode<E> addLeftChild(E val) {
      this.left = new TreeNode<E>(val, this);
      return this.left;
    }
  }
  ```

## Core: Pre-Order Traversals

* Pre-order traversal (aka depth first search):
  1. Visit yourself
  2. Visit all of your left subtree.
  3. Visit all of your right subtree.

  ```
  private void preOrder(TreeNode<E> node) {
    if (node != null) {
      node.visit();
      preOrder(node.getLeftChild());
      preOrder(node.getRightChild());
    }
  }
  public void preOrder() {
    preOrder(root);
  }
  ```

## Core: Post-Order, In-Order and Level-Order Traversals

* Post-order traversal:
  1. Visit your left subtree.
  2. Visit your right subtree.
  3. Visit yourself.

* In-order traversal:
  1. Visit left subtree.
  2. Visit yourself.
  3. Visit your right subtree.

* Level-order traversal (breadth-first traversal):
  1. Keep a "to visit" list.
  2. When you visit a node, place its child on the list.
  3. Pull a node off the list, and place its child on the list and so on.

  ```
  public class BinaryTree<E> {
    TreeNode<E> root;

    public void levelOrder() {
      Queue< TreeNode<E> > q = new LinkedList< TreeNode<E> >();
      q.add(root);
      while(!q.isEmpty()) {
        TreeNode<E> curr = q.remove();
        if (curr != null) {
          curr.visit();
          q.add(curr.getLeftChild());
          q.add(curr.getRightChild());
        }
      }
    }
  }
  ```

## Core: Introduction to Binary Search Trees

* Binary search
  * Requires sorted array of items.
  * Start at middle, if middle is more than what you want, go for right-half, otherwise left.
  * Slow to insert into array.

* Binary search tree
  * Get ``O(log n)`` search
  * Get ``(O(1))`` insertion.
  * Max 2 children per node (same as binary tree).
  * Left subtrees must be less than parent.
  * Right subtree must be greater than parent.

* Searching in a BST: start at root. Are you at the node? If not, throw away one half of the tree and start again.

# Run Time Analysis of BSTs
