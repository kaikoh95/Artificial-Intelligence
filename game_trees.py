"""Given a game tree, return the utility of the root of the tree when the root is a max node or min node correspondingly.
The functions use alpha-beta pruning. To make sure the pruning is correct, when pruning, the functions print what is being pruned.
If after seeing the i-th children, it is determined that the rest of the children can be ignored and proceed to print those remaining
children that are being pruned.
"""


def max_value(tree, alpha=float('-inf'), beta=float('inf')):

    bound = float('-inf')
    if type(tree) is int and abs(tree) != float('inf'):
        return tree

    for i in range(len(tree)):
        bound = max(bound, min_value(tree[i], alpha, beta))

        if bound >= beta:
            if tree[i + 1:]:
                print("Pruning:", ", ".join(map(str, tree[i + 1:])))
            return bound

        alpha = max(alpha, bound)

    return bound


def min_value(tree, alpha=float('-inf'), beta=float('inf')):

    bound = float('inf')
    if type(tree) is int and abs(tree) != float('inf'):
        return tree

    for i in range(len(tree)):
        bound = min(bound, max_value(tree[i], alpha, beta))

        if bound <= alpha:
            if tree[i + 1:]:
                print("Pruning:", ", ".join(map(str, tree[i + 1:])))
            return bound

        beta = min(beta, bound)

    return bound


def main():

    # Example 1
    tree = [3, [[1, 2], [[4, 5], [6, 7]]], 8]
    print("Game tree:", tree)
    print("Computing the utility of the root as a max node...")
    print("Root utility for maximiser:", max_value(tree))
    print("Computing the utility of the root as a min node....")
    print("Root utility for minimiser:", min_value(tree))

    # Example 2
    tree = [[[3, 12], 8], [2, [4, 6]], [14, 5, 2]]
    print("Game tree:", tree)
    print("Computing the utility of the root as a max node...")
    print("Root utility for maximiser:", max_value(tree))
    print("Computing the utility of the root as a min node....")
    print("Root utility for minimiser:", min_value(tree))

    # Example 3
    # no pruning when the root is max
    # but one child pruned when the root is min
    tree = [[1, 2], [3, 4]]
    print("Game tree:", tree)
    print("Computing the utility of the root as a max node...")
    print("Root utility for maximiser:", max_value(tree))
    print("Computing the utility of the root as a min node....")
    print("Root utility for minimiser:", min_value(tree))


main()
