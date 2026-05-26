"""
Execution engine for the computational graph.
"""

class ExecutionEngine:
    """
    Recursive executor
    with memoization cache.
    """

    def __init__(self):

        self.cache = {}

    def run(self, node):

        node_id = id(node)

        if node_id in self.cache:

            print(
                f"Cache Hit: "
                f"{type(node).__name__}"
            )

            return self.cache[node_id]

        inputs = [
            self.run(parent)
            for parent in node.parents
        ]

        result = node.compute(*inputs)

        self.cache[node_id] = result

        return result

    def clear_cache(self):

        self.cache.clear()

