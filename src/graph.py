"""
Graph data structures and algorithms for random graph generation and visualization.

This module provides:
- Graph: A simple undirected graph data structure
- erdos_renyi: Generate Erdős-Rényi random graphs
- visualize: Visualize graphs with circular layout
"""

import random
import matplotlib.pyplot as plt
import math


# ============================================================================
# GRAPH DATA STRUCTURE
# ============================================================================

class Graph:
    """
    A simple graph data structure for undirected graphs.

    Attributes:
        n: Number of vertices
        edges: Set of edges as tuples (i, j) where i < j
        adj: Adjacency list representation (dict: vertex -> list of neighbors)
    """

    def __init__(self, n):
        """
        Initialize an empty graph with n vertices.

        Args:
            n: Number of vertices (labeled 0 to n-1)
        """
        self.n = n
        self.edges = set()
        self.adj = {i: [] for i in range(n)}

    def add_edge(self, i, j):
        """
        Add an undirected edge between vertices i and j.

        Args:
            i, j: Vertex indices
        """
        if i == j:  # No self-loops
            return

        # Ensure i < j for canonical edge representation
        if i > j:
            i, j = j, i

        # Add edge if it doesn't exist
        if (i, j) not in self.edges:
            self.edges.add((i, j))
            self.adj[i].append(j)
            self.adj[j].append(i)

    def has_edge(self, i, j):
        """
        Check if edge (i,j) exists.

        Args:
            i, j: Vertex indices

        Returns:
            bool: True if edge exists, False otherwise
        """
        if i > j:
            i, j = j, i
        return (i, j) in self.edges

    def degree(self, v):
        """
        Return the degree of vertex v.

        Args:
            v: Vertex index

        Returns:
            int: Number of edges incident to vertex v
        """
        return len(self.adj[v])

    def num_edges(self):
        """
        Return the number of edges.

        Returns:
            int: Total number of edges in the graph
        """
        return len(self.edges)

    def is_connected(self):
        """
        Check if the graph is connected.

        A graph is connected if there is a path between every pair of vertices.
        Uses a simple flood-fill approach starting from vertex 0.

        Returns:
            bool: True if graph is connected, False otherwise

        Note:
            - Empty graph (n=0) is considered connected
            - Single vertex graph (n=1) is considered connected
            - Runtime: O(n^2) in worst case (brute force approach)
        """
        # Edge cases
        if self.n == 0 or self.n == 1:
            return True

        # Start flood-fill from vertex 0
        visited = {0}
        changed = True

        # Keep adding neighbors of visited vertices until no new vertices are added
        while changed:
            changed = False
            new_visited = set(visited)

            for v in visited:
                for neighbor in self.adj[v]:
                    if neighbor not in new_visited:
                        new_visited.add(neighbor)
                        changed = True

            visited = new_visited

        # Graph is connected if all vertices were visited
        return len(visited) == self.n

    def adjacency_matrix(self):
        """
        Compute the adjacency matrix of the graph.

        Returns:
            numpy.ndarray: n x n adjacency matrix where A[i,j] = 1 if (i,j) is an edge
        """
        import numpy as np
        A = np.zeros((self.n, self.n))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        return A

    def laplacian_matrix(self):
        """
        Compute the Laplacian matrix of the graph.

        The Laplacian is defined as L = D - A, where D is the degree matrix
        and A is the adjacency matrix.

        Returns:
            numpy.ndarray: n x n Laplacian matrix
        """
        import numpy as np
        A = self.adjacency_matrix()
        D = np.diag([self.degree(i) for i in range(self.n)])
        L = D - A
        return L

    def fiedler_value(self):
        """
        Compute the Fiedler value (algebraic connectivity) of the graph.

        The Fiedler value is the second smallest eigenvalue of the Laplacian matrix.
        For connected graphs, lambda_2 > 0. For disconnected graphs, lambda_2 = 0.

        Returns:
            float: The Fiedler value (lambda_2)

        Note:
            - Also known as algebraic connectivity
            - Measures how well-connected the graph is
            - Returns 0 for disconnected graphs
        """
        import numpy as np
        if self.n <= 1:
            return 0.0
        L = self.laplacian_matrix()
        eigenvalues = np.linalg.eigvalsh(L)  # Returns sorted eigenvalues
        return float(eigenvalues[1])  # Second smallest eigenvalue

    def euler_characteristic(self):
        """
        Compute the Euler characteristic of the graph.

        The Euler characteristic is defined as χ = V - E, where V is the number
        of vertices and E is the number of edges.

        Returns:
            int: The Euler characteristic (χ = n - m)

        Note:
            - For a tree: χ = 1
            - For a graph with k connected components and no cycles: χ = k
            - Becomes more negative as more edges are added
        """
        return self.n - self.num_edges()

    def __repr__(self):
        return f"Graph(n={self.n}, edges={self.num_edges()})"


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def erdos_renyi(n, p):
    """
    Generate an Erdős-Rényi random graph G(n, p).

    For each possible pair of vertices, add an edge with probability p.

    Args:
        n: Number of vertices
        p: Probability of edge between any two vertices (0 <= p <= 1)

    Returns:
        Graph: A random graph with n vertices

    Time Complexity:
        O(n^2) - must consider all possible pairs of vertices
    """
    G = Graph(n)

    # Iterate over all possible pairs of vertices
    for i in range(n):
        for j in range(i + 1, n):
            # Add edge with probability p
            if random.random() < p:
                G.add_edge(i, j)

    return G


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize(G, title="Random Graph", figsize=(10, 10), show_labels=False):
    """
    Visualize a graph using matplotlib with a circular layout.

    Args:
        G: Graph object to visualize
        title: Title for the plot
        figsize: Figure size as (width, height)
        show_labels: Whether to show vertex labels

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use circular layout (deterministic)
    pos = _circular_layout(G.n)

    # Draw edges
    for i, j in G.edges:
        x_coords = [pos[i][0], pos[j][0]]
        y_coords = [pos[i][1], pos[j][1]]
        ax.plot(x_coords, y_coords, 'gray', linewidth=1, alpha=0.5, zorder=1)

    # Draw vertices
    x_coords = [pos[i][0] for i in range(G.n)]
    y_coords = [pos[i][1] for i in range(G.n)]

    # Adjust vertex size based on graph size
    vertex_size = max(50, min(500, 5000 / G.n))

    ax.scatter(x_coords, y_coords, c='lightblue', s=vertex_size,
               edgecolors='darkblue', linewidths=2, zorder=2)

    # Add vertex labels
    if show_labels:
        font_size = max(6, min(12, 150 / math.sqrt(G.n)))
        for i in range(G.n):
            ax.text(pos[i][0], pos[i][1], str(i),
                    ha='center', va='center', fontsize=font_size,
                    fontweight='bold', zorder=3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

    # Set reasonable axis limits for circular layout
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    return fig, ax


def _circular_layout(n):
    """
    Position n vertices in a circular layout (deterministic).

    Arranges vertices evenly distributed around a circle.
    Vertex 0 is placed at the top (angle 0), and subsequent vertices
    are placed clockwise around the circle.

    Args:
        n: Number of vertices

    Returns:
        dict: Mapping from vertex index to (x, y) coordinates
    """
    if n == 0:
        return {}

    if n == 1:
        return {0: (0, 0)}

    pos = {}
    for i in range(n):
        # Angle for vertex i (starting from top, going clockwise)
        angle = -2 * math.pi * i / n + math.pi / 2
        pos[i] = (math.cos(angle), math.sin(angle))

    return pos


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    import os

    # Create assets directory if it doesn't exist
    os.makedirs("assets", exist_ok=True)

    # Set seed for reproducibility
    random.seed(42)

    print("=" * 60)
    print("Testing Erdős-Rényi Random Graph Generator")
    print("=" * 60)

    # Test 1: Small graph with high probability (likely connected)
    print("\n[Test 1] Generating G(10, 0.4)...")
    G1 = erdos_renyi(n=10, p=0.4)
    print(G1)
    print(f"Edges: {len(G1.edges)}")
    print(f"Average degree: {2 * len(G1.edges) / G1.n:.2f}")
    print(f"Connected: {G1.is_connected()}")

    visualize(G1, title=f"G(10, 0.4) - Connected: {G1.is_connected()}")
    plt.savefig("assets/graph_10.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: assets/graph_10.png")
    plt.close()

    # Test 2: Medium graph
    print("\n[Test 2] Generating G(20, 0.2)...")
    G2 = erdos_renyi(n=20, p=0.2)
    print(G2)
    print(f"Edges: {len(G2.edges)}")
    print(f"Average degree: {2 * len(G2.edges) / G2.n:.2f}")
    print(f"Connected: {G2.is_connected()}")

    visualize(G2, title=f"G(20, 0.2) - Connected: {G2.is_connected()}")
    plt.savefig("assets/graph_20.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: assets/graph_20.png")
    plt.close()

    # Test 3: Larger graph with lower probability
    print("\n[Test 3] Generating G(50, 0.1)...")
    G3 = erdos_renyi(n=50, p=0.1)
    print(G3)
    print(f"Edges: {len(G3.edges)}")
    print(f"Average degree: {2 * len(G3.edges) / G3.n:.2f}")
    print(f"Connected: {G3.is_connected()}")

    visualize(G3, title=f"G(50, 0.1) - Connected: {G3.is_connected()}")
    plt.savefig("assets/graph_50.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: assets/graph_50.png")
    plt.close()

    # Test 4: Large graph with low probability (likely disconnected)
    print("\n[Test 4] Generating G(100, 0.05)...")
    G4 = erdos_renyi(n=100, p=0.05)
    print(G4)
    print(f"Edges: {len(G4.edges)}")
    print(f"Average degree: {2 * len(G4.edges) / G4.n:.2f}")
    print(f"Connected: {G4.is_connected()}")

    visualize(G4, title=f"G(100, 0.05) - Connected: {G4.is_connected()}", show_labels=False)
    plt.savefig("assets/graph_100.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: assets/graph_100.png")
    plt.close()

    # Test 5: Very sparse graph (definitely disconnected)
    print("\n[Test 5] Generating G(20, 0.02)...")
    G5 = erdos_renyi(n=20, p=0.02)
    print(G5)
    print(f"Edges: {len(G5.edges)}")
    print(f"Average degree: {2 * len(G5.edges) / G5.n:.2f}")
    print(f"Connected: {G5.is_connected()}")

    visualize(G5, title=f"G(20, 0.02) - Connected: {G5.is_connected()}")
    plt.savefig("assets/graph_sparse.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: assets/graph_sparse.png")
    plt.close()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
