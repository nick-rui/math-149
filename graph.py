import random
import matplotlib.pyplot as plt
import math

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
        """Check if edge (i,j) exists."""
        if i > j:
            i, j = j, i
        return (i, j) in self.edges
    
    def degree(self, v):
        """Return the degree of vertex v."""
        return len(self.adj[v])
    
    def num_edges(self):
        """Return the number of edges."""
        return len(self.edges)
    
    def __repr__(self):
        return f"Graph(n={self.n}, edges={self.num_edges()})"


def erdos_renyi(n, p):
    """
    Generate an Erdős-Rényi random graph G(n, p).
    
    For each possible pair of vertices, add an edge with probability p.
    
    Args:
        n: Number of vertices
        p: Probability of edge between any two vertices (0 <= p <= 1)
    
    Returns:
        Graph object
    """
    G = Graph(n)
    
    # Iterate over all possible pairs of vertices
    for i in range(n):
        for j in range(i + 1, n):
            # Add edge with probability p
            if random.random() < p:
                G.add_edge(i, j)
    
    return G


def visualize(G, title="Random Graph", layout="circle", figsize=(8, 8)):
    """
    Visualize a graph using matplotlib.
    
    Args:
        G: Graph object to visualize
        title: Title for the plot
        layout: Layout algorithm - "circle" or "spring"
        figsize: Figure size as (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute vertex positions
    if layout == "circle":
        pos = _circular_layout(G.n)
    elif layout == "spring":
        pos = _spring_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    # Draw edges
    for i, j in G.edges:
        x_coords = [pos[i][0], pos[j][0]]
        y_coords = [pos[i][1], pos[j][1]]
        ax.plot(x_coords, y_coords, 'gray', linewidth=0.5, alpha=0.6, zorder=1)
    
    # Draw vertices
    x_coords = [pos[i][0] for i in range(G.n)]
    y_coords = [pos[i][1] for i in range(G.n)]
    ax.scatter(x_coords, y_coords, c='lightblue', s=300, 
               edgecolors='darkblue', linewidths=2, zorder=2)
    
    # Add vertex labels
    for i in range(G.n):
        ax.text(pos[i][0], pos[i][1], str(i), 
                ha='center', va='center', fontsize=10, zorder=3)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig, ax


def _circular_layout(n):
    """
    Position n vertices evenly around a circle.
    
    Returns:
        dict mapping vertex index -> (x, y) coordinates
    """
    pos = {}
    for i in range(n):
        angle = 2 * math.pi * i / n
        pos[i] = (math.cos(angle), math.sin(angle))
    return pos


def _spring_layout(G, iterations=50, k=None):
    """
    Simple spring/force-directed layout algorithm.
    
    Args:
        G: Graph object
        iterations: Number of iterations to run
        k: Optimal distance between vertices (default: 1/sqrt(n))
    
    Returns:
        dict mapping vertex index -> (x, y) coordinates
    """
    n = G.n
    if k is None:
        k = 1.0 / math.sqrt(n)
    
    # Initialize positions randomly
    pos = {i: (random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(n)}
    
    # Run spring algorithm
    for _ in range(iterations):
        # Calculate forces
        forces = {i: [0.0, 0.0] for i in range(n)}
        
        # Repulsive forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    # Repulsive force
                    force = k * k / distance
                    fx = force * dx / distance
                    fy = force * dy / distance
                    
                    forces[i][0] -= fx
                    forces[i][1] -= fy
                    forces[j][0] += fx
                    forces[j][1] += fy
        
        # Attractive forces for edges
        for i, j in G.edges:
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                # Attractive force
                force = distance * distance / k
                fx = force * dx / distance
                fy = force * dy / distance
                
                forces[i][0] += fx
                forces[i][1] += fy
                forces[j][0] -= fx
                forces[j][1] -= fy
        
        # Update positions (with damping)
        damping = 0.1
        for i in range(n):
            pos[i] = (
                pos[i][0] + damping * forces[i][0],
                pos[i][1] + damping * forces[i][1]
            )
    
    return pos


# Example usage and testing
if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    # Generate a small random graph
    print("Generating Erdős-Rényi graph G(10, 0.3)...")
    G = erdos_renyi(n=10, p=0.3)
    print(G)
    print(f"Edges: {sorted(G.edges)}")
    print(f"Degrees: {[G.degree(i) for i in range(G.n)]}")
    
    # Visualize with circular layout
    visualize(G, title="G(10, 0.3) - Circular Layout", layout="circle")
    plt.savefig("graph_circular.png", dpi=150, bbox_inches='tight')
    print("Saved: graph_circular.png")
    
    # Visualize with spring layout
    visualize(G, title="G(10, 0.3) - Spring Layout", layout="spring")
    plt.savefig("graph_spring.png", dpi=150, bbox_inches='tight')
    print("Saved: graph_spring.png")
    
    # Generate a larger, denser graph
    print("\nGenerating larger graph G(20, 0.2)...")
    G2 = erdos_renyi(n=20, p=0.2)
    print(G2)
    
    visualize(G2, title="G(20, 0.2) - Spring Layout", layout="spring")
    plt.savefig("graph_large.png", dpi=150, bbox_inches='tight')
    print("Saved: graph_large.png")
    
    plt.show()