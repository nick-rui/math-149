"""
Experiments for analyzing random graph properties.

This module provides experiments to measure various properties of Erdős-Rényi
random graphs, including connectivity probability.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from graph import erdos_renyi


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def get_p_connected(n, p, num_trials=1000):
    """
    Get the probability of a graph being connected, average Fiedler value, and average Euler characteristic.

    Args:
        n: Number of vertices
        p: Probability of edge between any two vertices
        num_trials: Number of random graphs to generate (default: 1000)

    Returns:
        tuple: (prob_connected, avg_lambda_2, avg_euler) where
            - prob_connected: Probability that graph is connected
            - avg_lambda_2: Average Fiedler value (algebraic connectivity)
            - avg_euler: Average Euler characteristic (χ = V - E)
    """
    connected_count = 0
    lambda_2_values = []
    euler_values = []

    for _ in range(num_trials):
        G = erdos_renyi(n, p)
        lambda_2 = G.fiedler_value()
        euler = G.euler_characteristic()
        lambda_2_values.append(lambda_2)
        euler_values.append(euler)

        if G.is_connected():
            connected_count += 1

    prob_connected = connected_count / num_trials
    avg_lambda_2 = np.mean(lambda_2_values)
    avg_euler = np.mean(euler_values)

    return prob_connected, avg_lambda_2, avg_euler


def experiment_connectivity_vs_p(n, p_values=None, num_trials=1000):
    """
    Run experiment to measure P(connected), average lambda_2, and average Euler characteristic for varying p with fixed n.

    Args:
        n: Number of vertices (fixed)
        p_values: List of p values to test (default: np.linspace(0, 1, 21))
        num_trials: Number of trials per p value

    Returns:
        tuple: (p_values, probabilities, lambda_2_values, euler_values) arrays
    """
    if p_values is None:
        p_values = np.linspace(0, 1, 21)  # 0, 0.05, 0.10, ..., 1.0

    probabilities = []
    lambda_2_values = []
    euler_values = []

    print(f"Running connectivity experiment for n={n}")
    print(f"Testing {len(p_values)} values of p with {num_trials} trials each")
    print("=" * 60)

    for i, p in enumerate(p_values, 1):
        prob, avg_lambda_2, avg_euler = get_p_connected(n, p, num_trials)
        probabilities.append(prob)
        lambda_2_values.append(avg_lambda_2)
        euler_values.append(avg_euler)
        print(f"[{i}/{len(p_values)}] p={p:.3f} -> P(connected)={prob:.3f}, avg λ₂={avg_lambda_2:.4f}, avg χ={avg_euler:.2f}")

    return np.array(p_values), np.array(probabilities), np.array(lambda_2_values), np.array(euler_values)


# ============================================================================
# VISUALIZATION AND OUTPUT
# ============================================================================

def plot_connectivity_probability(n, p_values, probabilities, output_path="assets/connectivity_experiment.png"):
    """
    Create a line chart showing P(connected) vs p.

    Args:
        n: Number of vertices
        p_values: Array of p values
        probabilities: Array of connectivity probabilities
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    plt.plot(p_values, probabilities, 'b-', linewidth=2, marker='o', markersize=6)

    plt.xlabel('p (Edge Probability)', fontsize=12)
    plt.ylabel('P(Graph is Connected)', fontsize=12)
    plt.title(f'Probability of Connectivity vs Edge Probability for G({n}, p)',
              fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    # Add reference lines
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='P = 0.5')

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {output_path}")
    plt.close()


def plot_fiedler_values(n, p_values, lambda_2_values, output_path="assets/fiedler_experiment.png"):
    """
    Create a line chart showing average Fiedler value (λ₂) vs p.

    Args:
        n: Number of vertices
        p_values: Array of p values
        lambda_2_values: Array of average Fiedler values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    plt.plot(p_values, lambda_2_values, 'g-', linewidth=2, marker='s', markersize=6)

    plt.xlabel('p (Edge Probability)', fontsize=12)
    plt.ylabel('Average λ₂ (Fiedler Value)', fontsize=12)
    plt.title(f'Average Algebraic Connectivity vs Edge Probability for G({n}, p)',
              fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {output_path}")
    plt.close()


def plot_euler_characteristic(n, p_values, euler_values, output_path="assets/euler_experiment.png"):
    """
    Create a line chart showing average Euler characteristic (χ) vs p.

    Args:
        n: Number of vertices
        p_values: Array of p values
        euler_values: Array of average Euler characteristic values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    plt.plot(p_values, euler_values, 'r-', linewidth=2, marker='D', markersize=6)

    plt.xlabel('p (Edge Probability)', fontsize=12)
    plt.ylabel('Average χ (Euler Characteristic)', fontsize=12)
    plt.title(f'Average Euler Characteristic vs Edge Probability for G({n}, p)',
              fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)

    # Add reference line at χ = 0
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {output_path}")
    plt.close()


def save_data(n, p_values, probabilities, lambda_2_values, euler_values, output_path="assets/connectivity_data.csv"):
    """
    Save experimental data to CSV file.

    Args:
        n: Number of vertices
        p_values: Array of p values
        probabilities: Array of connectivity probabilities
        lambda_2_values: Array of average Fiedler values
        euler_values: Array of average Euler characteristic values
        output_path: Path to save the data
    """
    with open(output_path, 'w') as f:
        f.write(f"# Connectivity, Fiedler value, and Euler characteristic experiment for n={n}\n")
        f.write("p,P(connected),avg_lambda_2,avg_euler_char\n")
        for p, prob, lambda_2, euler in zip(p_values, probabilities, lambda_2_values, euler_values):
            f.write(f"{p:.4f},{prob:.4f},{lambda_2:.6f},{euler:.6f}\n")

    print(f"✓ Data saved: {output_path}")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run connectivity experiments on random graphs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--n',
        type=int,
        help='Number of vertices in the graph'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=1000,
        help='Number of trials per p value'
    )
    parser.add_argument(
        '--num-p-values',
        type=int,
        default=21,
        help='Number of p values to test from 0 to 1'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='assets',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.n <= 0:
        print("Error: n must be a positive integer")
        return

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate p values
    p_values = np.linspace(0, 1, args.num_p_values)

    # Run experiment
    print("=" * 60)
    print("CONNECTIVITY, FIEDLER VALUE & EULER CHARACTERISTIC EXPERIMENT")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    p_values, probabilities, lambda_2_values, euler_values = experiment_connectivity_vs_p(
        args.n,
        p_values=p_values,
        num_trials=args.trials
    )

    # Save results
    print("\n" + "=" * 60)
    csv_path = os.path.join(args.output_dir, "connectivity_data.csv")
    connectivity_png = os.path.join(args.output_dir, "connectivity_experiment.png")
    fiedler_png = os.path.join(args.output_dir, "fiedler_experiment.png")
    euler_png = os.path.join(args.output_dir, "euler_experiment.png")

    save_data(args.n, p_values, probabilities, lambda_2_values, euler_values, output_path=csv_path)
    plot_connectivity_probability(args.n, p_values, probabilities, output_path=connectivity_png)
    plot_fiedler_values(args.n, p_values, lambda_2_values, output_path=fiedler_png)
    plot_euler_characteristic(args.n, p_values, euler_values, output_path=euler_png)

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
