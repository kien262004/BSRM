import numpy as np
import random, math
import time
from models.ACO.ACO import acs_vrp, calculate_route_cost

def main(cfg):
    """Main function for ACO algorithm that follows the common interface.
    
    Args:
        cfg: Configuration dictionary containing:
            - N: Number of customers
            - K: Number of technicians
            - d: List of service times
            - t: Travel time matrix
            - output_file: Path to output file
    """
    # Extract parameters from config
    N = cfg['N']
    K = cfg['K']
    d = cfg['D']
    t = cfg['C']
    d = [0] + d
    # Run ACO algorithm
    solution = acs_vrp(N, K, t, d)
    
    # Calculate costs for each route
    costs = [calculate_route_cost(r, t, d) for r in solution]
    
    # Format solution for output
    schedule = []
    for route in solution:
        # Convert route to list of customer indices (excluding depot)
        customer_route = [node for node in route[1:-1]]  # Exclude first and last depot
        schedule.append(customer_route)
    
    return schedule 