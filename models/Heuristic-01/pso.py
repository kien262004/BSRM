import random
import copy
import numpy as np
import math

# def read_input(filename):
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         N, K = map(int, lines[0].split())  # N: số khách hàng, K: số nhân viên
#         d = list(map(int, lines[1].split()))  # Thời gian bảo trì tại các khách hàng
#         c = [list(map(int, line.split())) for line in lines[2:]]  # Ma trận chi phí di chuyển
#     d.insert(0, 0)  # Thêm thời gian bảo trì tại trụ sở (điểm 0) = 0
#     return N, K, d, c

def compute_mst_cost(c):
    """Tính MST cost trên đồ thị đầy đủ qua Prim."""
    N = len(c)
    visited = [False]*N
    min_edge = [float('inf')]*N
    min_edge[0] = 0
    total = 0
    for _ in range(N):
        u = min((w,i) for i,w in enumerate(min_edge) if not visited[i])[1]
        visited[u] = True
        total += min_edge[u]
        for v in range(N):
            if not visited[v] and c[u][v] < min_edge[v]:
                min_edge[v] = c[u][v]
    return total

def improved_lower_bound(d, c, K):
    # MST bound
    mst_cost = compute_mst_cost(c)
    # thêm cạnh nhỏ nhất từ 0
    min0 = min(c[0][i] for i in range(1,len(c)))
    lb = (mst_cost + min0) / K
    return lb

def calculate_tsp_cost(route, c):
    total_cost = 0
    for i in range(len(route) - 1):
        total_cost += c[route[i]][route[i + 1]]
    total_cost += c[route[-1]][route[0]]
    return total_cost

def calculate_total_time(route, d, c):
    full_route = route
    travel_time = calculate_tsp_cost(full_route, c)
    maintenance_time = sum(d[i] for i in route if i != 0)
    total = travel_time + maintenance_time
    return total

def tsp_2opt_step(route_indices, d, c):
    N = len(route_indices)
    curr_cost = calculate_total_time(route_indices, d, c)
    for i in range(1, N - 1):
        for j in range(i + 1, N):
            temp = route_indices.copy()
            temp[i], temp[j] = temp[j], temp[i]
            new_cost = calculate_total_time(temp, d, c)
            if new_cost < curr_cost:
                return True, temp
    return False, route_indices


def tsp_2opt(route, d, c, max_iter=10000):
    N = len(route)
    for _ in range(max_iter):
        improved, route = tsp_2opt_step(route, d, c)
        if not improved:
            break
    optimized_route = route.copy()
    return optimized_route

def random_partition(customers, K):
    random.shuffle(customers)
    groups = [[] for _ in range(K)]
    for i, customer in enumerate(customers):
        groups[i % K].append(customer)
    return groups


def swap_nodes(route1, route2, i, j, d, c, lower_bound, max_diff):
    route1_copy = copy.deepcopy(route1)
    route2_copy = copy.deepcopy(route2)
    route1_copy[i], route2_copy[j] = route2_copy[j], route1_copy[i]
    new_route1 = tsp_2opt(route1_copy, d, c)
    new_route2 = tsp_2opt(route2_copy, d, c)
    time1 = calculate_total_time(new_route1, d, c)
    time2 = calculate_total_time(new_route2, d, c)
    if time1 < lower_bound or time2 < lower_bound or abs(time1 - time2) > max_diff:
        return route1, route2
    return new_route1, new_route2

def move_node(route1, route2, i, d, c, lower_bound, max_diff):
    route1_copy = copy.deepcopy(route1)
    route2_copy = copy.deepcopy(route2)
    tmp = route1_copy[i]
    route1_copy.pop(i)
    route2_copy.append(tmp)
    new_route1 = tsp_2opt(route1_copy, d, c)
    new_route2 = tsp_2opt(route2_copy, d, c)
    time1 = calculate_total_time(new_route1, d, c)
    time2 = calculate_total_time(new_route2, d, c)
    if time1 < lower_bound or time2 < lower_bound or abs(time1 - time2) > max_diff:
        return route1, route2
    return new_route1, new_route2

def position_to_partition(position, customers, K):
    sorted_indices = np.argsort(position)
    sorted_customers = [customers[i] for i in sorted_indices]
    groups = [[] for _ in range(K)]
    customers_per_group = len(customers) // K
    remainder = len(customers) % K
    
    idx = 0
    for i in range(K):
        num_customers = customers_per_group + (1 if i < remainder else 0)
        groups[i] = sorted_customers[idx:idx + num_customers]
        idx += num_customers
    
    return groups

def fitness_value(position, customers, K, d, c, lower_bound):
    groups = position_to_partition(position, customers, K)
    routes = [[0] + group for group in groups]
    for i in range(K):
        routes[i] = tsp_2opt(routes[i], d, c)
    times = [calculate_total_time(route, d, c) for route in routes]
    if any(time < lower_bound for time in times):
        return float('inf')
    return max(times), routes

def particle_swarm_optimization(customers, K, d, c, lower_bound, number_of_particles=20, number_of_iterations=100):
    kappa = 1
    phi1 = 2.05
    phi2 = 2.05
    phi = phi1 + phi2
    chi = (2 * kappa) / (abs(2 - phi - math.sqrt(abs(phi**2 - 4 * phi))))
    W = chi
    C1 = chi * phi1
    C2 = chi * phi2
    
    N = len(customers)
    vector_position_particle = [[random.uniform(-10, 10) for _ in range(N)] for _ in range(number_of_particles)]
    vector_velocity = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(number_of_particles)]
    
    pb_position = copy.deepcopy(vector_position_particle)
    pb_fitness = [float('inf')] * number_of_particles
    gb_fitness = float('inf')
    gb_position = copy.deepcopy(vector_position_particle[0])
    gb_routes = None
    
    for iteration in range(number_of_iterations):
        for i in range(number_of_particles):
            fitness, routes = fitness_value(vector_position_particle[i], customers, K, d, c, lower_bound)
            if fitness < pb_fitness[i]:
                pb_fitness[i] = fitness
                pb_position[i] = copy.deepcopy(vector_position_particle[i])
            if fitness < gb_fitness:
                gb_fitness = fitness
                gb_position = copy.deepcopy(vector_position_particle[i])
                gb_routes = copy.deepcopy(routes)
        
        for i in range(number_of_particles):
            for j in range(N):
                r1, r2 = random.random(), random.random()
                vector_velocity[i][j] = (
                    W * vector_velocity[i][j] +
                    C1 * r1 * (pb_position[i][j] - vector_position_particle[i][j]) +
                    C2 * r2 * (gb_position[j] - vector_position_particle[i][j])
                )
                vector_position_particle[i][j] += vector_velocity[i][j]
    
    return gb_routes, gb_fitness


def main():
    N, K = map(int, input().split())
    d = list(map(int, input().split()))
    d.insert(0, 0)
    c = [list(map(int, input().split())) for _ in range(N + 1)]
    # filename = r"D:\project\TTUD - Copy\N_100_K_20.txt"
    # N, K, d, c = read_input(filename)  # Đọc dữ liệu
    
    lower_bound = improved_lower_bound(d, c, K)
    customers = list(range(1, N + 1))
    
    #PSO
    routes, best_fitness = particle_swarm_optimization(customers, K, d, c, lower_bound)
    print(f"Best fitness: {best_fitness}")
    
    print(K)
    for route in routes:
        route_with_end = route + [0]
        print(len(route_with_end))
        print(*route_with_end)

if __name__ == "__main__":
    main()