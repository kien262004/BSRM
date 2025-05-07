import random
import copy

# Đọc dữ liệu từ file đầu vào
def read_input(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        N, K = map(int, lines[0].split())  # N: số khách hàng, K: số nhân viên
        d = list(map(int, lines[1].split()))  # Thời gian bảo trì tại các khách hàng
        c = [list(map(int, line.split())) for line in lines[2:]]  # Ma trận chi phí di chuyển
    d.insert(0, 0)
    return N, K, d, c

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

# Tính chi phí di chuyển của một hành trình khép kín (TSP)
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

# Thực hiện một bước cải thiện hành trình bằng thuật toán 2-opt
def tsp_2opt_step(route_indices, d, c):
    N = len(route_indices)
    curr_cost = calculate_total_time(route_indices, d, c)
    # Chỉ hoán đổi i < j, và tránh cả endpoints (0 ở vị trí 0)
    for i in range(1, N - 1):
        for j in range(i + 1, N):
            temp = route_indices.copy()
            temp[i], temp[j] = temp[j], temp[i]
            new_cost = calculate_total_time(temp, d, c)
            if new_cost < curr_cost:
                return True, temp
    return False, route_indices

def tsp_2opt(route, d, c, max_iter=10000):
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


# Hoán đổi hai khách hàng (không phải đỉnh 0) giữa hai hành trình
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

# Di chuyển một khách hàng (không phải đỉnh 0) từ hành trình này sang hành trình kia
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

def main():
    #filename = r"D:\project\TTUD - Copy\N_100_K_20.txt"
    #N, K, d, c = read_input(filename)  # Đọc dữ liệu
    N, K = map(int, input().split())
    d = list(map(int, input().split()))
    d.insert(0, 0)
    c = [list(map(int, input().split())) for _ in range(N+1)]
    
    customers = list(range(1, N + 1))
    groups = random_partition(customers, K)
    routes = [[0] + group for group in groups]
    for i in range(K):
        routes[i] = tsp_2opt(routes[i], d, c)
    lower_bound = improved_lower_bound(d, c, K)
    
    max_iter = 10000
    for _ in range(max_iter):
        times = [calculate_total_time(r, d, c) for r in routes]
        max_time_idx = times.index(max(times))
        min_time_idx = times.index(min(times))
        # 1) Nếu bằng nhau thì đã cân bằng hoặc không thể cải thiện
        if max_time_idx == min_time_idx:
            break
        max_diff = times[max_time_idx] - times[min_time_idx]

        # 2) Swap
        if len(routes[max_time_idx]) > 1 and len(routes[min_time_idx]) > 1:
            i = random.randint(1, len(routes[max_time_idx]) - 1)
            j = random.randint(1, len(routes[min_time_idx]) - 1)
            new_max, new_min = swap_nodes(
                routes[max_time_idx], routes[min_time_idx],
                i, j, d, c, lower_bound, max_diff
            )

        # 3) Cập nhật times, indices trước khi move
        times = [calculate_total_time(r, d, c) for r in routes]
        max_time_idx = times.index(max(times))
        min_time_idx = times.index(min(times))
        if max_time_idx == min_time_idx:
            break
        max_diff = times[max_time_idx] - times[min_time_idx]

        # 4) Move
        if len(routes[max_time_idx]) > 1:
            i = random.randint(1, len(routes[max_time_idx]) - 1)
            new_max, new_min = move_node(
                routes[max_time_idx], routes[min_time_idx],
                i, d, c, lower_bound, max_diff
            )
            routes[max_time_idx], routes[min_time_idx] = new_max, new_min

    print(K)
    for route in routes:
        route_with_end = route + [0]
        print(len(route_with_end))
        print(*route_with_end)

if __name__ == "__main__":
    main()
