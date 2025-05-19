import random
import matplotlib.pyplot as plt


# ------------------------------------------------------
# HÀM ĐỌC DỮ LIỆU ĐẦU VÀO
# ------------------------------------------------------
def read_input():
    """
    Đọc dữ liệu đầu vào theo định dạng:
      - Dòng 1: N K
      - Dòng 2: d(1), d(2), ..., d(N)
      - Tiếp theo (N+1) dòng, mỗi dòng gồm (N+1) số nguyên: ma trận t
    Trả về: N, K, d_list, t_matrix
    """
    N, K = map(int, input().split())
    d_list = list(map(int, input().split()))

    t_matrix = []
    for _ in range(N + 1):
        row = list(map(int, input().split()))
        t_matrix.append(row)

    return N, K, d_list, t_matrix


# ------------------------------------------------------
# HÀM TÍNH THỜI GIAN CỦA 1 TUYẾN
# ------------------------------------------------------
def route_time(route, d_list, t_matrix):
    """
    route: danh sách khách hàng (không bao gồm depot)
    d_list: thời gian phục vụ của khách hàng i (d_list[i-1] ứng với khách hàng i)
    t_matrix[u][v]: thời gian di chuyển từ điểm u đến v (u,v>=0, với 0 là depot)

    Trả về tổng thời gian của tuyến, tính từ depot -> khách hàng đầu tiên -> ... -> khách hàng cuối -> depot.
    """
    if not route:
        return 0

    total_time = 0
    # Từ depot (0) đến khách hàng đầu tiên
    total_time += t_matrix[0][route[0]]
    total_time += d_list[route[0] - 1]

    # Di chuyển qua các khách hàng
    for i in range(len(route) - 1):
        total_time += t_matrix[route[i]][route[i + 1]]
        total_time += d_list[route[i + 1] - 1]

    # Từ khách hàng cuối quay về depot
    total_time += t_matrix[route[-1]][0]
    return total_time


# ------------------------------------------------------
# GIẢI MÃ CHROMOSOME THÀNH CÁC TUYẾN
# ------------------------------------------------------
def decode_solution(solution, N, K):
    """
    solution gồm 2 phần: (perm, cuts)
      - perm: hoán vị của các khách hàng [1..N]
      - cuts: danh sách K-1 điểm cắt (đã sắp tăng dần)
    Trả về danh sách K tuyến, mỗi tuyến là list khách hàng.
    """
    perm, cuts = solution
    routes = []
    start = 0
    for i in range(K - 1):
        end = cuts[i]
        routes.append(perm[start:end])
        start = end
    routes.append(perm[start:])
    return routes


# ------------------------------------------------------
# HÀM TÍNH FITNESS (Mục tiêu: MINIMIZE MAKESPAN)
# ------------------------------------------------------
def compute_fitness(solution, N, K, d_list, t_matrix):
    """
    Fitness của lời giải là thời gian làm việc lớn nhất (makespan) trong các tuyến.
    """
    routes = decode_solution(solution, N, K)
    times = [route_time(route, d_list, t_matrix) for route in routes]
    return max(times)


# ------------------------------------------------------
# THUẬT TOÁN LEO ĐỒI (Hill Climbing)
# ------------------------------------------------------
def hill_climbing(solution, N, K, d_list, t_matrix, max_iterations=50, max_no_improve=20):
    """
    Cải thiện một lời giải bằng thuật toán leo đồi.

    Args:
        solution: Lời giải hiện tại (perm, cuts)
        N, K, d_list, t_matrix: Tham số bài toán
        max_iterations: Số lần lặp tối đa
        max_no_improve: Số lần lặp tối đa không cải thiện trước khi dừng

    Returns:
        Lời giải đã cải thiện và fitness của nó
    """
    perm, cuts = solution
    current_fitness = compute_fitness(solution, N, K, d_list, t_matrix)
    no_improve_count = 0

    for _ in range(max_iterations):
        improved = False

        # 1. Thử hoán đổi vị trí hai khách hàng trong hoán vị
        for _ in range(3):  # Thử một vài lần hoán đổi ngẫu nhiên
            i, j = random.sample(range(len(perm)), 2)
            new_perm = perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            new_solution = (new_perm, cuts)
            new_fitness = compute_fitness(new_solution, N, K, d_list, t_matrix)

            if new_fitness < current_fitness:
                perm, cuts = new_perm, cuts
                current_fitness = new_fitness
                improved = True
                break

        # 2. Thử thay đổi điểm cắt (nếu K > 1)
        if not improved and K > 1:
            for _ in range(3):  # Thử một vài lần thay đổi cắt ngẫu nhiên
                idx_cut = random.randint(0, K - 2)
                possible_positions = set(range(1, N))
                for c in cuts:
                    possible_positions.discard(c)
                if possible_positions:
                    new_cuts = cuts.copy()
                    new_cut = random.choice(list(possible_positions))
                    new_cuts[idx_cut] = new_cut
                    new_cuts.sort()
                    new_solution = (perm, new_cuts)
                    new_fitness = compute_fitness(new_solution, N, K, d_list, t_matrix)

                    if new_fitness < current_fitness:
                        perm, cuts = perm, new_cuts
                        current_fitness = new_fitness
                        improved = True
                        break

        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        else:
            no_improve_count = 0

    return (perm, cuts), current_fitness


# ------------------------------------------------------
# TẠO MỘT CHROMOSOME BAN ĐẦU SỬ DỤNG HEURISTIC VÀ RANDOM
# ------------------------------------------------------
def create_heuristic_solution(N, K, d_list, t_matrix):
    # Sử dụng thuật toán nearest neighbor để tạo hoán vị
    unvisited = list(range(1, N + 1))
    current = 0
    perm = []
    while unvisited:
        next_customer = min(unvisited, key=lambda x: t_matrix[current][x])
        perm.append(next_customer)
        unvisited.remove(next_customer)
        current = next_customer
    # Chia hoán vị thành K tuyến theo phân hoạch đều số lượng khách
    cuts = []
    index = 0
    for i in range(K - 1):
        seg_len = N // K + (1 if i < (N % K) else 0)
        index += seg_len
        cuts.append(index)
    return (perm, cuts)


def create_random_solution(N, K, d_list, t_matrix):
    # Dùng thuật toán heuristic với xác suất 0.2, ngược lại dùng random
    if random.random() < 0.5:
        return create_heuristic_solution(N, K, d_list, t_matrix)
    else:
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        cuts = sorted(random.sample(range(1, N), K - 1))
        return (perm, cuts)


# ------------------------------------------------------
# PHƯƠNG PHÁP CHỌN LỌC (Tournament Selection)
# ------------------------------------------------------
def selection(population, fitnesses, tournament_size=2):
    pop_size = len(population)
    best_idx = None
    for _ in range(tournament_size):
        idx = random.randint(0, pop_size - 1)
        if best_idx is None or fitnesses[idx] < fitnesses[best_idx]:
            best_idx = idx
    return best_idx


# ------------------------------------------------------
# PHƯƠNG PHÁP LAI GHÉP (Crossover)
# ------------------------------------------------------
def crossover(p1, p2, N, K):

    perm1, cuts1 = p1
    perm2, cuts2 = p2

    child_perm = pmx_crossover(perm1, perm2)
    child_cuts = cuts1[:] if random.random() < 0.5 else cuts2[:]
    child_cuts = sorted(child_cuts)
    return (child_perm, child_cuts)


def pmx_crossover(parent1, parent2):

    size = len(parent1)
    cxpoint1 = random.randint(0, size - 1)
    cxpoint2 = random.randint(cxpoint1, size - 1)

    child = [None] * size

    for i in range(cxpoint1, cxpoint2 + 1):
        child[i] = parent1[i]

    for i in range(cxpoint1, cxpoint2 + 1):
        if parent2[i] not in child:
            val = parent2[i]
            pos = i
            while True:
                val = parent1[pos]
                pos = parent2.index(val)
                if child[pos] is None:
                    child[pos] = parent2[i]
                    break

    # Điền các vị trí còn lại từ parent2
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    return child


# ------------------------------------------------------
# PHƯƠNG PHÁP ĐỘT BIẾN (Mutation)
# ------------------------------------------------------
def mutation(solution, N, K, mutation_rate=0.1):
    perm, cuts = solution

    if random.random() < mutation_rate:
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    if random.random() < mutation_rate and K > 1:
        idx_cut = random.randint(0, K - 2)
        possible_positions = set(range(1, N))
        for c in cuts:
            possible_positions.discard(c)
        if possible_positions:
            new_cut = random.choice(list(possible_positions))
            cuts[idx_cut] = new_cut
            cuts.sort()

    return (perm, cuts)


# ------------------------------------------------------
# GIẢI THUẬT DI TRUYỀN CHÍNH (Genetic Algorithm)
# ------------------------------------------------------
def genetic_algorithm(N, K, d_list, t_matrix,
                      pop_size=50, generations=200,
                      crossover_rate=0.8, mutation_rate=0.1,
                      use_hill_climbing=True,
                      hill_climbing_freq=10):  # Áp dụng hill climbing mỗi X thế hệ
    
    population = [create_random_solution(N, K, d_list, t_matrix) for _ in range(pop_size)]
    fitnesses = [compute_fitness(ind, N, K, d_list, t_matrix) for ind in population]

    best_sol = None
    best_fit = float('inf')

    best_fitness_history = []
    avg_fitness_history = []

    for gen in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            idx1 = selection(population, fitnesses)
            idx2 = selection(population, fitnesses)
            p1 = population[idx1]
            p2 = population[idx2]

            if random.random() < crossover_rate:
                child = crossover(p1, p2, N, K)
            else:
                child = p1  # Giữ nguyên cá thể hiện có

            child = mutation(child, N, K, mutation_rate)
            new_population.append(child)

        population = new_population
        fitnesses = [compute_fitness(ind, N, K, d_list, t_matrix) for ind in population]

        gen_best_fit = min(fitnesses)
        gen_avg_fit = sum(fitnesses) / len(fitnesses)

        best_fitness_history.append(gen_best_fit)
        avg_fitness_history.append(gen_avg_fit)

        best_idx = fitnesses.index(min(fitnesses))
        if fitnesses[best_idx] < best_fit:
            best_fit = fitnesses[best_idx]
            best_sol = population[best_idx]

        # Áp dụng hill climbing cho lời giải tốt nhất sau một số thế hệ nhất định
        if use_hill_climbing and (gen + 1) % hill_climbing_freq == 0:
            improved_sol, improved_fit = hill_climbing(best_sol, N, K, d_list, t_matrix)
            if improved_fit < best_fit:
                best_sol = improved_sol
                best_fit = improved_fit
                # Thêm lời giải đã cải thiện vào quần thể bằng cách thay thế cá thể tồi nhất
                worst_idx = fitnesses.index(max(fitnesses))
                population[worst_idx] = improved_sol
                fitnesses[worst_idx] = improved_fit

        # In thông tin quá trình mà không plot liên tục
        # if (gen + 1) % 50 == 0 or gen == generations - 1:
        #     print(
        #         f"Generation {gen + 1}/{generations} | Best makespan: {best_fit} | Average fitness: {gen_avg_fit:.2f}")

    # # Sau khi kết thúc vòng lặp, plot kết quả 1 lần duy nhất.
    # plt.figure(figsize=(12, 5))

    # # Plot fitness evolution
    # plt.subplot(1, 2, 1)
    # plt.plot(best_fitness_history, 'b-', label='Best Fitness')
    # plt.plot(avg_fitness_history, 'r-', label='Avg Fitness')
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness (Makespan)')
    # plt.title('Fitness Evolution')
    # plt.legend()

    # # Plot current best solution theo từng tuyến
    # plt.subplot(1, 2, 2)
    # routes = decode_solution(best_sol, N, K)
    # times = [route_time(route, d_list, t_matrix) for route in routes]

    # plt.bar(range(1, K + 1), times)
    # plt.axhline(y=max(times), color='r', linestyle='--', label='Makespan')
    # plt.xlabel('Route')
    # plt.ylabel('Time')
    # plt.title('Best Solution Routes')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    return best_sol, best_fit


# ------------------------------------------------------
# HÀM XUẤT KẾT QUẢ THEO YÊU CẦU
# ------------------------------------------------------
def print_solution(solution, N, K):
    """
    In ra kết quả theo định dạng:
      - Với mỗi k=1..K:
        + Dòng 1: Số Lk (số điểm của tuyến, bao gồm depot ở đầu và cuối)
        + Dòng 2: Danh sách các điểm theo thứ tự: depot -> khách hàng -> depot
    """
    routes = decode_solution(solution, N, K)
    print(len(routes))
    for route in routes:
        full_route = [0] + route + [0]
        print(len(full_route))
        print(" ".join(map(str, full_route)))


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
if __name__ == "__main__":
    # 1. Đọc input
    N, K, d_list, t_matrix = read_input()

    # 2. Chạy GA để tìm lời giải tối ưu
    best_sol, best_fit = genetic_algorithm(N, K, d_list, t_matrix,
                                        pop_size=100,
                                        generations=500,
                                        crossover_rate=0.8,
                                        mutation_rate=0.1,
                                        use_hill_climbing=True,
                                        hill_climbing_freq=20)
    # # 3. Xuất kết quả

    print_solution(best_sol, N, K)