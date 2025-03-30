import random
import sys
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
    for _ in range(N+1):
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
    total_time += d_list[route[0]-1]

    # Di chuyển qua các khách hàng
    for i in range(len(route)-1):
        total_time += t_matrix[route[i]][route[i+1]]
        total_time += d_list[route[i+1]-1]

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
    for i in range(K-1):
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
# TẠO MỘT CHROMOSOME NGẪU NHIÊN
# ------------------------------------------------------
def create_random_solution(N, K):
    """
    Tạo hoán vị ngẫu nhiên của N khách hàng và K-1 điểm cắt.
    """
    perm = list(range(1, N+1))
    random.shuffle(perm)
    # Chọn K-1 điểm cắt ngẫu nhiên trong khoảng [1, N-1]
    cuts = sorted(random.sample(range(1, N), K-1))
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
    """
    Lai ghép giữa 2 cá thể: sử dụng PMX cho phần perm và chọn ngẫu nhiên cuts từ một trong 2 cha.
    """
    perm1, cuts1 = p1
    perm2, cuts2 = p2

    child_perm = pmx_crossover(perm1, perm2)
    child_cuts = cuts1[:] if random.random() < 0.5 else cuts2[:]
    child_cuts = sorted(child_cuts)
    return (child_perm, child_cuts)

def pmx_crossover(parent1, parent2):
    """
    Thực hiện lai ghép PMX giữa 2 hoán vị.
    """
    size = len(parent1)
    cxpoint1 = random.randint(0, size - 1)
    cxpoint2 = random.randint(cxpoint1, size - 1)

    child = [None] * size
    # Copy đoạn giữa từ parent1
    for i in range(cxpoint1, cxpoint2 + 1):
        child[i] = parent1[i]

    # Map các giá trị từ parent2 sang child
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

    # Đột biến hoán vị: đổi chỗ ngẫu nhiên 2 phần tử
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    # Đột biến cuts: thay đổi ngẫu nhiên 1 điểm cắt
    if random.random() < mutation_rate and K > 1:
        idx_cut = random.randint(0, K-2)
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
                      crossover_rate=0.8, mutation_rate=0.1):
    population = [create_random_solution(N, K) for _ in range(pop_size)]
    fitnesses = [compute_fitness(ind, N, K, d_list, t_matrix) for ind in population]

    best_sol = None
    best_fit = float('inf')

    # For tracking progress
    best_fitness_history = []
    avg_fitness_history = []

    # Vòng lặp GA
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

        # Tính toán thống kê của thế hệ hiện tại
        gen_best_fit = min(fitnesses)
        gen_avg_fit = sum(fitnesses) / len(fitnesses)

        best_fitness_history.append(gen_best_fit)
        avg_fitness_history.append(gen_avg_fit)

        best_idx = fitnesses.index(min(fitnesses))
        if fitnesses[best_idx] < best_fit:
            best_fit = fitnesses[best_idx]
            best_sol = population[best_idx]

        # In thông tin quá trình mà không plot liên tục
        if (gen+1) % 50 == 0 or gen == generations - 1:
            print(f"Generation {gen+1}/{generations} | Best makespan: {best_fit} | Average fitness: {gen_avg_fit:.2f}")

    # Sau khi kết thúc vòng lặp, plot kết quả 1 lần duy nhất.
    plt.figure(figsize=(12, 5))

    # Plot fitness evolution
    plt.subplot(1, 2, 1)
    plt.plot(best_fitness_history, 'b-', label='Best Fitness')
    plt.plot(avg_fitness_history, 'r-', label='Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Makespan)')
    plt.title('Fitness Evolution')
    plt.legend()

    # Plot current best solution theo từng tuyến
    plt.subplot(1, 2, 2)
    routes = decode_solution(best_sol, N, K)
    times = [route_time(route, d_list, t_matrix) for route in routes]

    plt.bar(range(1, K+1), times)
    plt.axhline(y=max(times), color='r', linestyle='--', label='Makespan')
    plt.xlabel('Route')
    plt.ylabel('Time')
    plt.title('Best Solution Routes')
    plt.legend()

    plt.tight_layout()
    plt.show()

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
                                           generations=20000,
                                           crossover_rate=0.8,
                                           mutation_rate=0.1)
    print("Best fitness:", best_fit)
    print("Best solution:", best_sol)
    # 3. Xuất kết quả
    print_solution(best_sol, N, K)
    # khởi  tạo sớm
    # nâng cấp mutation
    # local search 
    # 
    
    # n = int(input())
    # ans = 0
    # def route_time2(route, d_list, t_matrix):
    #     """
    #     route: danh sách khách hàng (không bao gồm depot)
    #     d_list: thời gian phục vụ của khách hàng i (d_list[i-1] ứng với khách hàng i)
    #     t_matrix[u][v]: thời gian di chuyển từ điểm u đến v (u,v>=0, với 0 là depot)
        
    #     Trả về tổng thời gian của tuyến, tính từ depot -> khách hàng đầu tiên -> ... -> khách hàng cuối -> depot.
    #     """
    #     if not route:
    #         return 0

    #     total_time = 0

    #     # Di chuyển qua các khách hàng
    #     for i in range(len(route)-1):
    #         total_time += t_matrix[route[i]][route[i+1]]
    #         total_time += d_list[route[i+1]-1]

    #     return total_time
    # for i in range(n):
    #     k = int(input())
    #     route = list(map(int, input().split()))
    #     print(route)
    #     print(route_time2(route, d_list, t_matrix))
    #     ans = max(ans, route_time2(route, d_list, t_matrix))
    # print(ans)