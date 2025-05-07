import sys

def segerstedt_plus(N, K, d, t):
    import heapq

    # Bước 1: Ước lượng chi phí ban đầu cho mỗi khách hàng
    chi_phi = []
    for i in range(N):
        approx_cost = t[0][i + 1] + d[i] + t[i + 1][0]
        chi_phi.append((approx_cost, i))  # (proxy_cost, customer index)

    # Sắp xếp khách hàng theo chi phí giảm dần
    chi_phi.sort(reverse=True)

    # Bước 2: Khởi tạo K nhân viên: mỗi người là (tổng_time, index, danh sách khách hàng)
    workers = [(0, k, []) for k in range(K)]

    for _, i in chi_phi:  # Duyệt theo thứ tự chi phí giảm dần
        best_k = None
        best_time = float('inf')
        best_route = []

        for idx in range(K):
            _, k, assigned = workers[idx]
            min_time_local = float('inf')
            best_local_route = []

            # Chèn khách hàng i vào mọi vị trí có thể và chọn vị trí tốt nhất
            for pos in range(len(assigned) + 1):
                new_route = assigned[:pos] + [i + 1] + assigned[pos:]
                route = [0] + new_route + [0]
                total_time = 0
                for j in range(len(route) - 1):
                    total_time += t[route[j]][route[j + 1]]
                    if route[j + 1] != 0:
                        total_time += d[route[j + 1] - 1]

                if total_time < min_time_local:
                    min_time_local = total_time
                    best_local_route = new_route

            # Chọn nhân viên tốt nhất toàn cục
            if min_time_local < best_time:
                best_time = min_time_local
                best_k = idx
                best_route = best_local_route

        # Cập nhật tuyến cho nhân viên được chọn
        _, k, _ = workers[best_k]
        workers[best_k] = (best_time, k, best_route)

    # Bước 3: Chuẩn hóa output
    routes = []
    max_time = 0
    for total_time, _, assigned in workers:
        if not assigned:
            routes.append([])
            continue
        full_route = [0] + assigned + [0]
        actual_time = 0
        for j in range(len(full_route) - 1):
            actual_time += t[full_route[j]][full_route[j + 1]]
            if full_route[j + 1] != 0:
                actual_time += d[full_route[j + 1] - 1]
        max_time = max(max_time, actual_time)
        routes.append(assigned)

    return max_time, routes

def main():
    sys.setrecursionlimit(100000)
    N, K = map(int, input().split())
    d = list(map(int, input().split()))
    t = [list(map(int, input().split())) for _ in range(N + 1)]

    max_time, routes = segerstedt_plus(N, K, d, t)

    print(K)
    for route in routes:
        print(len(route) + 2)
        print("0", *route, "0")

if __name__ == "__main__":
    main()