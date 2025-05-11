from .algo import genetic_algorithm, decode_solution, route_time
from models.utils.helper import create_parameters, print_target_value, log_solution

def main(cfg):
    """
    Hàm main để chạy thuật toán GA
    
    Args:
        cfg: Dictionary chứa các tham số cấu hình
        
    Returns:
        schedule: Dictionary chứa lịch trình các tuyến
    """
    # Đọc dữ liệu từ file instance
    N = cfg['N']  # Số khách hàng
    K = cfg['K']  # Số xe
    d_list = cfg['D']  # Thời gian phục vụ
    t_matrix = cfg['C']  # Ma trận thời gian di chuyển
    
    # Chạy thuật toán GA
    best_sol, best_fit = genetic_algorithm(
        N, K, d_list, t_matrix,
        pop_size=100,
        generations=400,
        crossover_rate=0.8,
        mutation_rate=0.1,
        use_hill_climbing=True,
        hill_climbing_freq=20
    )
    
    # Chuyển đổi kết quả sang định dạng schedule
    routes = decode_solution(best_sol, N, K)
    
    # Thêm depot vào đầu và cuối mỗi tuyến
    schedule = []
    for route in routes:
        full_route = [0] + route + [0]  # Thêm depot (0) vào đầu và cuối
        schedule.append(full_route)
    
    return schedule 