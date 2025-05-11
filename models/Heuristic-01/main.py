from .pso import particle_swarm_optimization, improved_lower_bound
from .tsp_2otp import tsp_2opt
from models.utils.helper import create_parameters, print_target_value, log_solution

def main(cfg):
    """
    Hàm main để chạy thuật toán Heuristic-01 (PSO + TSP)
    
    Args:
        cfg: Dictionary chứa các tham số cấu hình
        
    Returns:
        schedule: List các tuyến, mỗi tuyến là list các điểm (bao gồm depot ở đầu và cuối)
    """
    # Đọc dữ liệu từ file instance
    N = cfg['N']  # Số khách hàng
    K = cfg['K']  # Số xe
    d_list = cfg['D']  # Thời gian phục vụ
    t_matrix = cfg['C']  # Ma trận thời gian di chuyển
    
    # Thêm thời gian phục vụ tại depot = 0
    d_list.insert(0, 0)
    
    # Tính lower bound
    lower_bound = improved_lower_bound(d_list, t_matrix, K)
    
    # Danh sách khách hàng
    customers = list(range(1, N + 1))
    
    # Chạy thuật toán PSO để phân cụm
    routes, best_fitness = particle_swarm_optimization(
        customers, K, d_list, t_matrix, lower_bound,
        number_of_particles=20,
        number_of_iterations=100
    )
    
    # Tối ưu từng tuyến bằng TSP-2opt
    schedule = []
    for route in routes:
        optimized_route = tsp_2opt(route, d_list, t_matrix) + [0]
        schedule.append(optimized_route)
    # print(schedule)
    return schedule