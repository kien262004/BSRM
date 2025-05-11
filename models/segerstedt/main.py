from .Segerstedt_Plus import segerstedt_plus
from models.utils.helper import create_parameters, print_target_value, log_solution

def main(cfg):
    """
    Hàm main để chạy thuật toán Segerstedt+
    
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
    
    # Chạy thuật toán Segerstedt+
    max_time, routes = segerstedt_plus(N, K, d_list, t_matrix)
    
    # Thêm depot vào đầu và cuối mỗi tuyến nếu chưa có
    schedule = []
    for route in routes:
        if route[0] != 0:
            route = [0] + route
        if route[-1] != 0:
            route = route + [0]
        schedule.append(route)
    
    return schedule