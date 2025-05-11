from .project import main as solve_milp
from models.utils.helper import create_parameters, print_target_value, log_solution

def main(cfg):
    """
    Hàm main để chạy thuật toán MILP
    
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
    
    # Thêm các tham số cấu hình cần thiết cho MILP
    milp_cfg = {
        'N': N,
        'K': K,
        'D': d_list,
        'C': t_matrix,
        'MAX': 1000000  # Giá trị đủ lớn cho các ràng buộc
    }
    
    # Chạy thuật toán MILP
    routes = solve_milp(milp_cfg)
    
    # Thêm depot vào đầu và cuối mỗi tuyến nếu chưa có
    schedule = []
    for route in routes:
        if route[0] != 0:
            route = [0] + route
        if route[-1] != 0:
            route = route + [0]
        schedule.append(route)
    
    return schedule