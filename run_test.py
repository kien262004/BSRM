import os
import importlib
import argparse
from models.utils.helper import create_parameters, print_target_value, log_solution

def get_available_algorithms():
    """Lấy danh sách các thuật toán có sẵn trong thư mục models"""
    algorithms = []
    for item in os.listdir('models'):
        if os.path.isdir(os.path.join('models', item)) and not item.startswith('__'):
            algorithms.append(item)
    return algorithms

def get_available_instances():
    """Lấy danh sách các file instance có sẵn"""
    return [f for f in os.listdir('instance') if f.endswith('.txt')]

def run_algorithm(algorithm_name, instance_file):
    """Chạy thuật toán được chọn với file instance được chọn"""
    # Tạo tên file output
    output_file = f'./output/{algorithm_name}_{os.path.splitext(instance_file)[0]}.txt'
    
    # Import module tương ứng
    try:
        module = importlib.import_module(f'models.{algorithm_name}.main')
        main_func = getattr(module, 'main')
    except (ImportError, AttributeError) as e:
        print(f"Lỗi: Không thể import thuật toán {algorithm_name}")
        print(f"Chi tiết lỗi: {str(e)}")
        return
    
    # Tạo parameters và chạy thuật toán
    cfg = create_parameters(f'./instance/{instance_file}', output_file)
    schedule = main_func(cfg)
    
    # Log kết quả
    log_solution(schedule, output_file)
    
    # In kết quả
    print(f"\nKết quả cho thuật toán {algorithm_name} với instance {instance_file}:")
    print_target_value(cfg, output_file)
    
    # In kết quả của instance gốc để so sánh
    print(f"\nKết quả của instance gốc:")
    print_target_value(cfg, f'./instance/{instance_file}', ofcontest=True)

def main():
    parser = argparse.ArgumentParser(description='Chạy test tự động cho các thuật toán')
    parser.add_argument('--algorithm', '-a', help='Tên thuật toán muốn chạy')
    parser.add_argument('--instance', '-i', help='Tên file instance muốn test')
    parser.add_argument('--list', '-l', action='store_true', help='Liệt kê các thuật toán và instance có sẵn')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nCác thuật toán có sẵn:")
        for algo in get_available_algorithms():
            print(f"- {algo}")
        
        print("\nCác instance có sẵn:")
        for instance in get_available_instances():
            print(f"- {instance}")
        return
    
    if not args.algorithm or not args.instance:
        print("Vui lòng cung cấp tên thuật toán và file instance")
        print("Sử dụng --list để xem danh sách các thuật toán và instance có sẵn")
        return
    
    if args.algorithm not in get_available_algorithms():
        print(f"Thuật toán {args.algorithm} không tồn tại")
        print("Sử dụng --list để xem danh sách các thuật toán có sẵn")
        return
    
    if args.instance not in get_available_instances():
        print(f"File instance {args.instance} không tồn tại")
        print("Sử dụng --list để xem danh sách các instance có sẵn")
        return
    
    run_algorithm(args.algorithm, args.instance)

if __name__ == "__main__":
    main() 