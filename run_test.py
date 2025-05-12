import os
import importlib
import argparse
import matplotlib.pyplot as plt
import numpy as np
from models.utils.helper import create_parameters, print_target_value, log_solution

def get_available_algorithms():
    """Lấy danh sách các thuật toán có sẵn trong thư mục models"""
    algorithms = []
    for item in os.listdir('models'):
        if os.path.isdir(os.path.join('models', item)) and not item.startswith('__'):
            # Bỏ qua thuật toán MILPS và thư mục utils
            if item not in ['MILPS', 'utils']:
                algorithms.append(item)
    return algorithms

def get_available_instances():
    """Lấy danh sách các file instance có sẵn"""
    instances = []
    # Thêm các instance từ thư mục custom
    custom_dir = os.path.join('instance', 'custom')
    if os.path.exists(custom_dir):
        for f in os.listdir(custom_dir):
            if f.endswith('.txt'):
                instances.append(os.path.join('custom', f))
    
    # Thêm các instance từ thư mục hustack
    hustack_dir = os.path.join('instance', 'hustack')
    if os.path.exists(hustack_dir):
        for f in os.listdir(hustack_dir):
            if f.endswith('.txt'):
                instances.append(os.path.join('hustack', f))
    
    return instances

def run_algorithm(algorithm_name, instance_file):
    """Chạy thuật toán được chọn với file instance được chọn"""
    # Tạo tên file output
    output_file = f'./output/{algorithm_name}_{os.path.basename(instance_file)}'
    
    # Import module tương ứng
    try:
        module = importlib.import_module(f'models.{algorithm_name}.main')
        main_func = getattr(module, 'main')
    except (ImportError, AttributeError) as e:
        print(f"Lỗi: Không thể import thuật toán {algorithm_name}")
        print(f"Chi tiết lỗi: {str(e)}")
        return None
    
    # Tạo parameters và chạy thuật toán
    cfg = create_parameters(f'./instance/{instance_file}', output_file)
    schedule = main_func(cfg)
    
    # Log kết quả
    log_solution(schedule, output_file)
    
    # In kết quả
    print(f"\nKết quả cho thuật toán {algorithm_name} với instance {instance_file}:")
    result = print_target_value(cfg, output_file)
    
    return result

def run_all_instances(algorithm_name):
    """Chạy thuật toán với tất cả các instance có sẵn"""
    instances = get_available_instances()
    if not instances:
        print("Không tìm thấy instance nào để test")
        return
    
    print(f"\nChạy thuật toán {algorithm_name} với {len(instances)} instance:")
    results = {}
    for instance in instances:
        print(f"\n{'='*50}")
        print(f"Instance: {instance}")
        result = run_algorithm(algorithm_name, instance)
        if result is not None:
            results[instance] = result
    
    return results

def run_all_algorithms(instances):
    """Chạy tất cả các thuật toán với các instance được chọn"""
    algorithms = get_available_algorithms()
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*50}")
        print(f"Chạy thuật toán: {algo}")
        algo_results = {}
        for instance in instances:
            print(f"\nInstance: {instance}")
            result = run_algorithm(algo, instance)
            if result is not None:
                algo_results[instance] = result
        results[algo] = algo_results
    
    return results

def plot_results(results, instances):
    """Vẽ biểu đồ so sánh kết quả của các thuật toán"""
    algorithms = list(results.keys())
    
    if len(instances) == 1:
        # Vẽ biểu đồ cho một instance
        instance = instances[0]
        values = [results[algo].get(instance, 0) for algo in algorithms]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(algorithms, values)
        
        # Thêm giá trị lên đầu mỗi cột
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.xlabel('Thuật toán')
        plt.ylabel('Makespan')
        plt.title(f'So sánh kết quả các thuật toán trên instance {os.path.basename(instance)}')
        plt.xticks(rotation=45)
        
        # Thêm lưới để dễ đọc
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Điều chỉnh layout để không bị cắt
        plt.tight_layout()
        
    else:
        # Vẽ biểu đồ cho nhiều instance
        x = np.arange(len(instances))
        width = 0.8 / len(algorithms)
        
        plt.figure(figsize=(15, 8))
        
        # Vẽ cột cho từng thuật toán
        for i, algo in enumerate(algorithms):
            values = [results[algo].get(instance, 0) for instance in instances]
            plt.bar(x + i*width - 0.4 + width/2, values, width, label=algo)
        
        # Cấu hình biểu đồ
        plt.xlabel('Instance')
        plt.ylabel('Makespan')
        plt.title('So sánh kết quả của các thuật toán')
        plt.xticks(x, [os.path.basename(instance) for instance in instances], rotation=45)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig('comparison_results.png')
    print(f"\nĐã lưu biểu đồ so sánh vào file 'comparison_results.png'")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Chạy test tự động cho các thuật toán')
    parser.add_argument('--algorithm', '-a', help='Tên thuật toán muốn chạy')
    parser.add_argument('--instance', '-i', help='Tên file instance muốn test (có thể là nhiều instance, phân cách bằng dấu phẩy)')
    parser.add_argument('--list', '-l', action='store_true', help='Liệt kê các thuật toán và instance có sẵn')
    parser.add_argument('--all', action='store_true', help='Chạy tất cả các instance cho thuật toán được chọn')
    parser.add_argument('--compare', '-c', action='store_true', help='Chạy tất cả các thuật toán và so sánh kết quả')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nCác thuật toán có sẵn:")
        for algo in get_available_algorithms():
            print(f"- {algo}")
        
        print("\nCác instance có sẵn:")
        for instance in get_available_instances():
            print(f"- {instance}")
        return
    
    if args.compare:
        if args.instance:
            # Chạy so sánh với các instance được chọn
            instances = [i.strip() for i in args.instance.split(',')]
        else:
            # Chạy so sánh với tất cả các instance
            instances = get_available_instances()
        
        results = run_all_algorithms(instances)
        plot_results(results, instances)
        return
    
    if not args.algorithm:
        print("Vui lòng cung cấp tên thuật toán")
        print("Sử dụng --list để xem danh sách các thuật toán có sẵn")
        return
    
    if args.algorithm not in get_available_algorithms():
        print(f"Thuật toán {args.algorithm} không tồn tại")
        print("Sử dụng --list để xem danh sách các thuật toán có sẵn")
        return
    
    if args.all:
        run_all_instances(args.algorithm)
        return
    
    if args.instance:
        # Chạy với một hoặc nhiều instance được chọn
        instances = [i.strip() for i in args.instance.split(',')]
        for instance in instances:
            if instance not in get_available_instances():
                print(f"File instance {instance} không tồn tại")
                continue
            run_algorithm(args.algorithm, instance)
        return
    
    print("Vui lòng cung cấp tên file instance hoặc sử dụng --all để chạy tất cả các instance")

if __name__ == "__main__":
    main() 