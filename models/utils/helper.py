def create_parameters(filename, output):
    cfg = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    N, K = map(int, lines[0].strip().split())
    D = list(map(int, lines[1].strip().split()))
    C = []
    for line in lines[2:N+3]:
        C.append(list(map(int, line.strip().split())))
    cfg['N'] = N
    cfg['K'] = K
    cfg['D'] = D
    cfg['C'] = C
    cfg['MAX'] = 10000
    cfg['output'] = output
    cfg['T'] = 10
    cfg['gamma'] = 0.5
    cfg['alpha'] = 0.5
    cfg['is_add_time_execute'] = True
    cfg['is_add_depost_time'] = False
    return cfg

def print_target_value(cfg, filename, ofcontest = False):
    N = cfg['N']
    D = cfg['D']
    C = cfg['C']
    start = N+5 if ofcontest else 0
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        if len(lines) <= start:
            print("Warning: Output file has insufficient lines")
            return 0
            
        employees = int(lines[start].strip())
        best = 0
        
        for i in range(employees):
            if start + i*2 + 2 >= len(lines):
                print(f"Warning: Missing data for employee {i+1}")
                continue
                
            try:
                trip = list(map(int, lines[start+i*2+2].strip().split()))
                if len(trip) < 2:
                    print(f"Warning: Trip for employee {i+1} is too short")
                    continue
                    
                temp = C[trip[0]][trip[1]]
                for j in range(1, len(trip) - 1):
                    temp += C[trip[j]][trip[j+1]] + D[trip[j]-1]
                best = max(best, temp)
            except (IndexError, ValueError) as e:
                print(f"Warning: Error processing trip for employee {i+1}: {str(e)}")
                continue
        
        print(best)
        return best
        
    except Exception as e:
        print(f"Error processing output file: {str(e)}")
        return 0

def log_solution(schedule, filename):
    with open(filename, 'w') as f:
        f.write(str(len(schedule)) + '\n')
        for i in range(len(schedule)):
            f.write(str(len(schedule[i])) + '\n')
            f.write(' '.join(list(map(str, schedule[i]))) + '\n')
            
        