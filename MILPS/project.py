from ortools.linear_solver import pywraplp

def create_parameters(filename, output):
    cfg = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    N, K = map(int, lines[1].strip().split())
    D = list(map(int, lines[2].strip().split()))
    C = []
    for line in lines[3:N+4]:
        C.append(list(map(int, line.strip().split())))
    cfg['N'] = N
    cfg['K'] = K
    cfg['D'] = D
    cfg['C'] = C
    cfg['MAX'] = 10000
    cfg['output'] = output
    
    employees = int(lines[N+5].strip())
    best = 0
    for i in range(employees):
        trip = list(map(int, lines[N+5+i*2+2].strip().split()))
        temp = C[trip[0]][trip[1]]
        for j in range(1, len(trip[:-1])):
            temp += C[trip[j]][trip[j+1]] + D[trip[j]-1]
        best = max(best, temp)
    print(best)
    return cfg
    

def create_variables(solver, cfg):
    N, K = cfg['N'], cfg['K']
    infinity = solver.Infinity()
    Z = [[solver.BoolVar(f'z_{v}_{i}') for i in range(N)] for v in range(K)]
    X = [[[solver.IntVar(0, 1 if i != j else 0, f'x_{v}_{i}_{j}') for j in range(N+1)] for i in range(N+1)] for v in range(K)]
    T = [solver.IntVar(0, infinity, f't_{i}') for i in range(N)]
    complete = solver.IntVar(0, infinity, 'T')
    variables = {}
    variables['Z'] = Z
    variables['X'] = X
    variables['T'] = T
    variables['complete'] = complete
    return variables

def create_constraints(solver, cfg, variables):
    N, K = cfg['N'], cfg['K']
    D, C = cfg['D'], cfg['C']
    MAX = cfg['MAX']
    constraints = {}
    infinity = solver.Infinity()
    # employee constraint
    employee_constraints = []
    # 1
    for i in range(N):
        constraint = solver.Constraint(1, 1)
        for v in range(K):
            constraint.SetCoefficient(variables['Z'][v][i], 1)
        employee_constraints.append(constraint)
    
    # 2
    for v in range(K):
        for i in range(N):
            constraint = solver.Constraint(0, 0)
            constraint.SetCoefficient(variables['Z'][v][i], -1)
            for j in range(N+1):
                constraint.SetCoefficient(variables['X'][v][i+1][j], 1)
            employee_constraints.append(constraint)
    
    # 3
    for v in range(K):
        for j in range(N):
            constraint = solver.Constraint(0, 0)
            constraint.SetCoefficient(variables['Z'][v][j], -1)
            for i in range(N+1):
                constraint.SetCoefficient(variables['X'][v][i][j+1], 1)
            employee_constraints.append(constraint)
    
    # 4, 5
    for v in range(K):
        constraint_1 = solver.Constraint(0, 0)
        constraint_2 = solver.Constraint(0, 1)
        for i in range(N):
            constraint_1.SetCoefficient(variables['X'][v][0][i+1], 1)
            constraint_1.SetCoefficient(variables['X'][v][i+1][0], -1)
            constraint_2.SetCoefficient(variables['X'][v][0][i+1], 1)
        employee_constraints.append(constraint_1)
        employee_constraints.append(constraint_2)
    
    # time constraints
    # 6
    time_constraints = []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            constraint = solver.Constraint(-infinity, MAX - D[j] - C[i+1][j+1])
            constraint.SetCoefficient(variables['T'][i], 1)
            constraint.SetCoefficient(variables['T'][j], -1)
            for v in range(K):
                constraint.SetCoefficient(variables['X'][v][i+1][j+1], MAX)
            time_constraints.append(constraint)
    # 7
    for j in range(N):
        constraint = solver.Constraint(-infinity, MAX - C[0][j+1] - D[j])
        constraint.SetCoefficient(variables['T'][j], -1)
        for v in range(K):
            constraint.SetCoefficient(variables['X'][v][0][j+1], MAX)
        time_constraints.append(constraint)

    # 8
    for i in range(N):
        constraint = solver.Constraint(-infinity, MAX - C[i+1][0])
        constraint.SetCoefficient(variables['complete'], -1)
        constraint.SetCoefficient(variables['T'][i], 1)
        for v in range(K):
            constraint.SetCoefficient(variables['X'][v][i+1][0], MAX)
        time_constraints.append(constraint)

    constraints = {}
    constraints['employee_constraints'] = employee_constraints
    constraints['time_constraints'] = time_constraints
    
    return constraints

def create_objective(solver, cfg, variables):
    objective = solver.Objective()
    objective.SetCoefficient(variables['complete'], 1)
    objective.SetMinimization()
    return objective

def output_solution(variable, cfg):
    with open(cfg['output'], 'w') as f:
        f.write(str(cfg['K']) + '\n')
        for v in range(cfg['K']):
            start = 0
            trip = [0]
            stop = False
            while not stop:
                for i in range(cfg['N']+1):
                    if variable['X'][v][start][i].solution_value() == 1:
                        start = i
                        trip.append(start)
                        break
                if start == 0: stop = True
            f.write(str(len(trip))+'\n')
            f.writelines([str(x)+' ' for x in trip])
            f.write('\n')
def main():
    cfg = create_parameters('./instance/case5.txt', './output/case5.txt')        
    solver = pywraplp.Solver.CreateSolver('SAT')
    variables = create_variables(solver, cfg)
    constraints = create_constraints(solver, cfg, variables)
    objective = create_objective(solver, cfg, variables)
    solver.Solve()
    print(objective.Value())
    output_solution(variables, cfg)
main()
    