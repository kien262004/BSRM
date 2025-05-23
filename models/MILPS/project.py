from ortools.linear_solver import pywraplp

def create_variables(solver, cfg):
    N, K = cfg['N'], cfg['K']
    infinity = solver.Infinity()
    Z = [[solver.BoolVar(f'z_{v}_{i}') for i in range(N+1)] for v in range(K)]
    X = [[[solver.IntVar(0, 1 if i != j else 0, f'x_{v}_{i}_{j}') for j in range(N+1)] for i in range(N+1)] for v in range(K)]
    T = [solver.IntVar(0, infinity, f't_{i}') for i in range(N+1)]
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
    for i in range(1,N+1):
        constraint = solver.Constraint(1, 1)
        for v in range(K):
            constraint.SetCoefficient(variables['Z'][v][i], 1)
        employee_constraints.append(constraint)
    
    # 2
    for v in range(K):
        for i in range(N+1):
            constraint = solver.Constraint(0, 0)
            constraint.SetCoefficient(variables['Z'][v][i], -1)
            for j in range(N+1):
                constraint.SetCoefficient(variables['X'][v][i][j], 1)
            employee_constraints.append(constraint)
    
    # 3
    for v in range(K):
        for j in range(N+1):
            constraint = solver.Constraint(0, 0)
            constraint.SetCoefficient(variables['Z'][v][j], -1)
            for i in range(N+1):
                constraint.SetCoefficient(variables['X'][v][i][j], 1)
            employee_constraints.append(constraint)
    
    # 4
    for v in range(K):
        for i in range(1, N+1):
            constraint = solver.Constraint(-infinity, 0)
            constraint.SetCoefficient(variables['Z'][v][i], 1)
            constraint.SetCoefficient(variables['Z'][v][0], -1)
            employee_constraints.append(constraint)
    
    # time constraints
    # 6
    time_constraints = []
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i == j: continue
            constraint = solver.Constraint(-infinity, MAX - D[j-1] - C[i][j])
            constraint.SetCoefficient(variables['T'][i], 1)
            constraint.SetCoefficient(variables['T'][j], -1)
            for v in range(K):
                constraint.SetCoefficient(variables['X'][v][i][j], MAX)
            time_constraints.append(constraint)
    # 7
    for j in range(1, N+1):
        constraint = solver.Constraint(-infinity, MAX - C[0][j] - D[j-1])
        constraint.SetCoefficient(variables['T'][j], -1)
        constraint.SetCoefficient(variables['T'][0], 1)
        for v in range(K):
            constraint.SetCoefficient(variables['X'][v][0][j], MAX)
        time_constraints.append(constraint)

    constraint = solver.Constraint(0, 0)
    constraint.SetCoefficient(variables['T'][0], 1)
    time_constraints.append(constraint)
    
    # 8
    for i in range(1, N+1):
        constraint = solver.Constraint(-infinity, MAX - C[i][0])
        constraint.SetCoefficient(variables['complete'], -1)
        constraint.SetCoefficient(variables['T'][i], 1)
        for v in range(K):
            constraint.SetCoefficient(variables['X'][v][i][0], MAX)
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
    solution = []
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
        solution.append(trip)
    return solution

def main(cfg):
    solver = pywraplp.Solver.CreateSolver('SAT')
    variables = create_variables(solver, cfg)
    constraints = create_constraints(solver, cfg, variables)
    objective = create_objective(solver, cfg, variables)
    solver.Solve()
    solution = output_solution(variables, cfg)
    return solution
    