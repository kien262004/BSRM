from ortools.linear_solver import pywraplp

solver = pywraplp.Solver('SAT')

    

def variables(solver, cfg):
    N, K = cfg['N'], cfg['K']
    infinity = solver.Infinity()
    Z = [[solver.BoolVar(f'z_{v}_{i}' for i in range(N))] for v in range(K)]
    X = [[[solver.BoolVar(f'x_{v}_{i}_{j}') for j in range(N+1)] for i in range(N+1)] for v in range(K)]
    T = [solver.IntVar(0, infinity, f't_{i}') for i in range(N)]
    complete = solver.InVar(0, infinity, f'T')
    variables = {}
    variables['Z'] = Z
    variables['X'] = X
    variables['T'] = T
    variables['complete'] = complete
    return variables
