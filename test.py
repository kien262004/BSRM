from models.MILPS.project import *
# from models.heuristic.main import *
from models.utils.helper import *

cfg = create_parameters('./instance/case5.txt', './output/case5.txt')
schedule = main(cfg)
print_target_value(cfg, './instance/case5.txt', ofcontest=True)
log_solution(schedule, './output/milp_case05.txt')
print_target_value(cfg, './output/milp_case05.txt')