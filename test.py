# from models.MILPS.project import *
from models.heuristic.main import *
from models.utils.helper import *

cfg = create_parameters('./instance/case6.txt', './output/case6.txt')
schedule = main(cfg)
print_target_value(cfg, './instance/case6.txt', ofcontest=True)
log_solution(schedule, './output/heuristic_case_06.txt')
print_target_value(cfg, './output/heuristic_case_06.txt')