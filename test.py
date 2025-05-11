# from models.MILPS.project import *
from models.heuristic.main import *
from models.utils.helper import *

cfg = create_parameters('./instance/case1.txt', './output/case1.txt')
schedule = main(cfg)
print_target_value(cfg, './instance/case1.txt', ofcontest=True)
log_solution(schedule, './output/case04.txt')
print_target_value(cfg, './output/case04.txt')