# optimizer.py
from deap import base, creator, tools, algorithms
import random
import numpy as np

# 创建适应度和个体类（最小化问题）
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def optimize_single_objective(objective_func, n_vars, bounds=(0, 500), n_gen=80, pop_size=50):
    """
    使用遗传算法优化单目标函数
    """
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, bounds[0], bounds[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_func)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_gen, 
                        stats=stats, halloffame=hof, verbose=False)
    
    return np.array(hof[0])