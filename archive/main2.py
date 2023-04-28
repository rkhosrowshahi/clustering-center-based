# Using only f5:
import numpy as np
from cec2017.functions import f5

from CCDE import CCDE

samples = 3
dimension = 50
x = np.random.uniform(-100, 100, size=(samples, dimension))
val = f5(x)
for i in range(samples):
    print(f"f5(x_{i}) = {val[i]:.6f}")

# Using all functions:
from cec2017.functions import all_functions
for f in all_functions:
    x = np.random.uniform(-100, 100, size=(samples, dimension))
    val = f(x)
    for i in range(samples):
        print(f"{f.__name__}(x_{i}) = {val[i]:.6f}")


from DE import differential_evolution

# best_sol_end_fit=[]

all_runs_sol_fit=[]
# all_runs_sol=[]

best_sol_generations=[]
def de_callback(xk, convergence):
    global best_sol_generations
    best_sol_generations.append(xk)

    return False
fn=3
D=100
#fun_info = bench.get_info(fn)

from tqdm import tqdm
for rcount in tqdm(range(3)):
    print(f'Run {rcount+1}')
    #l, h = fun_info['lower'], fun_info['upper']
    #bounds = [(l, h)] * fun_info['dimension']
    l, h = -100, 100
    bounds = [(l, h)] * D
    #fun_fitness = bench.get_function(fn)
    fun_fitness = all_functions[fn-1]
    opt = CCDE(fun_fitness, bounds, NC=0, mutation=0.5, maxfes=3e+05) # , disp=True, save_link=f'results/f{fn}_DE_run{rcount}')
     =opt.solve()
    # best_sol, best_fit = result.x, result.fun
    all_best_fs = opt.all_best_fs
    print(f'Best fitness = {best_fit:.6e}')

print(best_sol_generations)