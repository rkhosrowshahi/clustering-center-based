import numpy as np
import matplotlib.pyplot as plt
# from cec2017.functions import *
from cec2017.functions import all_functions
from os.path import exists
from CCDE import CCDE


class CCDE_cec2017:
    def __init__(self, fnum=1, dimensions=100, default_bounds=(-100, 100), max_FES=3e+05, num_runs=31,
                 popsize=100, mutation_rate=0.8, crossover_rate=0.9, NC=None, strategy='rand1bin',
                 seed=None, save_file_name=None):
        # problem = GET_FUNCTION.get(str(fnum))
        problem = all_functions[fnum - 1]
        bounds = [default_bounds] * dimensions
        fstar = fnum * 100
        self.num_runs = num_runs
        self.algo_name = "CCDE"
        self.NC = NC
        self.NC_text = f"_NC{NC}"
        if NC is None or NC == 0:
            self.algo_name = "DE"
            self.NC_text = ""
        self.model = CCDE(fobj=problem, bounds=bounds, strategy=strategy,
                          mutation=mutation_rate, crossover=crossover_rate,
                          NC=NC, maxfes=max_FES, popsize=popsize, seed=seed, fstar=fstar)
        self.outputs = None
        self.save_file_name = save_file_name
        if save_file_name is None:
            self.save_file_name = f"results/{self.algo_name}_f{fnum}_{dimensions}D_MR{mutation_rate}_CR{crossover_rate}_{strategy}{self.NC_text}.npz"
        self.popsize = popsize
        self.max_FES = max_FES

    def run(self, continue_to_last_run=True, show_progressbar=True):
        best_candidate, all_best_fitness, all_mean_fitness = [], [], []
        last_run = 0
        if exists(self.save_file_name) and continue_to_last_run:
            loadnpz = np.load(self.save_file_name)
            best_candidate, all_best_fitness= loadnpz["best_candidate"], loadnpz["all_best_fitness"]
            all_mean_fitness = loadnpz["all_mean_fitness"]
            last_run = len(best_candidate)
            print(f"The last saved run was {last_run} and the experiment will continue from the last...")
        for run in range(last_run, self.num_runs):
            self.outputs = self.model.solve(show_progress=True)

            print(f"{self.algo_name}, Run {run + 1}, Best Error: {self.outputs[1]:E}")
            if run != 0:
                best_candidate = np.concatenate([best_candidate, self.outputs[0].reshape(1, -1)], axis=0)
                all_best_fitness = np.concatenate([all_best_fitness, self.outputs[2].reshape(1, -1)], axis=0)
                all_mean_fitness = np.concatenate([all_mean_fitness, self.outputs[3].reshape(1, -1)], axis=0)
            else:
                best_candidate = self.outputs[0].reshape(1, -1)
                all_best_fitness = self.outputs[2].reshape(1, -1)
                all_mean_fitness = self.outputs[3].reshape(1, -1)
            np.savez(self.save_file_name, best_candidate=best_candidate,
                     all_best_fitness=all_best_fitness, all_mean_fitness=all_mean_fitness)

        best_f = all_best_fitness[:, -1]
        print(f"{self.algo_name} Summary:")
        print("Best:", "{:e}".format(np.amin(best_f)))
        # print("Median:", "{:e}".format(np.median(best_f)))
        # print("Worst:", "{:e}".format(np.amax(best_f)))
        print("Mean:", "{:e}".format(np.mean(best_f)))
        print("Std:", "{:e}".format(np.std(best_f)))
        print("******************************************")

    def plot(self, figsize=(12, 5)):
        loadnpz = np.load(self.save_file_name)
        all_best_fitness = loadnpz["all_best_fitness"]

        intervals = np.arange(self.popsize, self.max_FES + 1, self.popsize)
        fig, ax = plt.subplots(figsize=(12, 5))
        label_text = ""
        if self.NC != 0:
            label_text = f"({self.NC_text}-clusters)"
        ax.plot(intervals, np.mean(all_best_fitness, axis=0), label=f"{self.algo_name}{label_text}")
        ax.set_yscale("log")
        ax.ticklabel_format(axis='x', style='sci', scilimits=[-1, 1], useMathText=True)
        ax.set_xlabel('Function Evaluations (FES)')
        ax.set_ylabel('f(x)-f(x*)')
        ax.set_title('Logarithmic scale')
        ax.grid()
        plt.legend(loc='best')
        plt.show()
