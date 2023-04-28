import numpy as np
import pandas as pd
from scipy.stats import ranksums
from CEC2017_CCDE import CCDE_cec2017

print("Dimensions size?")
D = int(input())
algos = ['CCDE', 'DE']
df_cols = ['Fn', 'CCDE', 'DE']
df = pd.DataFrame(columns=df_cols)
csv_link = f"tables/D{D}_{algos[0]}_{algos[1]}.csv"
df.to_csv(csv_link)
for fi in range(1, 30):
    if fi == 2:
        continue
    test1 = CCDE_cec2017(fnum=fi, dimensions=D, default_bounds=(-100, 100), max_FES=3e+05, num_runs=31,
                         popsize=100, mutation_rate=0.8, crossover_rate=0.9, NC=10)
    test1.run(continue_to_last_run=True)

    loadnpz = np.load(test1.save_file_name)
    test1_all_best_fitness = loadnpz["all_best_fitness"]
    df_new = pd.DataFrame([
        {df_cols[0]: f"F{fi}",
         df_cols[1]: f"{np.mean(test1_all_best_fitness[:, -1])} ({np.std(test1_all_best_fitness[:, -1])})",
         df_cols[2]: ""}
    ])
    df2 = pd.concat([df, df_new])
    df2.to_csv(csv_link, index=False)

    test2 = CCDE_cec2017(fnum=fi, dimensions=D, default_bounds=(-100, 100), max_FES=3e+05, num_runs=31,
                         popsize=100, mutation_rate=0.8, crossover_rate=0.9, NC=0)
    test2.run(continue_to_last_run=True)

    loadnpz = np.load(test2.save_file_name)
    test2_all_best_fitness = loadnpz["all_best_fitness"]

    stat_sign = "="
    stat, pval = ranksums(test1_all_best_fitness[:, -1], test2_all_best_fitness[:, -1])
    if pval <= 0.05:
        if stat < 0:
            stat_sign = "+"
        else:
            stat_sign = "-"

    df_new = pd.DataFrame([
        {df_cols[0]: f"F{fi}",
         df_cols[1]: f"{np.mean(test1_all_best_fitness[:, -1]):.4e} ({np.std(test1_all_best_fitness[:, -1]):.4e}) {stat_sign}",
         df_cols[2]: f"{np.mean(test2_all_best_fitness[:, -1]):.4e}"}
    ])
    df2 = pd.concat([df, df_new], ignore_index=True)
    df2.to_csv(csv_link, index=False)
