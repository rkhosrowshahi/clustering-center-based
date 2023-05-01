import numpy as np
import pandas as pd
from scipy.stats import ranksums
from CCDE_CEC2017 import CCDE_CEC2017

print("Dimensions size?")
D = int(input())
algos = ['CCDE', 'DE']
df_cols = ['Fn', 'CCDE', 'DE']
df = pd.DataFrame(columns=df_cols)
csv_link = f"tables/D{D}_{algos[0]}_{algos[1]}.csv"
df.to_csv(csv_link)
f_stats = np.zeros((31, 3))  # W/ T/ L
for fi in range(1, 31):
    if fi == 2:
        continue
    print(f"Function F{fi} =>>")
    test1 = CCDE_CEC2017(fnum=fi, dimensions=D, default_bounds=(-100, 100), max_FES=3e+05, num_runs=31,
                         popsize=100, mutation_rate=0.8, crossover_rate=0.9, NC=10)
    test1.run(continue_to_last_run=True)

    loadnpz = np.load(test1.save_file_name)
    test1_all_best_fitness = loadnpz["all_best_fitness"]
    test1_mean = np.mean(test1_all_best_fitness[:, -1])
    # if test1_mean < 0.1e-8:
    #    print("True")
    #    test1_mean=0
    test1_mean = 0 if test1_mean < 1e-8 else test1_mean
    test1_std = np.std(test1_all_best_fitness[:, -1])
    test1_std = 0 if test1_std < 1e-8 else test1_std
    # if test1_std <= 0.0e-8:
    #    test1_std=0
    df_new = pd.DataFrame([
        {df_cols[0]: f"F{fi}",
         df_cols[1]: f"{test1_mean:.4e} ({test1_std:.4e})",
         df_cols[2]: ""}
    ])
    df2 = pd.concat([df, df_new])
    df2.to_csv(csv_link, index=False)

    test2 = CCDE_CEC2017(fnum=fi, dimensions=D, default_bounds=(-100, 100), max_FES=3e+05, num_runs=31,
                         popsize=100, mutation_rate=0.8, crossover_rate=0.9, NC=0)
    test2.run(continue_to_last_run=True)

    loadnpz = np.load(test2.save_file_name)
    test2_all_best_fitness = loadnpz["all_best_fitness"]

    stat_sign = "="
    stat, pval = ranksums(test1_all_best_fitness[:, -1], test2_all_best_fitness[:, -1])
    if pval <= 0.05:
        if stat < 0:
            f_stats[fi - 1, 0] = 1
            stat_sign = "+"
        else:
            f_stats[fi - 1, 2] = 1
            stat_sign = "-"
    else:
        f_stats[fi - 1, 1] = 1

    test1_mean = np.mean(test1_all_best_fitness[:, -1])
    test1_mean = 0 if test1_mean < 1e-8 else test1_mean
    test1_std = np.std(test1_all_best_fitness[:, -1])
    test1_std = 0 if test1_std < 1e-8 else test1_std
    test2_mean = np.mean(test2_all_best_fitness[:, -1])
    test2_mean = 0 if test2_mean < 1e-8 else test2_mean
    test2_std = np.std(test2_all_best_fitness[:, -1])
    test2_std = 0 if test2_std < 1e-8 else test2_std
    df_new = pd.DataFrame([
        {df_cols[0]: f"F{fi}",
         df_cols[1]: f"{test1_mean:.4e} ({test1_std:.4e}) {stat_sign}",
         df_cols[2]: f"{test2_mean:.4e} ({test2_std:.4e})"}
    ])
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(csv_link, index=False)

    print(f"----------------------------------------------")

f_stats_sum = np.sum(f_stats, axis=0).astype(int)
# print(f_stats, f_stats_sum)
df_new = pd.DataFrame([
    {df_cols[0]: f"w/t/l",
     df_cols[1]: f"{f_stats_sum[0]}/{f_stats_sum[1]}/{f_stats_sum[2]}",
     df_cols[2]: f""}
])
df = pd.concat([df, df_new], ignore_index=True)
df.to_csv(csv_link, index=False)
