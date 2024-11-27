import numpy as np
from scipy.optimize import linear_sum_assignment
from sm_success_viz import plot_envy, plot_success, plot_items
from tqdm import tqdm
import matplotlib.pyplot as plt
from dynamic import Dynamic

dyn = Dynamic(rounds=1000, agents=2, items=2, processing=1)
plot_envy(dyn.maxweightMatchings())
plot_envy(dyn.randomSerialDictatorship())
plot_envy(dyn.randomSerialDictatorshipRRStyle())

# dyn.set_verbose(False)
# plot_envy(dyn.randomSerialDictatorship())

# plot_envy(dyn.pref_with_processing())

################################## EXPERIMENT ###############################

# class experiment:
#     def __init__(self, num_rounds):
#         self.num_rounds = num_rounds

#     def run_specific_amount_of_rounds(self, agents, items, rounds):
#         successful_iterations = 0
#         for i in range(self.num_rounds):
#             dyn = Dynamic(rounds=rounds, agents=agents, items=items)
#             # envy = dyn.maxweightMatchings()
#             envy = dyn.randomSerialDictatorship()
#             if max(envy[:, -1]) <= 0:
#                 successful_iterations += 1
        
#         if successful_iterations == 0:
#             return 0

#         return successful_iterations / self.num_rounds

#     def run_exp(self, round_cap, agents, items):
#         ans = []
#         for i in tqdm(range(1, round_cap)):
#             success_percent = self.run_specific_amount_of_rounds(agents, items, i)
#             ans.append((i, success_percent))
#         return ans
    
#     # run for 1 item in batch to max agents 
#     def run_item_search(self, agents, round_cap):
#         all_data = []
#         for i in tqdm(range(1, agents + 1)):
#             success_percent = self.run_exp(agents=agents, items=i, round_cap=round_cap)
#             all_data.append(success_percent)
#         return all_data

# exp = experiment(1000)
# num_agents = 2
# # collect all plots
# all_plots = []
# for i in range(1, num_agents + 1):  # Adjust range as needed
#     all_plots.append(exp.run_item_search(agents=i, round_cap=1000))

# # plot all subplots together
# num_rows = (num_agents // 6) + 1
# num_cols = 6
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))

# axes = axes.flatten()

# for idx, data in enumerate(all_plots):
#     for batch_idx, ans in enumerate(data):
#         rounds = [item[0] for item in ans]
#         success_percent = [item[1] for item in ans]
#         axes[idx].plot(rounds, success_percent, marker='o', label=f'Success % batch size {batch_idx + 1}')
    
#     # axes[idx].set_xlabel('Round')
#     # axes[idx].set_ylabel('Success Percentage (%)')
#     # axes[idx].set_title(f'Success Percentage for {idx + 1} Agents')
#     # axes[idx].legend()
#     axes[idx].grid(True)

# # hide unused subplots
# for ax in axes[len(all_plots):]:
#     ax.set_visible(False)

# plt.tight_layout()
# plt.show()