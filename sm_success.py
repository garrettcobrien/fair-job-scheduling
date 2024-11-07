import numpy as np
from scipy.optimize import linear_sum_assignment
from sm_success_viz import plot_envy, plot_success, plot_items
from tqdm import tqdm
import matplotlib.pyplot as plt

class Dynamic:
    def __init__(self, rounds, agents, items, processing=1):
        self.rounds = rounds
        self.n = agents
        self.m = items
        self.verbose = False
        self.p = processing
    
    def set_verbose(self, verbose):
        self.verbose = verbose

    def maxweightMatchings(self):
        # n * n, array, agents x bundles 
        allocations = np.zeros((self.n, self.n))
        envy = np.zeros((self.n, self.rounds))

        for round in range(self.rounds):
            #n agents, m bundles, n * m array
            valuations = np.random.rand(self.n, self.m)

            # calculates maximum weight matching 
            row_ind, col_ind = linear_sum_assignment(-valuations)

            max_weight = valuations[row_ind, col_ind].sum()
            matching = list(zip(row_ind, col_ind))

            for match in matching:
                bundle = match[0]
                item = match[1]

                for agent in range(self.n):
                    allocations[agent][bundle] += valuations[agent][item]

            if self.verbose == True:
                print('valuations')
                print(valuations)
                print("\nMaximum weight matching:", matching)
                print("Maximum weight:", max_weight)
                print('allocations')
                print(allocations)

            for agent in range(self.n):
                # create mask to exclude agent's own allocation... to find max envy
                A_mask = np.ma.masked_array(allocations[agent], mask=False)
                A_mask.mask[agent] = True
                max_excluding_k = A_mask.max()
                envy[agent][round] = max_excluding_k - allocations[agent][agent]
        return envy
    
    def preferrential_choice(self):
        #n * n, array, agents x bundles

        allocations = np.zeros((self.n, self.n))
        envy = np.zeros((self.n, self.rounds))


        for round in range(self.rounds):
            #n agents, m items, n * m array
            valuations = np.random.rand(self.n, self.m)

            #first round randomly allocate from 1 - n
            if round == 0:
                selection_order = np.arange(self.n)
            else:
                currentEnvy = envy[:, round - 1]
                selection_order = np.argsort(-currentEnvy)

            # what's allocated
            availableItemsMask = np.ones(self.m, dtype=bool)


            matching = []

            for agent in selection_order:
                agentValues = np.ma.masked_array(valuations[agent], mask=availableItemsMask.__invert__())

                #if all available items have been allocated break
                if agentValues.count() == 0:
                    break

                chosenItem = agentValues.argmax()
                
                availableItemsMask[chosenItem] = False

                matching.append((agent, chosenItem))



            for match in matching:
                bundle = match[0]
                item = match[1]

                for agent in range(self.n):
                    allocations[agent][bundle] += valuations[agent][item]

            for agent in range(self.n):
                A_mask = np.ma.masked_array(allocations[agent], mask=False)
                A_mask.mask[agent] = True
                max_excluding_k = A_mask.max()
                envy[agent][round] = max_excluding_k - allocations[agent][agent]


            if self.verbose == True:
                print('matching', matching)
                print('valuations')
                print(valuations)
                print('allocations')
                print(allocations)
                print('envy')
                print(envy)
        return envy

    # can't except item for next 2 rounds, etc
    def pref_with_processing(self):
        #n * n, array, agents x bundles

        allocations = np.zeros((self.n, self.n))
        envy = np.zeros((self.n, self.rounds))

        current_processing = np.zeros(self.n)

        for round in range(self.rounds):
            current_processing = np.maximum(current_processing - 1, 0)

            #n agents, m items, n * m array
            valuations = np.random.rand(self.n, self.m)

            #first round randomly allocate from 1 - n
            if round == 0:
                selection_order = np.arange(self.n)
            else:
                currentEnvy = envy[:, round - 1]
                selection_order = np.argsort(-currentEnvy)

            selection_order = selection_order[current_processing[selection_order] == 0]
            availableItemsMask = np.ones(self.m, dtype=bool)

            matching = []


            for agent in selection_order:
                agentValues = np.ma.masked_array(valuations[agent], mask=availableItemsMask.__invert__())

                #if all available items have been allocated break
                if agentValues.count() == 0:
                    break

                chosenItem = agentValues.argmax()
                
                availableItemsMask[chosenItem] = False

                matching.append((agent, chosenItem))

                current_processing[agent] = 2



            for match in matching:
                bundle = match[0]
                item = match[1]

                for agent in range(self.n):
                    allocations[agent][bundle] += valuations[agent][item]

            for agent in range(self.n):
                A_mask = np.ma.masked_array(allocations[agent], mask=False)
                A_mask.mask[agent] = True
                max_excluding_k = A_mask.max()
                envy[agent][round] = max_excluding_k - allocations[agent][agent]


            if self.verbose == True:
                print('matching', matching)
                print('valuations')
                print(valuations)
                print('allocations')
                print(allocations)
                print('envy')
                print(envy)
        return envy

# dyn = Dynamic(rounds=1000, agents=9, items=1, processing=4)
# plot_envy(dyn.maxweightMatchings())

# dyn.set_verbose(False)
# plot_envy(dyn.preferrential_choice())

# plot_envy(dyn.pref_with_processing())

class experiment:
    def __init__(self, num_rounds):
        self.num_rounds = num_rounds

    def run_specific_amount_of_rounds(self, agents, items, rounds):
        successful_iterations = 0
        for i in range(self.num_rounds):
            dyn = Dynamic(rounds=rounds, agents=agents, items=items)
            envy = dyn.maxweightMatchings()
            if max(envy[:, -1]) <= 0:
                successful_iterations += 1
        
        if successful_iterations == 0:
            return 0

        return successful_iterations / self.num_rounds

    def run_exp(self, round_cap, agents, items):
        ans = []
        for i in tqdm(range(1, round_cap)):
            success_percent = self.run_specific_amount_of_rounds(agents, items, i)
            ans.append((i, success_percent))
        return ans
    
    # run for 1 item in batch to max agents 
    def run_item_search(self, agents, round_cap):
        all_data = []
        for i in tqdm(range(1, agents + 1)):
            success_percent = self.run_exp(agents=agents, items=i, round_cap=round_cap)
            all_data.append(success_percent)
        return all_data

exp = experiment(100)
num_agents = 15
# collect all plots
all_plots = []
for i in range(1, num_agents + 1):  # Adjust range as needed
    all_plots.append(exp.run_item_search(agents=i, round_cap=30))

# plot all subplots together
num_rows = (num_agents // 6) + 1
num_cols = 6
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))

axes = axes.flatten()

for idx, data in enumerate(all_plots):
    for batch_idx, ans in enumerate(data):
        rounds = [item[0] for item in ans]
        success_percent = [item[1] for item in ans]
        axes[idx].plot(rounds, success_percent, marker='o', label=f'Success % batch size {batch_idx + 1}')
    
    # axes[idx].set_xlabel('Round')
    # axes[idx].set_ylabel('Success Percentage (%)')
    # axes[idx].set_title(f'Success Percentage for {idx + 1} Agents')
    # axes[idx].legend()
    axes[idx].grid(True)

# hide unused subplots
for ax in axes[len(all_plots):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()