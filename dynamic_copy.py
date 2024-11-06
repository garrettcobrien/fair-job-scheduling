import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from visualization_copy import plot_envy

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
        # value of the other allocations
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
        allocations = np.zeros((self.n, self.n))
        envy = np.zeros((self.n, self.rounds))

        for round in range(self.rounds):
            valuations = np.random.rand(self.n, self.m)
            if round == 0:
                selection_order = np.arange(self.n)
            else:
                currentEnvy = envy[:, round - 1]
                selection_order = np.argsort(-currentEnvy)

            availableItemsMask = np.ones(self.m, dtype=bool)
            matching = []

            for agent in selection_order:
                agentValues = np.ma.masked_array(valuations[agent], mask=availableItemsMask.__invert__())
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
        return envy

    def pref_with_processing(self):
        allocations = np.zeros((self.n, self.n))
        envy = np.zeros((self.n, self.rounds))
        # wait (processing) time for each agent
        current_processing = np.zeros(self.n)

        for round in range(self.rounds):
            current_processing = np.maximum(current_processing - 1, 0)
            valuations = np.random.rand(self.n, self.m)
            if round == 0:
                selection_order = np.arange(self.n)
            else:
                currentEnvy = envy[:, round - 1]
                selection_order = np.argsort(-currentEnvy)
            # exclude agents that are currently processing
            selection_order = selection_order[current_processing[selection_order] == 0]
            availableItemsMask = np.ones(self.m, dtype=bool)
            matching = []

            for agent in selection_order:
                agentValues = np.ma.masked_array(valuations[agent], mask=availableItemsMask.__invert__())
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
        return envy

# create multiple graphs with varying number of items (small multiples)
agents = 8
rounds = 1000

fig, axes = plt.subplots(3, agents, figsize=(18, 10), sharex=True, sharey=True)

# methods to test and show in the small multiples
methods = [
    ('Maxweight Matchings', Dynamic.maxweightMatchings),
    ('Preferential Choice', Dynamic.preferrential_choice),
    ('Pref with Processing', Dynamic.pref_with_processing)
]

for row, (title, method) in enumerate(methods):
    for col in range(agents):
        items = col + 1
        dyn = Dynamic(rounds=rounds, agents=agents, items=items, processing=4)
        envy = method(dyn)
        
        # plot on given axes
        ax = axes[row, col]
        plot_envy(envy, ax=ax, label_prefix=f'Items {items}')
    
    axes[row, 0].set_ylabel('Envy')
    axes[row, 0].set_title(title, fontsize=12, pad=20)

for ax in axes[-1, :]:
    ax.set_xlabel('Round')

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title='Agents')

plt.tight_layout(rect=[0, 0, 0.85, 1])  
plt.show()