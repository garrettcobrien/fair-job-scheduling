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
        allocations = np.zeros((self.n, self.n))
        envy = np.zeros((self.n, self.rounds))

        for round in range(self.rounds):
            valuations = np.random.rand(self.n, self.m)
            row_ind, col_ind = linear_sum_assignment(-valuations)
            matching = list(zip(row_ind, col_ind))

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
        current_processing = np.zeros(self.n)

        for round in range(self.rounds):
            current_processing = np.maximum(current_processing - 1, 0)
            valuations = np.random.rand(self.n, self.m)
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

# Create multiple graphs with varying number of items
agents = 7
rounds = 1000

# Set up the figure and axes for subplots
fig, axes = plt.subplots(3, agents, figsize=(18, 10), sharex=True, sharey=True)

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
        
        # Use the visualization function to plot on the given axis
        ax = axes[row, col]
        plot_envy(envy, ax=ax, label_prefix=f'Items {items}')
    
    # Set a title for each row
    axes[row, 0].set_ylabel('Envy')
    axes[row, 0].set_title(title, fontsize=12, pad=20)

# Set common labels
for ax in axes[-1, :]:
    ax.set_xlabel('Round')

# Create a single legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', title='Agents')

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
plt.show()