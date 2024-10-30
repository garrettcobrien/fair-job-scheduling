import numpy as np
from scipy.optimize import linear_sum_assignment
from visualization import plot_envy


class Dynamic:
    def __init__(self, rounds, agents, items):
        self.rounds = rounds
        self.n = agents
        self.m = items
        self.verbose = False
    
    def set_verbose(self, verbose):
        self.verbose = verbose


    def maxweightMatchings(self):
        #n * n, array, agents x bundles

        allocations = np.zeros((self.n, self.n))
        envy = np.zeros((self.n, self.rounds))


        for round in range(self.rounds):
            #n agents, m bundles, n * m array
            valuations = np.random.rand(self.n, self.m)

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
                A_mask = np.ma.masked_array(allocations[agent], mask=False)
                A_mask.mask[agent] = True
                max_excluding_k = A_mask.max()
                envy[agent][round] += max_excluding_k - allocations[agent][agent]
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

            availableItemsMask = np.ones(self.m, dtype=bool)


            matching = []

            for agent in selection_order:
                agentValues = np.ma.masked_array(valuations[agent], mask=availableItemsMask.__invert__())
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
                envy[agent][round] += max_excluding_k - allocations[agent][agent]


            if self.verbose == True:
                print('matching', matching)
                print('valuations')
                print(valuations)
                print('allocations')
                print(allocations)
                print('envy')
                print(envy)
        return envy
    
dyn = Dynamic(rounds=40, agents=5, items=5)
#plot_envy(dyn.maxweightMatchings())

dyn.set_verbose(False)
plot_envy(dyn.preferrential_choice())


