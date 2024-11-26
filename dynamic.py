import numpy as np
from scipy.optimize import linear_sum_assignment
from visualization import plot_envy, plot_success, plot_items, plot_min_rounds
from tqdm import tqdm
import pandas as pd

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
    
    def randomSerialDictatorship(self):
        # n * n, array, agents x bundles 
        allocations = np.zeros((self.n, self.n))
        envy = np.zeros((self.n, self.rounds))

        for round in range(self.rounds):
            # n agents, m items, n * m array
            valuations = np.random.rand(self.n, self.m)

            # draw a random permutation of agents
            agent_order = np.random.permutation(self.n)

            # track which items have been allocated
            available_items = np.ones(self.m, dtype=bool)

            for agent in agent_order:
                # mask the unavailable items
                agent_values = np.ma.masked_array(valuations[agent], mask=~available_items)

                # if all items are allocated, break
                if agent_values.count() == 0:
                    break

                # choose the item with the highest value
                chosen_item = agent_values.argmax()
                available_items[chosen_item] = False

                # allocate the chosen item to the agent
                allocations[agent][chosen_item] += valuations[agent][chosen_item]

            if self.verbose == True:
                print('valuations')
                print(valuations)
                print("\nAgent order:", agent_order)
                print('allocations')
                print(allocations)

            for agent in range(self.n):
                # create mask to exclude agent's own allocation... to find max envy
                A_mask = np.ma.masked_array(allocations[agent], mask=False)
                A_mask.mask[agent] = True
                max_excluding_k = A_mask.max()
                envy[agent][round] = max_excluding_k - allocations[agent][agent]
        return envy
    
    def randomSerialDictatorshipRRStyle(self):
        # n * n, array, agents x bundles 
        allocations = np.zeros((self.n, self.n))
        envy = np.zeros((self.n, self.rounds))

        # initialize the agent order
        agent_order = np.arange(self.n)

        for round in range(self.rounds):
            # n agents, m items, n * m array
            valuations = np.random.rand(self.n, self.m)

            # rotate the agent order
            agent_order = np.roll(agent_order, -1)

            # track which items have been allocated
            available_items = np.ones(self.m, dtype=bool)

            for agent in agent_order:
                # mask the unavailable items
                agent_values = np.ma.masked_array(valuations[agent], mask=~available_items)

                # if all items are allocated, break
                if agent_values.count() == 0:
                    break

                # choose the item with the highest value
                chosen_item = agent_values.argmax()
                available_items[chosen_item] = False

                # Allocate the chosen item to the agent
                allocations[agent][chosen_item] += valuations[agent][chosen_item]

            if self.verbose:
                print('valuations')
                print(valuations)
                print("\nAgent order:", agent_order)
                print('allocations')
                print(allocations)

            for agent in range(self.n):
                # create mask to exclude agent's own allocation to find max envy
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


class ExperimentRunner:
    def __init__(self, num_rounds):
        self.num_rounds = num_rounds
        self.success_minimum = 0.97

    def run_specific_amount_of_rounds(self, agents, items, rounds):
        successful_iterations = 0
        for i in range(self.num_rounds):
            dyn = Dynamic(rounds=rounds, agents=agents, items=items)
            # envy = dyn.maxweightMatchings()
            envy = dyn.randomSerialDictatorship()
            if max(envy[:, -1]) <= 0:
                successful_iterations += 1

        if successful_iterations == 0:
            return 0

        return successful_iterations / self.num_rounds

    def determine_success_percent(self, agents, items, rounds):
        '''
        Determine if then given amount of rounds results in an EF allocation for n agents, and batch size b
        '''
        maximum_failed_rounds = self.num_rounds - self.num_rounds * self.success_minimum
        successful_iterations = 0
        for i in range(self.num_rounds):
            dyn = Dynamic(rounds=rounds, agents=agents, items=items)
            # envy = dyn.maxweightMatchings()
            envy = dyn.randomSerialDictatorship()
            if max(envy[:, -1]) <= 0:
                successful_iterations += 1
            
            if i - successful_iterations >= maximum_failed_rounds:
                return 0

        if successful_iterations == 0:
            return 0
        
        if successful_iterations / self.num_rounds < self.success_minimum:
            return 0
        return 1

    def run_exp(self, round_cap, agents, items):
        ans = []
        for i in tqdm(range(1, round_cap)):
            success_percent = self.run_specific_amount_of_rounds(agents, items, i)
            ans.append((i, success_percent))
        return ans
    
    def run_exp_till_high_prob(self, round_cap, agents, items):
        for round in range(1, round_cap):
            success = self.determine_success_percent(agents, items, round)
            if success == 1:
                return (items, round)
        return (items, -1)
    
    def run_item_search(self, agents, max_items,round_cap):
        ans = []
        for i in tqdm(range(1, max_items + 1)):
            success_percent = self.run_exp(agents=agents, items=i, round_cap=round_cap)
            ans.append(success_percent)
        return ans

    def round_min_search(self, agents, round_cap):
        ans = []
        for i in tqdm(range(1, agents + 1)):
            high_prob_round = self.run_exp_till_high_prob(agents=agents, items=i, round_cap=round_cap)
            ans.append(high_prob_round)
        return ans

    def run_agents(self, max_agents, round_cap):
        ans = []
        for num_agents in tqdm(range(2, max_agents + 1)):
            round_minimums = self.round_min_search(agents=num_agents, round_cap=round_cap)
            ans.append(round_minimums)
        return ans


exp = ExperimentRunner(100)
#plot_success(exp.run_exp(round_cap=25, agents=5, items=1))
#plot_items(exp.run_item_search(agents=9, max_items=9, round_cap=30))


round_exp = ExperimentRunner(100)
#print(round_exp.round_min_search(agents=9, round_cap=25))
#plot_success(round_exp.round_min_search(agents=9, round_cap=1000))


out = round_exp.run_agents(max_agents=2, round_cap=100)
print(out)
dict_data = []
for row in out:
    row_dict = {tup[0]: tup[1] for tup in row}  # Use the first item as column, second as value
    dict_data.append(row_dict)
pd.DataFrame(dict_data).to_csv('fileout.csv')
print(dict_data)
#plot_min_rounds(out)
