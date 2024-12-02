import numpy as np

def randomSerialDictatorship(n, m, rounds):
        #n * n, array, agents x bundles
        allocations = np.zeros((n, n))
        envy = np.zeros((n, rounds))

        first_item_values = np.zeros(rounds)
        second_item_values = np.zeros(rounds)

        for round in range(rounds):
            #n agents, m items, n * m array
            # valuations = np.random.rand(n, m)
            # valuations = np.random.uniform(0, 1, (n, m))

            # valuations from normal distribution
            valuations = np.random.normal(0, 1, (n, m))

            selection_order = np.random.permutation(n)

            # what's allocated
            availableItemsMask = np.ones(m, dtype=bool)


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

                for agent in range(n):
                    allocations[agent][bundle] += valuations[agent][item]

            for agent in range(n):
                A_mask = np.ma.masked_array(allocations[agent], mask=False)
                A_mask.mask[agent] = True
                max_excluding_k = A_mask.max()
                envy[agent][round] = max_excluding_k - allocations[agent][agent]

        for i in range(n):
            print('agent', i, 'value:', allocations[i][i] / (rounds))

        return envy

for j in range(1, 9):
    print("Number of agents:", j)
    print(randomSerialDictatorship(j, j, 10**4))
    print()