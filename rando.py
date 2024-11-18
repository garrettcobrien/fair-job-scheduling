import numpy as np

from scipy.optimize import linear_sum_assignment

bundles = [0,0]
preferDif = [0,0]
difAllocs = 0
preferSam = [0,0]
firstGetsPrefItem = 0
    
rounds = 1000000
for i in range(rounds):
    valuations = np.random.rand(2, 2)
    row_ind, col_ind = linear_sum_assignment(-valuations)

    max_weight = valuations[row_ind, col_ind].sum()
    matching = list(zip(row_ind, col_ind))
    #print(matching)
    for i, match in enumerate(matching):
        bundles[i] += valuations[match[0]][match[1]]
        if i == 0:
            firstAlloc = valuations[match[0]][match[1]]
        else:
            secondAlloc = valuations[match[0]][match[1]]

    #both agents get pref item
    if firstAlloc == max(valuations[:, 0]) and secondAlloc == max(valuations[:, 1]):
        preferDif[0] += firstAlloc
        preferDif[1] += secondAlloc
        difAllocs += 1
    else:
        #agent 1 gets pref item and agent 2 doesn't
        if firstAlloc == max(valuations[:, 0]):
            firstGetsPrefItem += 1
        preferSam[0] += firstAlloc
        preferSam[1] += secondAlloc







     

print(bundles[0] / rounds, " ", bundles[1] / rounds)

print(preferDif[0] / difAllocs, " ", preferDif[1] / difAllocs)
print("num of rounds where alloc is different", difAllocs / rounds)

print('num rounds where alloc is same, agent 1 gets preferred item', firstGetsPrefItem / rounds)

sameAllocs = rounds - difAllocs

print(preferSam[0] / sameAllocs, " ", preferSam[1] / sameAllocs)