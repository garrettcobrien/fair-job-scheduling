import numpy as np

from scipy.optimize import linear_sum_assignment

bundles = [0,0]
preferDif = [0,0]
difAllocs = 0

preferSamAndGet= [0,0]
preferSamAndNot = [0,0]

firstGetsPref= 0
secondGetsPref = 0

rounds = 1000000

envy = np.zeros((2, rounds))

for i in range(rounds):
    valuations = np.random.rand(2, 2)
    row_ind, col_ind = linear_sum_assignment(-valuations)

    max_weight = valuations[row_ind, col_ind].sum()
    matching = list(zip(row_ind, col_ind))

    for j, match in enumerate(matching):
        bundles[j] += valuations[match[0]][match[1]]
        if j == 0:
            firstAlloc = valuations[match[0]][match[1]]
        else:
            secondAlloc = valuations[match[0]][match[1]]

    envy[0, i] = max(valuations[1]) - firstAlloc
    envy[1, i] = max(valuations[0]) - secondAlloc

    # both players get what they want
    if np.argmax(valuations[0]) == matching[0][1] and np.argmax(valuations[1]) == matching[1][1]:
        #print('c1')
        preferDif[0] += firstAlloc
        preferDif[1] += secondAlloc
        difAllocs += 1
    else:
        # first player gets pref
        if np.argmax(valuations[0]) == matching[0][1]:
            firstGetsPref += 1
            preferSamAndGet[0] += firstAlloc
            preferSamAndNot[1] += secondAlloc
        # second player gets pref
        else:
            secondGetsPref += 1
            preferSamAndNot[0] += firstAlloc
            preferSamAndGet[1] += secondAlloc


    #both agents get pref item
    '''
    if firstAlloc == max(valuations[:, 0]) and secondAlloc == max(valuations[:, 1]):
        preferDif[0] += firstAlloc
        preferDif[1] += secondAlloc
        difAllocs += 1
    else:
        #agent 1 gets pref item and agent 2 doesn't
        if firstAlloc == max(valuations[:, 0]):
            firstGetsPref += 1
        if secondAlloc == max(valuations[:, 1]):
            secondGetsPref += 1
        preferSam[0] += firstAlloc
        preferSam[1] += secondAlloc
    '''

sameAllocs = rounds - difAllocs

print('avg item value allocated')
print(bundles[0] / rounds, " ", bundles[1] / rounds)

print('avg item value when both get pref item')
print(preferDif[0] / difAllocs, " ", preferDif[1] / difAllocs)

print("num of rounds where alloc is different", difAllocs / rounds)
print('num rounds where alloc is same, agent 1 gets preferred item', firstGetsPref / rounds)
print('num rounds where alloc is same, agent 2 gets preferred item', secondGetsPref / rounds)


print('agent 1')
print(preferSamAndGet[0] / firstGetsPref, " ", preferSamAndNot[0] / secondGetsPref)
print('agent 2')
print(preferSamAndGet[1] / secondGetsPref, " ", preferSamAndNot[1] / firstGetsPref)

print('same allocs')
print(sameAllocs)

avg_envy_agent_0 = np.mean(envy[0])
avg_envy_agent_1 = np.mean(envy[1])

print('Average envy for agent 0:', avg_envy_agent_0)
print('Average envy for agent 1:', avg_envy_agent_1)