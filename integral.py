import numpy as np

num_samples = 10**8
v1_samples = np.random.uniform(0, 1, num_samples)
v2_samples = np.random.uniform(0, 1, num_samples)
x1_samples = np.random.uniform(0, 1, num_samples)
x2_samples = np.random.uniform(0, 1, num_samples)

# indicator for max weight matching
indicator_samples = (v1_samples + x2_samples >= v2_samples + x1_samples)

own_values = np.zeros(num_samples)
other_values = np.zeros(num_samples)

# assign values on max weight matching condition
own_values[indicator_samples] = v1_samples[indicator_samples]
own_values[~indicator_samples] = v2_samples[~indicator_samples]

other_values[indicator_samples] = v2_samples[indicator_samples]
other_values[~indicator_samples] = v1_samples[~indicator_samples]

# compute probabilities and expectations
p = np.mean(indicator_samples)
expected_own_mc = np.mean(own_values)
expected_other_mc = np.mean(other_values)

print(f"Probability : {p}")
print(f"Expected value of agent's own item: {expected_own_mc}")
print(f"Expected value of agent for the other agent's item: {expected_other_mc}")
