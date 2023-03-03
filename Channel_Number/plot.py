import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate three columns of random data
np.random.seed(123)  # Set random seed for reproducibility
column1 = np.random.normal(loc=5, scale=1, size=4)
column2 = np.random.normal(loc=7, scale=1.5, size=4)
column3 = np.random.normal(loc=6, scale=1.2, size=4)

# Calculate means and standard deviations for each group
means = [np.mean(column1), np.mean(column2), np.mean(column3)]
stds = [np.std(column1, ddof=1), np.std(column2, ddof=1), np.std(column3, ddof=1)]

# Calculate statistical significance between the pairs of groups using two-sample t-tests
p_values = []
for i in range(3):
    for j in range(i+1, 3):
        t_stat, p_value = stats.ttest_ind([column1[i], column2[i], column3[i]],
                                          [column1[j], column2[j], column3[j]])
        p_values.append(p_value)

# Set significance level
alpha = 1

# Plot the data in a bar figure
fig, ax = plt.subplots()
barplot = ax.bar(['Group 1', 'Group 2', 'Group 3'], means, yerr=stds, capsize=10)

# Add significance asterisks and notations for each pair of groups
for i in range(3):
    for j in range(i+1, 3):
        print(i*(3-i//2)+(j-1))
        # if p_values[i*(3-i//2)+(j-1)] < alpha:
        #     ymax = max(means[i], means[j]) + max(stds[i], stds[j])
        #     ypos = [ymax] * 2
        #     xpos = [i, j]
        #     ax.plot(xpos, ypos, lw=1.5, c='k')
        #     ax.plot([i, i], [ymax, ymax-0.1*np.ptp(ax.get_ylim())], c='k', lw=1)
        #     ax.plot([j, j], [ymax, ymax-0.1*np.ptp(ax.get_ylim())], c='k', lw=1)
        #     ax.text(np.mean(xpos), ymax*1.05, '*', ha='center', va='center', fontsize=16)
        #     ax.text(np.mean(xpos), ymax*1.1, f'p={p_values[i*(3-i//2)+(j-1)]:.3f}', ha='center', va='center', fontsize=14)
        #     plt.plot(p_values)
# Add labels and title
ax.set_xlabel('Group')
ax.set_ylabel('Mean')
ax.set_title('Means with Standard Deviation and Significance Asterisks')

# Show the figure
plt.show()
