import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style('darkgrid')

height = [99, 95, 80]

# Choose the names of the bars
bars = ('Random Forest', 'Ada Booster', 'Gradient')
x_pos = np.arange(len(bars))

# Create bars
plt.bar(x_pos, height)

# Create names on the x-axis
plt.xticks(x_pos, bars, color='orange')
plt.yticks(color='orange')

# Show graphic
plt.show()