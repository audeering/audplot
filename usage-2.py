import audplot
import numpy as np
import matplotlib.pyplot as plt


truth = np.random.randn(100)
prediction = np.random.randn(100)

plot_funcs = [
    audplot.distribution,
    audplot.scatter,
    audplot.series,
]
fig, axs = plt.subplots(1, len(plot_funcs), figsize=[12, 3])
plt.suptitle('Multiple plots in one figure')
for plot_func, ax in zip(plot_funcs, axs):
    plot_func(truth, prediction, ax=ax)

plt.tight_layout()