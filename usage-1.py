import audplot
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()  # get prettier plots

truth = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
prediction = ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'C']
plot_labels = ['c1', 'c2', 'c3']

plt.figure(figsize=[2.8, 2.5])
plt.title('Confusion Matrix')
audplot.confusion_matrix(truth, prediction)

# replace labels
locs, _ = plt.xticks()
plt.xticks(locs, plot_labels)
plt.yticks(locs, plot_labels)

plt.tight_layout()