import audplot
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()  # get prettier plots

truth = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
prediction = ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'C']
label_aliases = {'A': 'c1', 'B': 'c2', 'C': 'c3'}

plt.figure(figsize=[2.8, 2.5])
plt.title('Confusion Matrix')
audplot.confusion_matrix(truth, prediction, label_aliases=label_aliases)

plt.tight_layout()