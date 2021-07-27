truth = ['A', 'B', 'B', 'B', 'C', 'C', 'C'] * 1000
prediction = ['A', 'B', 'C', 'C', 'A', 'A', 'C'] * 1000
confusion_matrix(truth, prediction)
