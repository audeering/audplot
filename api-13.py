np.random.seed(0)
truth = np.random.normal(loc=0.0, scale=1.0, size=1000)
prediction = np.random.normal(loc=0.05, scale=0.5, size=1000)
distribution(truth, prediction)
