truth = np.array([1] * 1000 + [0] * 1000)
# Random prediction
pred1 = np.random.random_sample(2000)
# Better than random prediction
pred2 = np.zeros(2000,)
pred2[:1000] = np.random.normal(loc=0.6, scale=0.1, size=1000)
pred2[1000:] = np.random.normal(loc=0.4, scale=0.1, size=1000)
pred2 = np.clip(pred2, 0, 1)
transform = detection_error_tradeoff(
    truth,
    pred1,
    xlim=[0.01, 0.99],  # use large limits for random
    ylim=[0.01, 0.99],
    label='pred1',
)
# Add pred2 to plot using transformed FMR and FNMR values
import audmetric
fmr, fnmr, _ = audmetric.detection_error_tradeoff(truth, pred2)
_ = plt.plot(transform(fmr), transform(fnmr), label='pred2')
_ = plt.legend()
plt.tight_layout()
