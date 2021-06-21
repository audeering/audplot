Usage
=====

All the plotting functions use :mod:`seaborn`
and :mod:`matplotlib` under the hood.

This means you can show or update the figures accordingly
by adding a title.
Or if you want to show different label names
than present in your truth and prediction values,
you can change the labels after plotting.

.. plot::
    :context: reset

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

To show multiple graphs in one figure
you can specify the axes to draw on.

.. plot::
    :context: reset

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
