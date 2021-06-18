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
