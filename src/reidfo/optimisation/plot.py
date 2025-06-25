import matplotlib.pyplot as plt
from hyperopt import Trials


# reviewed
def _plot(losses, save_path: str = None, close=False):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, linestyle='-')
    plt.xlabel('Trial')
    plt.ylabel('Loss')
    plt.title('Loss vs. Trials')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    if close:
        plt.close()
    else:
        plt.show()


# reviewed
def plot_loss(trials: Trials, save_path: str = None, close=False):
    """
    Plots the loss values from hyperopt
    :param trials: A hyperopt Trials object
    :param save_path: Plot output path
    :param figsize: Figure size
    :param close: Close the plot
    """
    if len(trials.trials) < 2:
        return
    losses = [trial['result'].get('loss', None) for trial in trials.trials]
    _plot(losses, save_path=save_path, close=close)


# reviewed
def plot_best_loss(trials: Trials, save_path: str = None, close=False):
    """
    Plots the best loss found so far over the hyperopt trials.

    :param trials: A hyperopt Trials object.
    :param save_path: If provided, the plot will be saved to this path.
    :param figsize: Figure size.
    :param close: If True, the plot will be closed after saving/showing.
    """
    if len(trials.trials) < 2:
        return
    losses = [trial['result'].get('loss', 1e5) for trial in trials.trials]
    best_so_far = []
    current_best = float('inf')
    for loss in losses:
        if loss < current_best:
            current_best = loss
        best_so_far.append(current_best)

    _plot(best_so_far, save_path=save_path, close=close)
