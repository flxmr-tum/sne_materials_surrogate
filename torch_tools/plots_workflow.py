import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

def figure_ia(plotfunc):
    def wrap_figure(*args, **kwargs):
        if "use_fig" not in kwargs.keys():
            fig = plt.figure()
            res = plotfunc(*args, **kwargs, **{"use_fig": fig})
            plt.tight_layout()
            plt.show()
            return res
        elif isinstance(kwargs["use_fig"], str):
            kwargs = kwargs.copy()
            path = kwargs.pop("use_fig")
            fig = mpl.figure.Figure()
            res = plotfunc(*args, **kwargs, **{"use_fig": fig})
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(path)
            return res
        else:
            return plotfunc(*args, **kwargs)
    return wrap_figure


@figure_ia
def plot_prediction_error(
        test : tuple, train=None,
        y_label="", title="",
        use_fig=None, rounding=4
):
    fig = use_fig
    fig.suptitle(title)
    gs = GridSpec(2, 2, width_ratios=(2, 1), height_ratios=(3,1))
    vs_plot = fig.add_subplot(gs[:-1, :], adjustable="box", aspect=1)
    hist_plot = fig.add_subplot(gs[-1, 0])
    data_plot = fig.add_subplot(gs[-1, -1])
    #data_plot.get_xaxis().set_visible(False)
    #data_plot.get_yaxis().set_visible(False)
    data_plot.axis('off')

    if train:
        vs_plot.scatter(train[0], train[1], label="train data")
    vs_plot.scatter(test[0], test[1], label="test data")
    if train:
        vs_plot.legend()
    data_range = np.min(test[0])-0.5, np.max(test[0])-0.5, 
    vs_plot.set_xlim(data_range)
    vs_plot.set_ylim(data_range)
    vs_plot.plot(data_range, data_range)
    vs_plot.set_xlabel(f"{y_label}, true")
    vs_plot.set_ylabel(f"{y_label}, pred")

    dtest = test[1]-test[0]
    vartest = np.var(dtest)
    hist_range = (-3*vartest, 3*vartest)
    hist_plot.set_xlim(hist_range)
    dtrain = []
    if train:
        dtrain = train[1]-train[0]
    hist_plot.hist((dtest, dtrain), range=hist_range, histtype="barstacked")
    hist_plot.set_xlabel(r'$\Delta${}'.format(y_label))

    metrics = []
    maes = np.mean(np.abs(dtest)), np.mean(np.abs(dtrain))
    mses = mean_squared_error(test[0], test[1]), mean_squared_error(train[0], train[1]) if train else None
    r2 = r2_score(test[0], test[1]), r2_score(train[0], train[1]) if train else None
    for label, calc in zip(["MAE", "MSE", "R²"], [maes, mses, r2]):
        metrics.append([label, "{}/{}".format(round(calc[0], rounding), round(calc[1], rounding) if calc[1] else "-")])

    data_plot.table(metrics, loc="center", fontsize=24)
    return {"mae_te" : maes[0], "mae_tr" : maes[1],
            "mse_te" : mses[0], "mse_tr" : mses[1],
            "r2_te" : r2[0], "r2_tr" : r2[1],}


@figure_ia
def plot_ae_latent(
        latent_activations,
        generative_factors,
        overlay_column=None,
        generative_names=None,
        average=False,
        use_fig=None):
    fig = use_fig
    total_datapoints = latent_activations.shape[0]
    if generative_factors.shape[0] != total_datapoints:
        raise Exception
    no_neurons = latent_activations.shape[1]
    no_gen = generative_factors.shape[1]

    use_fig.set_size_inches(8*no_neurons, 8*no_gen)

    if isinstance(generative_factors, (pd.DataFrame, pd.Series)):
        generative_names = generative_factors.columns
        generative_factors = np.array(generative_factors)
    elif generative_names:
        pass
    else:
        generative_names = [f"genfac-{i}" for i in range(no_gen)]

    label_spec = {"fontsize" : "xx-large"}
    gs = GridSpec(1+no_gen, 1+no_neurons, width_ratios=[0.1]+[1,]*no_neurons, height_ratios=[0.1]+[1,]*no_gen)
    for genfac in range(no_gen):
        genfac_label = fig.add_subplot(gs[genfac+1, 0])
        genfac_label.axis('off')
        genfac_label.text(0, 0, generative_names[genfac], **label_spec)
        for neuronidx in range(no_neurons):
            if genfac == 0:
                neuronidx_label = fig.add_subplot(gs[0, neuronidx+1])
                neuronidx_label.axis('off')
                neuronidx_label.text(0, 0, f"neuron {neuronidx}", **label_spec)
            ax = fig.add_subplot(gs[genfac+1, neuronidx+1])
            if not average:
                if overlay_column is not None:
                    ax.scatter(generative_factors[:, genfac], latent_activations[:, neuronidx], c=overlay_column)
                else:
                    ax.scatter(generative_factors[:, genfac], latent_activations[:, neuronidx])
            else:
                # TODO overlay
                genfac_data = generative_factors[:, genfac]
                activation_data = latent_activations[:, neuronidx]
                if not isinstance(np.min(genfac_data), float):
                    print(f"Can't plot average for {genfac_label}")
                    continue
                genfac_plot = np.linspace(np.min(genfac_data), np.max(genfac_data), 50)
                activation_plot = []
                for start, end in zip(genfac_plot[:-1], genfac_plot[1:]):
                    activation_plot.append(np.mean(
                        activation_data[np.logical_and(genfac_data > start, genfac_data <= end)]
                    ))#
                activation_plot = np.convolve(np.array(activation_plot), np.ones(2), mode='same')
                ax.plot(genfac_plot[1:], activation_plot)
                ax.set_ylim(-3, 3)

    plt.tight_layout()


@figure_ia
def plot_fingerprints(
        original,
        reconstructed,
        use_fig=None):
    if isinstance(original, np.ndarray) and isinstance(reconstructed, np.ndarray):
        original = [original,]
        reconstructed = [reconstructed,]
    elif isinstance(original, list) and (len(original)==len(reconstructed)):
        pass
    else:
        raise Exception

    tot_plots = len(original)
    square_plots = np.sqrt(tot_plots)
    n_horizontal = int(2*np.floor(square_plots))
    n_vertical = int(np.ceil(tot_plots/n_horizontal))

    use_fig.set_size_inches(8*n_horizontal, 6*n_vertical)
    
    gs = GridSpec(n_vertical, n_horizontal)

    for idx in range(len(original)):
        ax = use_fig.add_subplot(gs[idx])
        x = np.arange(len(original[idx]))
        ax.plot(x, original[idx], linestyle='--', lw=0.5, label="original")
        ax.plot(x, reconstructed[idx], linestyle='--', lw=0.5, label="recon")
        ax.legend()
