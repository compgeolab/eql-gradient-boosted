import matplotlib.pyplot as plt


def plot_prediction(
    prediction, target, units, figsize=(18, 6), shrink_cbar=0.7, bins=20, show=True
):
    """
    Plot prediction and comparison with target

    Parameters
    ----------
    prediction : xr.DataArray
        xr.DataArray containing the prediction to plot
    target : xr.DataArray
        xr.DataArray containing the target grid (true values of the field)
    units : str
        Units of the predicted and target fields
    """
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=figsize)
    tmp = prediction.plot.pcolormesh(ax=ax1, center=False, add_colorbar=False)
    plt.colorbar(tmp, ax=ax1, shrink=shrink_cbar, label=units)
    ax1.set_aspect("equal")
    ax1.set_title("Prediction")

    difference = target - prediction
    tmp = difference.plot.pcolormesh(
        ax=ax2, center=0, add_colorbar=False, cmap="seismic"
    )
    plt.colorbar(tmp, ax=ax2, shrink=shrink_cbar, label=units)
    ax2.set_aspect("equal")
    ax2.set_title("Difference between target and prediction")
    plt.tight_layout()

    ax3.hist(difference.values.ravel(), bins=bins)
    ax3.set_title("Histogram of prediction errors")

    if show:
        plt.show()
