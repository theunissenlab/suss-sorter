import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .core import DataNode


def _get_square_dims(n):
    root = np.sqrt(n)
    if root == np.floor(root):
        return int(root), int(root)
    else:
        a = int(np.ceil(root))
        b = (n + a) // a
	# ugly becuase i cant do math
        if (min(a, b) - 1) * max(a, b) >= n:
            return min(a, b) - 1, max(a, b)
        else:
            return a, b


def draw_on(ax_or_fig):
    if isinstance(ax_or_fig, matplotlib.axes.Axes):
        _dummy = ax_or_fig.twinx()
        _dummy.axis("off")
        _dummy.patch.set_alpha(0.0)
        _ax = _dummy.twiny()
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
    elif isinstance(ax_or_fig, matplotlib.figure.Figure):
        _ax = ax_or_fig.add_axes(
            [0, 0, 1, 1],
            xlim=(0, 1),
            ylim=(0, 1)
        )

    _ax.patch.set_alpha(0.0)
    _ax.axis("off")

    return _ax


def write(ax_or_fig, x, y, text, **kwargs):
    _ax = draw_on(ax_or_fig)
    _ax.text(x, y, text, **kwargs)

    return _ax


def waveforms(
        node,
        fig=None,
        color=None,
        alpha=0.1,
        width=2,
        height=2,
        ylim=(-250, 100),
        median_color="#dd22dd",
        median_linewidth=2,
        axis=True,
        quick=False,
    ):

    cols, rows = _get_square_dims(len(node.children))
    if fig is None:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * width, rows * height), frameon=False)
    else:
        axes = fig.subplots(rows, cols, figsize=(cols * width, rows * height), frameon=False)

    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for child, ax in zip(node.children, axes.flatten()):
        if quick:
            ax.plot(
                child.flatten().waveforms[::int(len(child.flatten().waveforms) / 100) + 1].T,
                color=color,
                alpha=alpha)
        else:
            ax.plot(child.flatten().waveforms.T, color=color, alpha=alpha)
        ax.plot(child.waveform, color=median_color, linewidth=median_linewidth)
        ax.set_ylim(*ylim)

    return fig, axes


def animate_2d(
        node,
        projection_fn,
        timestep=60.0,
        figsize=(4, 4),
        s=20,
        max_frames=None,
        n_lags: "How many trailing steps to display" = 1,
        alpha: "Set opacity of trailing timesteps" = (1.0,),
        xlim=None,
        ylim=None,
        show_time=True,
        show_waveforms=True,
        waveforms_ylim=(-250, 100),
        waveforms_alpha=0.05,
        interval: "interval between frames in milliseconds" = 100,
        save_gif: "save animation as a gif" = False,
        save_gif_filename=None,
        save_gif_dpi=100
    ):
    """Generate animated 2d scatter plot with clusters labeled by color
    
    Node should be either a hierarchical node where each child is one cluster,
    or a leaf node (flattened)
    """
    if n_lags > 1 and len(alpha) != n_lags:
        raise ValueError("Length of alphas must match n_lags")

    node = node.flatten(label=True)

    windows = list(node.windows(dt=timestep))
    unique_labels = np.unique(node.labels)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1], xlim=xlim, ylim=ylim)

    # draw axes just for reference
    ax.vlines(0, *ax.get_ylim(), linewidth=2, linestyle="dotted", alpha=0.3, color="#222222")
    ax.hlines(0, *ax.get_xlim(), linewidth=2, linestyle="dotted", alpha=0.3, color="#222222")
    ax.axis("off")

    colors = {}
    scatter_plots = []
    for lag in range(n_lags):
        scatter_plots.append({})
        for label in unique_labels:
            scat = ax.scatter([], [], s=s, alpha=alpha[lag])
            scatter_plots[-1][label] = scat
            colors[label] = scat.get_edgecolor()[0]

    if show_time:
        time_template = "t = {:.1f} min"
        _text_ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
        _text_ax.patch.set_alpha(0.0)
        time_label = _text_ax.text(0.7, 0.9, time_template.format(0), fontsize=12)

    if show_waveforms:
        wf_ax = fig.add_axes([0.7, 0, 0.3, 0.2], ylim=waveforms_ylim)
        wf_ax.patch.set_alpha(0.0)
        wf_ax.axis("off")

    def draw(frame):
        wf_ax.clear()

        for label in unique_labels:
            for lag, frame_idx in zip(range(n_lags), range(frame, max(frame - n_lags, -1), -1)):
                window, (t_start, t_stop) = windows[frame_idx]
                if label not in window.labels:
                    scatter_plots[lag][label].set_offsets(np.zeros((0, 2)))
                else:
                    scatter_plots[lag][label].set_offsets(
                            projection_fn(window.waveforms[window.labels == label])
                    )
                    if lag == 0 and show_waveforms:
                        wf_ax.plot(
                                window.waveforms[window.labels == label].T,
                                c=colors[label],
                                alpha=waveforms_alpha
                        )
                        wf_ax.set_ylim(*waveforms_ylim)
                        wf_ax.axis("off")

        if show_time:
            time_label.set_text(time_template.format(windows[frame_idx][1][0] / 60.0))

        return [scatters.values() for scatters in scatter_plots]

    anim = animation.FuncAnimation(
            fig,
            draw,
            frames=max_frames or len(windows),
            interval=interval)

    if save_gif:
        if not save_gif_filename:
            print(
                    "Not saving gif because save_gif_filename not specified. "
                    "Save using anim.save(filename, dpi=dpi, writer='imagemagick')"
            )
        anim.save(save_gif_filename, dpi=save_gif_dpi, writer="imagemagick")
        print("Saved gif at {}".format(save_gif_filename))

    plt.close(fig)

    return anim


def time_vs_1d(
        nodes,
        background_node,
        colors=None,
        alpha=1.0,
        s=20,
        background_color="Gray",
        background_alpha=0.1,
        background_s=10,
        projections=None,
        fig=None,
        figsize=(10, 2)
    ):
    if isinstance(nodes, DataNode):
        nodes = [nodes]

    if isinstance(colors, str):
        colors = [colors]

    full_node = DataNode(children=np.concatenate([nodes, [background_node]])).flatten(label=True)

    if projections is None and len(nodes) == 1:
        lda = (
            LinearDiscriminantAnalysis(n_components=2)
            .fit(full_node.waveforms, full_node.labels)
        )
        projections = [lambda data: lda.transform(data)]
    elif projections is None:
        projections = 1

    if isinstance(projections, int):
        pca = PCA(n_components=projections).fit(full_node.flatten().waveforms)
        projections = [
            (lambda d: lambda data: pca.transform(data)[:, d])(dim)
            for dim in np.arange(projections)
        ]

    fig = fig if fig is not None else plt.figure(figsize=figsize)

    axes = []
    for idx, proj in enumerate(projections):
        ax = fig.add_axes(
            [0, idx / len(projections), 1, 1 / len(projections)]
        )
        ax.axis("off")

        ax.scatter(
            background_node.times,
            proj(background_node.waveforms),
            s=background_s,
            color=background_color,
            alpha=background_alpha
        )

        for node_idx, node in enumerate(nodes):
            scat = ax.scatter(
                node.times,
                proj(node.waveforms),
                s=s,
                color=colors[node_idx] if colors else None,
                alpha=alpha
            )

        axes.append(ax)

    return fig, axes
