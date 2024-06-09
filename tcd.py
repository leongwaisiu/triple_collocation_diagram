"""
Triple Collocation diagram (Siu et al., submitted) implementation.

This is a heavy modification of Yannick Copin's work for implementing
Taylor diagram.  The original code is placed in 
https://gist.github.com/ycopin/3342888.

If you use it for your research, please cite the following publication:
    Siu et al., 2024: Summarizing multiple aspects of triple collocation 
        analysis in a single diagram, submitted.

"""
#!/usr/bin/env python
# Python library imports
# Third party imports
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.floating_axes import FloatingSubplot, GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator, MaxNLocator
# Local imports


class TripleCollocationDiagram:
    """
    Triple collocation diagram.
    
    Plot standard deviation and correlation of three datasets
    in a single-quadrant polar plot, with r=std and theta=arccos(correlation).
    """
    def __init__(
        self,
        errstd,
        corrcoef,
        fig=None,
        rect=111,
        rlocs=None,
        srange=None,
        sgrid_locator=None,
        normal=False,
        extend=False,
        horizon='sigstd',
    ):
        """Set up Triple Collocation diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.

        Parameters
        ----------
        errstd : list or array_like
            List of error standard deviation of datasets (can be more than three).
        corrcoef : list or array_like
            List of correlation coefficient of datasets (can be more than three).
        fig : a matplotlib Figure class, default: None
            Input Figure.
        rect : integer, default: 111
            Subplot definition.
        rlocs : list or array_like, default: None
            Specify correlation grid location.
        srange : integer, default: [0,1.5]
            Std axis extension/range, in units of normalized std.
        sgrid_locator : a matplotlib grid_finder class, default: None
            Specify std grid locations.
        normal : bool, default: False
            If True, normalize total standard deviation to 1.
        extend : bool, default: False
            If True, extend diagram to negative correlations.
            Legacy kwarg, currently not in use.
        horizon : str, default: 'sigstd'
            Specify the variable on the horizon. 
        """
        # Error standard deviation
        if isinstance(errstd, list):
            self.errstd = errstd
        else:
            self.errstd = [*errstd]
        # Correlation coefficient
        if isinstance(corrcoef, list):
            self.corrcoef = corrcoef
        else:
            self.corrcoef = [*corrcoef]
        # Total standard deviation
        self.totstd = [
            np.sqrt(a**2/(1-b**2)) for a, b in zip(self.errstd, self.corrcoef)
        ]
        # Signal standard deviation
        self.sigstd = [
            np.sqrt(b**2-a**2) for a, b in zip(self.errstd, self.totstd)
        ]
        # Specify horizon variable
        if horizon.lower() in {'sigstd','totstd'}:
            self.horizon = horizon
        else:
            raise ValueError(f'Horizon variable must be sigstd/totstd.')
        # Default radial axis range
        if srange is None:
            if normal:
                srange = [0, 1.5]
            else:
                srange = [0, np.amax(self.totstd)*1.25]
        # Set radial axis extent
        self.smin = srange[0]
        self.smax = srange[1]

        # Set up input Figure
        if fig is None:
            fig = plt.figure()
        # Set up base polar transform
        tr = PolarAxes.PolarTransform()
        # Correlation labels
        if rlocs is None:
            rlocs = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
            # rlocs = np.array([0.0,0.3,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0])
        # Diagram extended to negative correlations (not used)
        if extend:
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        # Diagram limited to positive correlations
        else:
            self.tmax = np.pi / 2
        tlocs = np.arccos(rlocs)  # Conversion to polar angles (theta)
        gl1 = FixedLocator(tlocs)  # Positions
        tf1 = DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        # Set up the plot boundary
        ghelper = GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
            grid_locator2=sgrid_locator,
            tick_formatter2=None,
        )

        # Set up Axes
        ax = FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        # Adjust all axes
        ax.axis[:].major_ticklabels.set_pad(1)
        ax.axis[:].major_ticks.set_tick_out(True)
        ax.axis[:].label.set_pad(1)
        # Adjust 'Angle-axis'
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation coefficient")
        # Adjust 'X-axis'
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Signal standard deviation")
        # Adjust 'Y-axis'
        ax.axis["right"].set_axis_direction("top")
        # ax.axis["right"].label.set_text("Error standard deviation")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left"
        )
        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)  # Unused

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates
        self.samplePoints = []  # Initialize sample points

    def add_totstd(self, ind, label=None, normal=False, *args, **kwargs):
        """Add total standard deviation to TC diagram.
        *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        # (theta, radius)
        if normal:
            if self.horizon == 'sigstd':
                azimuth = np.arccos(self.corrcoef[ind])
                radius = self.totstd[ind]/self.errstd[ind]      
            elif self.horizon == 'totstd':
                azimuth = np.arccos(1)
                radius = [1]
        else:
            if self.horizon == 'sigstd':
                azimuth = np.arccos(self.corrcoef[ind])
                radius = self.totstd[ind]      
            elif self.horizon == 'totstd':
                azimuth = np.arccos(1)
                radius = self.totstd[ind]      

        (l,) = self.ax.plot(
            azimuth, radius, label=label, clip_on=False, *args, **kwargs
        )
        # Collect sample points for latter use (e.g. legend)
        if label is not None:
            self.samplePoints.append(l)
        return l

    def add_errstd(self, ind, label=None, normal=False, *args, **kwargs):
        """Add error standard deviation to TC diagram.
        *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        # (theta, radius)
        if normal:
            if self.horizon == 'sigstd':
                azimuth = np.arccos(0)
                radius = [1]     
            elif self.horizon == 'totstd':
                azimuth = np.arccos(self.corrcoef[ind])
                radius = self.corrcoef[ind]
        else:
            if self.horizon == 'sigstd':
                azimuth = np.arccos(0)
                radius = self.errstd[ind]      
            elif self.horizon == 'totstd':
                azimuth = np.arccos(self.corrcoef[ind])
                radius = self.sigstd[ind]      

        (l,) = self.ax.plot(
            azimuth, radius, label=label, clip_on=False, *args, **kwargs,
        )
        # Collect sample points for latter use (e.g. legend)
        if label is not None:
            self.samplePoints.append(l)
        return l

    def add_sigstd(self, ind, label=None, normal=False, *args, **kwargs):
        """Add signal standard deviation to TC diagram.
        *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        # (theta, radius)
        if normal:
            if self.horizon == 'sigstd':
                azimuth = np.arccos(1)
                radius = self.sigstd[ind]/self.errstd[ind]
            elif self.horizon == 'totstd':
                azimuth = np.arccos(self.corrcoef[ind])
                radius = self.corrcoef[ind]
        else:
            if self.horizon == 'sigstd':
                azimuth = np.arccos(1)
                radius = self.sigstd[ind]      
            elif self.horizon == 'totstd':
                azimuth = np.arccos(self.corrcoef[ind])
                radius = self.sigstd[ind]      

        (l,) = self.ax.plot(
            azimuth, radius, label=label, clip_on=False, *args, **kwargs,
        )
        # Collect sample points for latter use (e.g. legend)
        if label is not None:
            self.samplePoints.append(l)
        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid to the Taylor diagram.
        *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        self._ax.grid(*args, **kwargs)

    def add_totstd_contours(self, ind, normal=False, levels=1, **kwargs):
        """Add total standard deviation contours, defined by *levels*.
        *kwargs* is directly propagated to the
        `Figure.plot` command.
        """
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax), np.linspace(0, self.tmax)
        )
        if levels == 1:
            totstd = self.totstd[ind]
            if normal:
                if self.horizon == 'sigstd':
                    totstd = self.totstd[ind]/self.errstd[ind]             
                elif self.horizon == 'totstd':
                    totstd = 1               
            else:
                totstd = self.totstd[ind]                
            contours = self.ax.contour(ts, rs, rs, [totstd], **kwargs)
        else:
            contours = self.ax.contour(ts, rs, rs, levels, **kwargs)
        return contours

    def add_sigstd_contours(self, ind, levels=1, **kwargs):
        """Add signal standard deviation contours, defined by *levels*.
        *kwargs* is directly propagated to the
        `Figure.plot` command.
        """
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax), np.linspace(0, self.tmax)
        )
        if levels == 1:
            sigstd = self.sigstd[ind]
            contours = self.ax.contour(ts, rs, rs, [sigstd], **kwargs)
        else:
            contours = self.ax.contour(ts, rs, rs, levels, **kwargs)
        return contours

    def add_errstd_contours(self, ind, levels=1, **kwargs):
        """Add error standard deviation contours, defined by *levels*.
        *kwargs* is directly propagated to the
        `Figure.plot` command.
        """
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax), np.linspace(0, self.tmax)
        )
        if self.horizon == 'sigstd':
            if levels == 1:
                errstd = self.errstd[ind]
                contours = self.ax.contour(ts, rs, rs, [errstd], **kwargs)
            else:
                contours = self.ax.contour(ts, rs, rs, levels, **kwargs)
        elif self.horizon == 'totstd':
            rms = np.sqrt(
                self.totstd[ind] ** 2 + rs**2 - 2 * self.totstd[ind] * rs * np.cos(ts)
            )
            if levels == 1:
                errstd = self.errstd[ind]
                contours = self.ax.contour(ts, rs, rms, [errstd], **kwargs)
            else:
                contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours

    def add_totstd_side(self, ind, **kwargs):
        """Add a line extending from origin to the location of totstd.
        *kwargs* is directly propagated to the
        `Figure.plot` command.
        """
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax), np.linspace(0, self.tmax)
        )
        if self.horizon == 'sigstd':
            rline = self.ax.plot(
                [0,np.arccos(self.corrcoef[ind])],[0,self.totstd[ind]], **kwargs
            )
        return rline

    def add_skillscore_contours(self, levels=None, **kwargs):
        """Add skill score contours, defined by *levels*.
        *kwargs* is directly propagated to the
        `Figure.plot` command.
        """
        rs, ts = np.meshgrid(
            np.linspace(self.smin + 1e-3, self.smax), np.linspace(0, self.tmax)
        )
        # Compute skill score grid
        corr = np.cos(ts)
        xs = rs*np.cos(ts)
        ys = rs*np.sin(ts)
        zs = np.sqrt(xs**2+ys**2)
        ss = (1 - ys**2/(zs**2+xs**2))*corr
        if levels is None:
            levels = np.linspace(0.1, 0.9, 9)
        contours = self.ax.contour(ts, rs, ss, levels, **kwargs)
        return contours

    def calc_skillscore(self, ind):
        """Compute skill score."""
        xs = self.sigstd[ind]
        ys = self.errstd[ind]
        zs = self.totstd[ind]
        corr = self.corrcoef[ind]
        ss = (1 - ys**2/(zs**2+xs**2))*corr
        return ss

def test1():
    """Plot a base Triple Collocation diagram."""
    # Define subplots
    rects = [111]
    # System, error, and signal std
    errstds = {rects[0]: [0.06373, 0.02726, 0.05112]}
    corrcoefs = {rects[0]: [0.7963, 0.9256, 0.8577]}
    # Plot specifics
    sranges = {rects[0]: (0.0, 0.125)}
    labels = {rects[0]: ["RSP", "HSRL-2", "MODIS"]}
    titles = {rects[0]: ""}
    # gl2 = MaxNLocator(steps=[1,4,10])
    gl2 = MaxNLocator(steps=[3])
    rlocs = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
    # Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
    cmaps = [
        "tab:blue",
        "tab:red",
        "tab:orange",
        "tab:brown",
        "tab:purple",
        "tab:green",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    # Set up plot
    fig = plt.figure(figsize=(4, 4), dpi=300, constrained_layout=False)
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.125, top=0.9, wspace=0.3, hspace=0.3
    )
    # fig.suptitle("Triple collolcation analysis", size='large')

    for rect in rects:
        # Add TCD axes
        dia = TripleCollocationDiagram(
            errstds[rect],
            corrcoefs[rect],
            fig=fig,
            rect=rect,
            rlocs=rlocs,
            srange=sranges[rect],
            sgrid_locator=gl2,
            horizon='sigstd',
        )
        for i, _ in enumerate(dia.totstd):
            # Add total standard deviation point
            dia.add_totstd(
                i, 
                marker="o", 
                ls="", 
                mfc="none",
                # mfc=cmaps[i], 
                mec=cmaps[i], 
                ms=6)
            # Add error standard deviation point
            dia.add_errstd(
                i,
                marker="o",
                ls="",
                mfc="none",
                # mfc=cmaps[i], 
                mec=cmaps[i], 
                ms=6,
                label=labels[rect][i],
            )
            # Add signal standard deviation point
            dia.add_sigstd(
                i, 
                marker="o", 
                ls="",
                mfc="none",
                # mfc=cmaps[i], 
                mec=cmaps[i], 
                ms=6)
            # Add line from origin to total standard deviation
            dia.add_totstd_side(
                i,
                linestyle="solid",
                linewidth=0.8,
                color=colors.to_hex(cmaps[i]),
            )

        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        dia._ax.set_title(titles[rect])
        dia._ax.axis[:].major_ticklabels.set_fontsize(8)
        dia._ax.axis[:].label.set_fontsize(8)
        dia._ax.text(-0.12,0.5,"Error standard deviation",
                     transform=dia.ax.transAxes, rotation="vertical",va="center",
                     fontsize=8)
        # Add grid
        dia.add_grid(axis="both", lw=0.5)

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html
    fig.legend(
        dia.samplePoints,
        [p.get_label() for p in dia.samplePoints],
        bbox_to_anchor=(0.6, 0.825),
        numpoints=1,
        loc="lower center",
        borderaxespad=0.0,
        ncol=1,
        fontsize=8,
        frameon=False,
    )
    fig.text(0.73,0.96,'$\sigma_{\epsilon_i}$',size=8)
    fig.text(0.865,0.96,'$r_i$',size=8)
    # for rect in rects:
    for i, (errstd, corrcoef) in enumerate(zip(dia.errstd,dia.corrcoef)):
        fig.text(0.725,0.923-i*0.04, f'{errstd:7.4f}',fontsize=8)
        fig.text(0.85,0.923-i*0.04, f'{corrcoef:7.3f}',fontsize=8)

    # Default path is to your home directory.
    outfile = "fig_tcd1.eps"
    plt.savefig(outfile)
    plt.show()
    return dia

def test2():
    """Plot a base Triple Collocation diagram."""
    # Define subplots
    rects = [111]
    # System, error, and signal std
    errstds = {rects[0]: [0.06373, 0.02726, 0.05112, 0.0441, 0.02871, 0.04979]}
    corrcoefs = {rects[0]: [0.7963, 0.9256, 0.8577, 0.8356, 0.9015, 0.8670]}
    # Plot specifics
    sranges = {rects[0]: (0.0, 0.125)}
    labels = {
        rects[0]: [
            "RSP (before)",
            "HSRL-2 (before)",
            "MODIS (before)",
            "RSP (after)",
            "HSRL-2 (after)",
            "MODIS (after)",
        ]
    }
    titles = {rects[0]: ""}
    # gl2 = MaxNLocator(steps=[1,4,10])
    gl2 = MaxNLocator(steps=[3])
    rlocs = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
    # Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
    cmaps = ["tab:blue", "tab:red", "tab:orange", "tab:blue", "tab:red", "tab:orange"]
    markers = ["o", "o", "o", "s", "s", "s"]

    # Set up plot
    fig = plt.figure(figsize=(4, 4), dpi=300, constrained_layout=False)
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.175, top=0.95, wspace=0.3, hspace=0.3
    )
    # fig.suptitle("Triple collolcation analysis", size='large')

    for rect in rects:
        # Add TCD axes
        dia = TripleCollocationDiagram(
            errstds[rect],
            corrcoefs[rect],
            fig=fig,
            rect=rect,
            rlocs=rlocs,
            srange=sranges[rect],
            sgrid_locator=gl2,
            horizon='sigstd',
        )
        for i, _ in enumerate(dia.totstd):
            # Add total standard deviation point
            dia.add_totstd(
                i, 
                marker=markers[i], 
                ls="", 
                mfc="none",
                # mfc=cmaps[i], 
                mec=cmaps[i], 
                ms=6)
            # Add error standard deviation point
            dia.add_errstd(
                i,
                marker=markers[i],
                ls="",
                mfc="none",
                # mfc=cmaps[i], 
                mec=cmaps[i], 
                ms=6,
                label=labels[rect][i],
            )
            # Add signal standard deviation point
            dia.add_sigstd(
                i, 
                marker=markers[i], 
                ls="",
                mfc="none",
                # mfc=cmaps[i], 
                mec=cmaps[i], 
                ms=6)

        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        dia._ax.set_title(titles[rect])
        dia._ax.axis[:].major_ticklabels.set_fontsize(8)
        dia._ax.axis[:].label.set_fontsize(8)
        dia._ax.text(-0.12,0.5,"Error standard deviation",
                     transform=dia.ax.transAxes, rotation="vertical",va="center",
                     fontsize=8)
        # Add grid
        dia.add_grid(axis="both", lw=0.5)

        # Set up arrows
        arrows = [
            mpatches.FancyArrowPatch(
                (np.arccos(dia.corrcoef[n]), dia.totstd[n]),
                (np.arccos(dia.corrcoef[n + 3]), dia.totstd[n + 3]),
                facecolor=cmaps[n],
                edgecolor=cmaps[n],
                shrinkA=2.5,
                shrinkB=2.5,
                arrowstyle="->,head_length=2,head_width=2",
                mutation_scale=1,
            )
            for n in np.arange(3)
        ]
        # Add arrows
        for arrow in arrows:
            dia.ax.add_patch(arrow)

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html
    fig.legend(
        dia.samplePoints,
        [p.get_label() for p in dia.samplePoints],
        bbox_to_anchor=(0.5, 0.0),
        numpoints=1,
        loc="lower center",
        borderaxespad=0.0,
        ncol=2,
        fontsize=7,
        frameon=False,
    )

    # Default path is to your home directory.
    outfile = "fig_tcd2.eps"
    plt.savefig(outfile)
    plt.show()
    return dia

def test3():
    """Plot a base Triple Collocation diagram."""
    # Define subplots
    rects = [111]
    # System, error, and signal std
    errstds = {rects[0]: [0.06373, 0.02726, 0.05112, 0.0441, 0.02871, 0.04979]}
    corrcoefs = {rects[0]: [0.7963, 0.9256, 0.8577, 0.8356, 0.9015, 0.8670]}
    # Plot specifics
    sranges = {rects[0]: (0.0, 0.125)}
    labels = {
        rects[0]: [
            "RSP (before)",
            "HSRL-2 (before)",
            "MODIS (before)",
            "RSP (after)",
            "HSRL-2 (after)",
            "MODIS (after)",
        ]
    }
    titles = {rects[0]: ""}
    # gl2 = MaxNLocator(steps=[1,4,10])
    gl2 = MaxNLocator(steps=[3])
    rlocs = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
    # Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
    cmaps = ["tab:blue", "tab:red", "tab:orange", "tab:blue", "tab:red", "tab:orange"]
    markers = ["o", "o", "o", "s", "s", "s"]

    # Set up plot
    fig = plt.figure(figsize=(4, 4), dpi=300, constrained_layout=False)
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.175, top=0.95, wspace=0.3, hspace=0.3
    )
    # fig.suptitle("Triple collolcation analysis", size='large')

    for rect in rects:
        # Add TCD axes
        dia = TripleCollocationDiagram(
            errstds[rect],
            corrcoefs[rect],
            fig=fig,
            rect=rect,
            rlocs=rlocs,
            srange=sranges[rect],
            sgrid_locator=gl2,
            horizon='sigstd',
        )
        for i, _ in enumerate(dia.totstd):
            # Add total standard deviation point
            dia.add_totstd(
                i, 
                marker=markers[i], 
                ls="", 
                mfc="none",
                # mfc=cmaps[i], 
                mec=cmaps[i], 
                ms=6,
                label=labels[rect][i],
                )
            # ss = dia.calc_skillscore(i)

        # Add skill score
        ss_conts = dia.add_skillscore_contours(
            levels=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99],
            linestyles="solid",
            linewidths=0.8,
            colors="tab:brown",
        )
        # Add skill score label
        dia.ax.clabel(ss_conts, inline=True, fontsize=6, fmt="%.2f")

        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        dia._ax.set_title(titles[rect])
        dia._ax.axis[:].major_ticklabels.set_fontsize(8)
        dia._ax.axis[:].label.set_fontsize(8)
        dia._ax.text(-0.12,0.5,"Error standard deviation",
                     transform=dia.ax.transAxes, rotation="vertical",va="center",
                     fontsize=8)
        # Add grid
        dia.add_grid(axis="both", lw=0.5)

        # Set up arrows
        arrows = [
            mpatches.FancyArrowPatch(
                (np.arccos(dia.corrcoef[n]), dia.totstd[n]),
                (np.arccos(dia.corrcoef[n + 3]), dia.totstd[n + 3]),
                facecolor=cmaps[n],
                edgecolor=cmaps[n],
                shrinkA=2.5,
                shrinkB=2.5,
                arrowstyle="->,head_length=2,head_width=2",
                mutation_scale=1,
            )
            for n in np.arange(3)
        ]
        # Add arrows
        for arrow in arrows:
            dia.ax.add_patch(arrow)

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html
    fig.legend(
        dia.samplePoints,
        [p.get_label() for p in dia.samplePoints],
        bbox_to_anchor=(0.5, 0.0),
        numpoints=1,
        loc="lower center",
        borderaxespad=0.0,
        ncol=2,
        fontsize=7,
        frameon=False,
    )

    # Default path is to your home directory.
    outfile = "fig_tcd3.eps"
    plt.savefig(outfile)
    plt.show()
    return dia


if __name__ == '__main__':

    test1()
    test2()
    test3()

