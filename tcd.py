#!/usr/bin/env python
"""
Triple Collocation diagram (Siu et al., submitted) implementation.

This is a heavy modification of Yannick Copin's work for implementing
Taylor diagram.  The original code is placed in 
https://gist.github.com/ycopin/3342888.

If you use it for your research, please cite the following publication:
    Siu et al., 2023: Summarizing multiple aspects of triple collocation 
        analysis in a single diagram, submitted.

"""
# Python library imports
# Third party imports
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.floating_axes import (FloatingSubplot,
                                                   GridHelperCurveLinear)
from mpl_toolkits.axisartist.grid_finder import (DictFormatter, FixedLocator,
                                                 MaxNLocator)
# Local imports


class TripleCollocationDiagram(object):
    """
    Triple collocation diagram.
    Plot standard deviation and correlation of three datasets 
    in a single-quadrant polar plot, with r=std and theta=arccos(correlation).
    """
    def __init__(self, totstd, errstd, sigstd, fig=None, rect=111,
                 rlocs=None, srange=[0,1.5], sgrid_locator=None, 
                 normal=False, extend=False):
        """
        Set up Triple Collocation diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        
        Parameters
        ----------
        totstd : list or array_like
            List of total standard deviation of datasets (can be more than three).
        errstd : list or array_like
            List of error standard deviation of datasets (can be more than three).
        sigstd : list or array_like
            List of signal standard deviation of datasets (can be more than three).
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
        extend: bool, default: False
            If True, extend diagram to negative correlations. 
            Legacy kwarg, currently not in use.            
        """
        # Total standard deviation
        if isinstance(totstd, list):
            self.totstd = totstd
        else:
            self.totstd = [*totstd]
        # Error standard deviation
        if isinstance(errstd, list):
            self.errstd = errstd
        else:
            self.errstd = [*errstd]
        # Signal standard deviation
        if isinstance(sigstd, list):
            self.sigstd = sigstd
        else:
            self.sigstd = [*sigstd]
        # Standard deviation axis extent (in units of total std)
        if normal:
            self.smin = srange[0]
            self.smax = srange[1]
        else:
            self.smin = srange[0]*self.totstd[0]
            self.smax = srange[1]*self.totstd[0]
        # Correlation coefficient
        self.corrcoef = [(a**2-c**2-b**2)/(-2*b*c) for a,b,c in 
                         zip(self.errstd,self.totstd,self.sigstd)]
        # Set up input Figure
        if fig is None:
            fig = plt.figure()

        # Set up base polar transform
        tr = PolarAxes.PolarTransform()
        # Correlation labels
        if rlocs is None:
            rlocs = np.array([0.0,0.2,0.4,0.6,0.7,0.8,0.85,0.9,0.95,0.99,1.0])
            # rlocs = np.array([0.0,0.3,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0])
        # Diagram extended to negative correlations
        if extend:
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        # Diagram limited to positive correlations
        else:
            self.tmax = np.pi/2
        tlocs = np.arccos(rlocs)    # Conversion to polar angles (theta)
        gl1 = FixedLocator(tlocs)   # Positions
        tf1 = DictFormatter(dict(zip(tlocs,map(str,rlocs))))
        # Set up the plot boundary
        ghelper = GridHelperCurveLinear(
            tr, extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1,
            grid_locator2=sgrid_locator, tick_formatter2=None)

        # Set up Axes
        ax = FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        # Adjust all axes
        ax.axis[:].major_ticklabels.set_pad(1)
        ax.axis[:].major_ticks.set_tick_out(True)
        ax.axis[:].label.set_pad(1)
        # Adjust 'Angle-axis'
        ax.axis['top'].set_axis_direction('bottom')
        ax.axis['top'].toggle(ticklabels=True, label=True)
        ax.axis['top'].major_ticklabels.set_axis_direction('top')
        ax.axis['top'].label.set_axis_direction('top')
        ax.axis['top'].label.set_text('Correlation coefficient')
        # Adjust 'X-axis'
        ax.axis['left'].set_axis_direction('bottom')
        ax.axis['left'].label.set_text('Standard deviation')
        # Adjust 'Y-axis'
        ax.axis['right'].set_axis_direction('top')
        ax.axis['right'].toggle(ticklabels=True)
        ax.axis['right'].major_ticklabels.set_axis_direction(
            'bottom' if extend else 'left')
        if self.smin:
            ax.axis['bottom'].toggle(ticklabels=False, label=False)
        else:
            ax.axis['bottom'].set_visible(False)    # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates
        self.samplePoints = []          # Initialize sample points

    def add_totstd(self, ind, label=None, normal=False, *args, **kwargs):
        """
        Add total standard deviation to TC diagram.
        *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        # (theta, radius)
        if normal:
            l, = self.ax.plot([0], [1], label=label, 
                              clip_on=False, *args, **kwargs)
        else:
            l, = self.ax.plot([0], self.totstd[ind], label=label, 
                              clip_on=False, *args, **kwargs)
        # Collect sample points for latter use (e.g. legend)
        if label is not None:
            self.samplePoints.append(l)
        return l

    def add_errstd(self, ind, label=None, normal=False, *args, **kwargs):
        """
        Add error standard deviation to TC diagram.
        *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        # (theta, radius)
        if normal:
            l, = self.ax.plot(np.arccos(self.corrcoef[ind]), self.corrcoef[ind],
                              label=label, *args, **kwargs)
        else:
            l, = self.ax.plot(np.arccos(self.corrcoef[ind]), self.sigstd[ind],
                              label=label, *args, **kwargs)
        # Collect sample points for latter use (e.g. legend)
        if label is not None:
            self.samplePoints.append(l)
        return l

    def add_grid(self, *args, **kwargs):
        """
        Add a grid to the Taylor diagram.
        *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        self._ax.grid(*args, **kwargs)

    def add_totstd_contours(self, ind, levels=1, **kwargs):
        """
        Add total standard deviation contours, defined by *levels*.
        *kwargs* is directly propagated to the
        `Figure.plot` command.
        """
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        if levels == 1:
            totstd = self.totstd[ind]
            contours = self.ax.contour(ts, rs, rs, [totstd], **kwargs)
        else:
            contours = self.ax.contour(ts, rs, rs, levels, **kwargs)
        return contours

    def add_sigstd_contours(self, ind, levels=1, **kwargs):
        """
        Add signal standard deviation contours, defined by *levels*.
        *kwargs* is directly propagated to the
        `Figure.plot` command.
        """
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        if levels == 1:
            sigstd = self.sigstd[ind]
            contours = self.ax.contour(ts, rs, rs, [sigstd], **kwargs)
        else:
            contours = self.ax.contour(ts, rs, rs, levels, **kwargs)
        return contours

    def add_errstd_contours(self, ind, levels=1, **kwargs):
        """
        Add error standard deviation contours, defined by *levels*.
        *kwargs* is directly propagated to the
        `Figure.plot` command.
        """
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute error standard deviation grid
        rms = np.sqrt(self.totstd[ind]**2 + rs**2
                      - 2*self.totstd[ind]*rs*np.cos(ts))
        if levels == 1:
            errstd = self.errstd[ind]
            contours = self.ax.contour(ts, rs, rms, [errstd], **kwargs)
        else:
            contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours

    def add_skillscore_contours(self, levels=None, **kwargs):
        """
        Add skill score contours, defined by *levels*.
        *kwargs* is directly propagated to the
        `Figure.plot` command.
        """
        rs, ts = np.meshgrid(np.linspace(self.smin+1E-3, self.smax),
                             np.linspace(0, self.tmax))
        # Compute skill score grid
        corr = np.cos(ts)
        # From equation (5) in Taylor (2001)
        ss = 4/(rs+1/rs)**2 * ((1+corr)/2)**4
        # ss = 4/(rs+1/rs)**2 * ((1+corr)/2)
        if levels is None:
            levels = np.linspace(0.1,0.9,9)        
        contours = self.ax.contour(ts, rs, ss, levels, **kwargs)
        return contours


def test1():
    """Plot a base Triple Collocation diagram."""
    # Define subplots
    rects = [111]
    # System, error, and signal std
    totstds = {rects[0]: [0.1054, 0.072,  0.09945]}
    errstds = {rects[0]: [0.06373,0.02726,0.05112]}
    sigstds = {rects[0]: [0.08393,0.06664,0.0853]}
    # Plot specifics
    sranges = {rects[0]: (0.0,0.125/max(totstds[rects[0]]))}
    labels = {rects[0]: ['RSP','HSRL-2','MODIS']}
    titles = {rects[0]: ''}
    # gl2 = MaxNLocator(steps=[1,4,10])
    gl2 = MaxNLocator(steps=[3])
    rlocs = np.array([0.0,0.2,0.4,0.6,0.7,0.8,0.85,0.9,0.95,0.99,1.0])
    # Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
    cmaps = ['tab:blue','tab:red','tab:orange','tab:brown','tab:purple',
             'tab:green','tab:pink','tab:gray','tab:olive','tab:cyan']

    # Set up plot
    fig = plt.figure(figsize=(4,4), dpi=300, constrained_layout=False)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.125, top=0.9,
                        wspace=0.3, hspace=0.3)
    # fig.suptitle("Triple collolcation analysis", size='large')

    for rect in rects:
        # Add TCD axes
        dia = TripleCollocationDiagram(
            totstds[rect], errstds[rect], sigstds[rect],
            fig=fig, rect=rect,
            rlocs=rlocs, srange=sranges[rect], sgrid_locator=gl2)
        for i, _ in enumerate(totstds[rect]):
            # Add total standard deviation point
            dia.add_totstd(i, marker='o', ls='',
                mfc=cmaps[i], mec=cmaps[i], ms=6)
            # Add error standard deviation point
            dia.add_errstd(i, marker='o', ls='',
                mfc='none', mec=cmaps[i], ms=6,label=labels[rect][i])
            # Add signal standard deviation contour(s)
            sig_conts = dia.add_sigstd_contours(i, linestyles=[(0,(5,3))],
                linewidths=0.8, colors=colors.to_hex(cmaps[i]))
            # Add error standard deviation contour(s)
            err_conts = dia.add_errstd_contours(i, linestyles='solid',
                linewidths=0.8, colors=colors.to_hex(cmaps[i]))
            # Add contour labels
            dia.ax.clabel(sig_conts, inline=True, fontsize=6, fmt='%.3f')
            dia.ax.clabel(err_conts, inline=True, fontsize=6, fmt='%.3f')

        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        dia._ax.set_title(titles[rect])
        dia._ax.axis[:].major_ticklabels.set_fontsize(8)
        dia._ax.axis[:].label.set_fontsize(8)
        # Add grid
        dia.add_grid(axis='both',lw=0.5)

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html
    fig.legend(dia.samplePoints, [p.get_label() for p in dia.samplePoints],
               bbox_to_anchor=(0.5, 0), numpoints=1, loc='lower center',
               borderaxespad=0.0, ncol=6, fontsize=8, frameon=False)

    # Default path is to your home directory.
    outfile = 'fig_tcd1.pdf'
    plt.savefig(outfile)
    # plt.show()
    return dia


def test2():
    """Plot a base Triple Collocation diagram with more than 3 datasets."""
    # Define subplots
    rects = [111]
    # System, error, and signal std
    totstds = {rects[0]: [0.1054, 0.072,  0.09945,0.08029,0.06632,0.09991]}
    errstds = {rects[0]: [0.06373,0.02726,0.05112,0.0441, 0.02871,0.04979]}
    sigstds = {rects[0]: [0.08393,0.06664,0.0853, 0.06709,0.05979,0.08662]}
    # Plot specifics
    sranges = {rects[0]: (0.0,0.125/max(totstds[rects[0]]))}
    labels = {rects[0]: ['RSP (before)','HSRL-2 (before)','MODIS (before)',
                         'RSP (after)','HSRL-2 (after)','MODIS (after)']}
    titles = {rects[0]: ''}
    # gl2 = MaxNLocator(steps=[1,4,10])
    gl2 = MaxNLocator(steps=[3])
    rlocs = np.array([0.0,0.2,0.4,0.6,0.7,0.8,0.85,0.9,0.95,0.99,1.0])
    # Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
    cmaps = ['tab:blue','tab:red','tab:orange',
             'tab:blue','tab:red','tab:orange']
    markers = ['o','o','o','s','s','s']

    # Set up plot
    fig = plt.figure(figsize=(4,4), dpi=300, constrained_layout=False)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.175, top=0.95,
                        wspace=0.3, hspace=0.3)
    # fig.suptitle("Triple collolcation analysis", size='large')

    for rect in rects:
        # Add TCD axes
        dia = TripleCollocationDiagram(
            totstds[rect], errstds[rect], sigstds[rect],
            fig=fig, rect=rect,
            rlocs=rlocs, srange=sranges[rect], sgrid_locator=gl2)
        for i, _ in enumerate(totstds[rect]):
            # Add total standard deviation point
            dia.add_totstd(i, marker=markers[i], ls='',
                mfc=cmaps[i], mec=cmaps[i], ms=5)
            # Add error standard deviation point
            dia.add_errstd(i, marker=markers[i], ls='',
                mfc='none', mec=cmaps[i], ms=5, label=labels[rect][i])

        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        dia._ax.set_title(titles[rect])
        dia._ax.axis[:].major_ticklabels.set_fontsize(8)
        dia._ax.axis[:].label.set_fontsize(8)
        # Add grid
        dia.add_grid(axis='both',lw=0.5)

        # Set up arrows
        arrows = [
            mpatches.FancyArrowPatch(
            (np.arccos(dia.corrcoef[n]),dia.sigstd[n]),
            (np.arccos(dia.corrcoef[n+3]),dia.sigstd[n+3]),
            facecolor=cmaps[n],edgecolor=cmaps[n],shrinkA=2.5, shrinkB=2.5,
            arrowstyle='->,head_length=2,head_width=2',mutation_scale=1) 
            for n in np.arange(3)
            ]
        # Add arrows
        for arrow in arrows:
            dia.ax.add_patch(arrow)

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html
    fig.legend(dia.samplePoints, [p.get_label() for p in dia.samplePoints],
               bbox_to_anchor=(0.5, 0.0), numpoints=1, loc='lower center',
               borderaxespad=0.0, ncol=2, fontsize=7, frameon=False)

    # Default directory is to your home directory.
    outfile = 'fig_tcd2.pdf'
    plt.savefig(outfile)
    
    # plt.show()
    return dia


def test3():
    """Plot a Triple Collocation diagram with skill score contours."""
    # Define subplots
    rects = [111]
    # System, error, and signal std
    totstds = {rects[0]: [0.1054, 0.072,  0.09945,0.08029,0.06632,0.09991]}
    errstds = {rects[0]: [0.06373,0.02726,0.05112,0.0441, 0.02871,0.04979]}
    sigstds = {rects[0]: [0.08393,0.06664,0.0853, 0.06709,0.05979,0.08662]}
    # Plot specifics
    sranges = {rects[0]: (0.0,1.4)}
    labels = {rects[0]: ['RSP (before)','HSRL-2 (before)','MODIS (before)',
                         'RSP (after)','HSRL-2 (after)','MODIS (after)']}
    titles = {rects[0]: ''}
    # gl2 = MaxNLocator(steps=[1,4,10])
    gl2 = MaxNLocator(steps=[5])
    rlocs = np.array([0.0,0.2,0.4,0.6,0.7,0.8,0.85,0.9,0.95,0.99,1.0])
    # Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
    cmaps = ['tab:blue','tab:red','tab:orange','tab:blue','tab:red','tab:orange']
    markers = ['o','o','o','s','s','s']

    # Set up plot
    fig = plt.figure(figsize=(4,4), dpi=300, constrained_layout=False)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.175, top=0.95,
                        wspace=0.3, hspace=0.3)
    # fig.suptitle("Triple collolcation analysis", size='large')

    for rect in rects:
        # Add TCD axes
        dia = TripleCollocationDiagram(
            totstds[rect], errstds[rect], sigstds[rect],
            fig=fig, rect=rect, normal=True,
            rlocs=rlocs, srange=sranges[rect], sgrid_locator=gl2)
        for i, _ in enumerate(totstds[rect]):
            # Add total standard deviation point
            dia.add_totstd(i, marker=markers[i], ls='', normal=True, 
                mfc=cmaps[i], mec=cmaps[i], ms=5)
            # Add error standard deviation point
            dia.add_errstd(i, marker=markers[i], ls='', normal=True, 
                mfc='none', mec=cmaps[i], ms=5, label=labels[rect][i])

        # Add skill score
        ss_conts = dia.add_skillscore_contours(
            levels=np.linspace(0.1,0.9,9),
            linestyles='solid', linewidths=0.8, 
            colors='tab:brown')
        # Add skill score label
        dia.ax.clabel(ss_conts, inline=True, fontsize=6, fmt='%.1f')

        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        dia._ax.set_title(titles[rect])
        dia._ax.axis[:].major_ticklabels.set_fontsize(8)
        dia._ax.axis[:].label.set_fontsize(8)
        dia._ax.axis['left'].label.set_text('Standard deviation (normalized)')
        # Add grid
        dia.add_grid(axis='both',lw=0.5)

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html
    fig.legend(dia.samplePoints, [p.get_label() for p in dia.samplePoints],
               bbox_to_anchor=(0.5, 0.0), numpoints=1, loc='lower center',
               borderaxespad=0.0, ncol=2, fontsize=7, frameon=False)

    # Default directory is to your home directory.
    outfile = 'fig_tcd3.pdf'
    plt.savefig(outfile)
    # plt.show()

    return dia


if __name__ == '__main__':

    test1()
    test2()
    test3()

