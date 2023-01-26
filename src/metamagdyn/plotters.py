# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as plt

fig_ext = '.eps'


def plotPxhxf(SimAnlyzer, figName='Pxhxf', flim=None, hlim=None,
              SaveDir=None, cmap='jet'):
    '''
    plotPxhxf(SimAnlyzer, 
              figName='Pxhxf', 
              flim = None, hlim = None, 
              SaveDir=None, 
              cmap='jet')

    Plots the color map P vs H vs f

    Parameters
    ----------
    SimAnlyzer : SimAnlyzer object
        SimAnlyzer.Calc_Absortion must be run before calling this function
    figName : string or number
        Figure name or id
    flim : array, default = None
        Frequency limits. min = flim[0], max = min = flim[-1]
        None -> use SimAnlyzer.fs
    hlim : array, default = None
        Field limits. min = hlim[0], max = min = hlim[-1]
        None -> use SimAnlyzer.hs
    SaveDir : Directory string , default = None
        If SaveDir != None the figure will be saved with SaveDir/figName.svg
    cmap : ~matplotlib.colors.Colormap
        Colormap to use
    '''
    fig = plt.figure(figName, figsize=(3.3, 2.3))
    fig.clear()
    sp = plt.subplot(111)

    if flim == None:
        flim = SimAnlyzer.fs
    if hlim == None:
        hlim = SimAnlyzer.hs

    fi0, fi_1 = SimAnlyzer.fi(flim[0]), SimAnlyzer.fi(flim[-1])
    hi0, hi_1 = SimAnlyzer.hi(hlim[0]), SimAnlyzer.hi(hlim[-1])

    if hlim[0] > hlim[-1]:
        extent = numpy.array([hlim[-1], hlim[0], flim[0]/1E9, flim[-1]/1E9])
        S = SimAnlyzer.absorbed_power[hi0:hi_1+1:-1,
                                      fi0:fi_1+1].T.copy()
    else:
        extent = numpy.array([hlim[0], hlim[-1], flim[0]/1E9, flim[-1]/1E9])
        S = SimAnlyzer.absorbed_power[hi0:hi_1+1,
                                      fi0:fi_1+1].T.copy()

    S = S/S[~numpy.isnan(S)].max()
    S = S.clip(0, 1)
    im = sp.imshow(S,
                   origin='lower',
                   interpolation='none',
                   extent=extent,
                   cmap=cmap,
                   aspect='auto'
                   )
    sp.set_xlim(*extent[[0, 1]])
    sp.set_ylim(*extent[[2, 3]])

    sp.set_ylabel(r'$\mathrm{f\, (GHz)}$', labelpad=-2)
    sp.set_xlabel(r'$\mathrm{H\, (Oe)}$', labelpad=0)

    spp = {'top': 0.965,
           'bottom': 0.15,
           'left': 0.13,
           'right': 1.05,
           'hspace': 0,
           'wspace': 0}
    fig.subplots_adjust(**spp)

    cb = fig.colorbar(im, pad=0.02)
    cb.set_ticks([0.005, 0.5, 0.995])
    cb.set_ticklabels([r'$0$', '',  r'$1$'])
    cb.set_label(r'$\mathrm{P/P_{max}}$', labelpad=-8)

    fig.canvas.draw()

    if SaveDir != None:
        fig.savefig(SaveDir + figName + fig_ext)


def plotMxH(SimAnlyzer, figName='MxH', M_dir='Auto',
            SaveDir=None):

    if M_dir == 'Auto':
        M_dir = SimAnlyzer.DC_dir

    H = SimAnlyzer.hs
    M = numpy.einsum('i,ij->j', SimAnlyzer.DC_dir,
                     SimAnlyzer.mStatic)

    fig = plt.figure(figName,  (5, 4))
    plt.plot(H, M, '.-')
    plt.grid(True)
    plt.xlabel(r'$\rm H \, (Oe)$')
    plt.ylabel(r'$\rm M/Ms$')
    plt.tight_layout()

    if SaveDir != None:
        fig.savefig(SaveDir + figName + fig_ext)


def plotV(self, figName, z_slice=0, qS=10, osc=False):
    plotData = numpy.swapaxes(self.data, 0, 1)[:, :, z_slice].copy()
    plotData[plotData.real == 0] = numpy.nan
    color = numpy.arctan2(plotData[:, :, 1].real,
                          plotData[:, :, 0].real)*180/numpy.pi
    color[color < 0] += 360

    if osc:
        angle_0 = numpy.arctan2(
            plotData[:, :, 1].imag, plotData[:, :, 0].imag)*180/numpy.pi
        angle_osc = numpy.arctan2(plotData[:, :, 1].real + plotData[:, :, 1].imag,
                                  plotData[:, :, 0].real + plotData[:, :, 0].imag)*180/numpy.pi
        color = angle_osc - angle_0

    f = plt.figure(figName, figsize=(3.5, 3))
    f.clear()
    (nx, ny) = color.shape
    xs = (numpy.arange(-nx/2, nx/2, 1)+0.5)*self.dx
    ys = (numpy.arange(-ny/2, ny/2, 1)+0.5)*self.dy
    X, Y = numpy.meshgrid(xs, ys)
    extent = (-nx/2*self.dx, nx/2*self.dx, -ny/2*self.dy, +nx/2*self.dy)
    plt.quiver(X[::qS, ::qS], Y[::qS, ::qS], plotData[::qS, ::qS,
               0].real, plotData[::qS, ::qS, 1].real)  # , units='xy')
    if osc:
        #color[0,0] = 5.5
        #color[-1,-1] = -5.5
        #color = numpy.clip(color, -5.5, 5.5)
        cmap = 'jet'
    else:
        color[0, 0] = 0
        color[-1, -1] = 360
        cmap = 'hsv'
    plt.imshow(color,
               origin='lower',
               cmap=cmap,
               interpolation='none',
               extent=extent,
               aspect=1
               )
    plt.colorbar()


def plotC(self, figName, comp=0, z_slice=0, osc=True):

    if comp == 3:
        plotData = numpy.linalg.norm(numpy.swapaxes(
            self.data.real, 0, 1)[:, :, z_slice], axis=-1)
    else:
        plotData = numpy.swapaxes(self.data.real, 0, 1)[
            :, :, z_slice, comp].copy()
    plotData[plotData == 0] = numpy.nan

    if osc:
        amp = numpy.median(numpy.abs(plotData)) * 5
        plotData = numpy.clip(plotData, -amp, amp)
        plotData[0, 0] = amp
        plotData[-1, -1] = -amp
        plotData = plotData*1000
    f = plt.figure(figName, figsize=(3.5, 3))
    f.clear()
    (nx, ny) = plotData.shape
    extent = (-nx/2*self.dx, nx/2*self.dx, -ny/2*self.dy, +nx/2*self.dy)
    plt.imshow(plotData,
               origin='lower',
               cmap='jet',
               interpolation='none',
               extent=extent,
               aspect=1
               )
    plt.colorbar()


def Plot_vf_Eq(self, h,  z_slice=0, qS=10, figName='Auto', aspect=1, SaveDir=None, cmap='hsv'):
    self.EqVF(h)
    if figName == 'Auto':
        figName = 'Eq %0.1f Oe' % (self.hs[self.Last_hi])

    plotData = numpy.swapaxes(self.vf_Eq.data, 0, 1)[:, :, z_slice].copy()
    plotData[self.nan_mask] = numpy.nan
    color = numpy.arctan2(plotData[:, :, 1].real,
                          plotData[:, :, 0].real)*180/numpy.pi
    color[color < 0] += 360

    fig = plt.figure(figName, figsize=(3.5, 3))
    fig.clear()
    (nx, ny) = color.shape

    dx = self.vf_Eq.dx
    dy = self.vf_Eq.dy
    nx = self.vf_Eq.xnodes
    ny = self.vf_Eq.ynodes
    extent = numpy.array([-nx/2.0*dx, nx/2.0*dx, -ny/2.0*dy, +ny/2.0*dy])*1E6

    xs = (numpy.arange(-nx/2, nx/2, 1)+0.5)*dx*1E6
    ys = (numpy.arange(-ny/2, ny/2, 1)+0.5)*dy*1E6
    X, Y = numpy.meshgrid(xs, ys)
    plt.quiver(X[::qS, ::qS], Y[::qS, ::qS], plotData[::qS,
               ::qS, 0].real, plotData[::qS, ::qS, 1].real)
    color[0, 0] = 0
    color[-1, -1] = 360
    plt.imshow(color,
               origin='lower',
               cmap=cmap,
               interpolation='none',
               extent=extent,
               aspect=aspect
               )
    plt.colorbar()
    if SaveDir != None:
        fig.savefig(SaveDir + figName + '.png')
