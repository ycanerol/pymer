#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import scipy.ndimage as snd

import analysis_scripts as asc
import iofuncs as iof

from randpy import randpy
import matplotlib as mpl


class Stimulus:
    def __init__(self, exp, stimnr, maxframes=None):
        self.exp = exp
        self.stimnr = stimnr
        self.clusters, self.metadata = asc.read_spikesheet(self.exp)
        self.nclusters = self.clusters.shape[0]
        self.exp_dir = iof.exp_dir_fixer(exp)
        self.exp_foldername = os.path.split(self.exp_dir)[-1]
        self.stimname = iof.getstimname(exp, stimnr)
#        self.get_frametimings()
        self._getstimtype()
        self.refresh_rate = self.metadata['refresh_rate']
        self.sampling_rate = self.metadata['sampling_freq']
        self.maxframes = maxframes
        if maxframes:
            self.maxframes_i = maxframes + 1
        else:
            self.maxframes_i = None

    def _getstimtype(self):
        sortedstim = asc.stimulisorter(self.exp)

        for key, val in sortedstim.items():
            if self.stimnr in val:
                stimtype = key
        self.stimtype = stimtype

    def get_frametimings(self):
        frametimings = asc.readframetimes(self.exp, self.stimnr)[:self.maxframes_i]
        self.frametimings = frametimings

    def readpars(self):
        self.param_file = asc.read_parameters(self.exp, self.stimnr)

    def clusterstats(self):
        print(f'There are {self.nclusters} clusters in this experiment.')
        for i in range(1, 5):
            count = self.clusters[self.clusters[:, 2] < i+1, :].shape[0]
            print(f'({100*count/self.nclusters:>5.1f}%) {count:>4} are rated {i}',
                  ' or better'*(i != 1), '.', sep='')

    def read_raster(self, i):
        ch, cl = self.clusters[i, :2]
        return asc.read_raster(self.exp, self.stimnr, ch, cl)

    def binnedspiketimes(self, i):
        return asc.binspikes(self.read_raster(i), self.frametimings)[:self.maxframes_i]


class Parameters:
    """
    Dummy class to hold parameters.
    """
    def __init__(self):
        pass
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return self.__str__()


class OMB(Stimulus):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stimtype != 'OMB':
            raise ValueError('The stimulus is not OMB.')
        self.readpars()
        self._setdefaults()
        self._checkimplemented()
        self._get_frametimings()
        self._generatestimulus()

    def _setdefaults(self):
        pars = Parameters()
        param_file = self.param_file
        pars.stimframes = param_file.get('stimFrames', 108000)
        pars.preframes = param_file.get('preFrames', 200)
        pars.nblinks = param_file.get('Nblinks', 2)
        pars.initial_stepseed = param_file.get('seed', -10000)
        pars.objseed = param_file.get('objseed', -1000)
        pars.stepsize = param_file.get('stepsize', 2)
        pars.gausssteps = param_file.get('gaussteps', True)
        pars.smoothtrajectory = param_file.get('smoothtrajectory', False)
        pars.eyemovements = param_file.get('eyemovements', False)
        pars.bgnoise = param_file.get('bgnoise', 4)

        if pars.bgnoise != 1:
            pars.bgstixel = param_file.get('bgstixel', 5)
        else:
            pars.bgstixel = param_file.get('bgstixel', 10)

        pars.bgcontrast = param_file.get('bgcontrast', 0.3)
        pars.bggenerationseed = -10000
        pars.filterstd = param_file.get('filterstdv', pars.bgstixel)
        pars.meanintensity = param_file.get('meanintensity', 0.5)
        pars.contrast = param_file.get('contrast', 1)
        pars.squareheight = 800
        pars.squarewidth = 800
        self.pars = pars

    def _get_frametimings(self):
        filter_length, frametimings = asc.ft_nblinks(self.exp,
                                                     self.stimnr,
                                                     self.pars.nblinks,
                                                     self.refresh_rate)
        self.frametimings = frametimings
        self.filter_length = filter_length

    def _checkimplemented(self):
        pars = self.pars
        if pars.bgnoise != 4:
            raise NotImplementedError('Only gaussian correlated binary '
                                      'noise is implemented.')
        if (pars.smoothtrajectory
            or pars.eyemovements
            or not pars.gausssteps):
            raise NotImplementedError('Analysis of only non-smoothed '
                                      'gaussian steps are implemented.')

    def _generatesteps(self):
        pars = self.pars
        frametimings = self.frametimings[:-1]
        self.frame_duration = np.ediff1d(frametimings).mean()
        if self.maxframes is None:
            ntotal = int(pars.stimframes/pars.nblinks)
            if ntotal != frametimings.shape[0]:
                print(f'For {self.exp}\nstimulus {iof.getstimname(self.exp, self.stimnr)} :\n'
                      f'Number of frames specified in the parameters file ({ntotal}'
                      f' frames) and frametimings ({frametimings.shape[0]}) do not'
                      ' agree!'
                      ' The stimulus was possibly interrupted during recording.'
                      ' ntotal is changed to match actual frametimings.')
                ntotal = frametimings.shape[0]
        else:
            ntotal = self.maxframes

        # Make a copy of the initial seed to not change it
        pars.stepseed = pars.initial_stepseed
        randnrs, pars.stepseed = randpy.gasdev(pars.stepseed, ntotal*2)
        randnrs = np.array(randnrs)*pars.stepsize

        xsteps = randnrs[::2]
        ysteps = randnrs[1::2]

        steps = np.vstack((xsteps, ysteps))
        # HINT
#        steps *= -1
        steps /= pars.bgstixel

        self.bgsteps = steps
        self.ntotal = ntotal

    def _generatetraj(self):
        self._generatesteps()
        self.bgtraj = np.cumsum(self.bgsteps, axis=1)
        self.bgtraj_clipped = np.fmod(self.bgtraj, 1.5*self.texpars.noiselim[0])

    def _generatetexture(self):
        pars = self.pars
        self.texpars = Parameters()
        bgstixel = pars.bgstixel
        filterstd = pars.filterstd
        meanintensity = pars.meanintensity
        filterwidth = filterstd/bgstixel*3
        noiselim = (np.ceil(np.array([pars.squareheight,
                                      pars.squarewidth])/bgstixel)
                    ).astype(int)

        # Gaussian filter is applied to the noise field by a for loop in the cpp code,
        # and its norm is
        xx, yy = np.meshgrid(np.arange(2*filterwidth), np.arange(2*filterwidth))
        gfilter = np.exp2(-((xx-filterwidth)**2
                            +(yy-filterwidth)**2)/(2*(filterstd/bgstixel)**2))

        norm = gfilter.sum()
        randnrs = np.reshape(randpy.gasdev(pars.bggenerationseed,
                              noiselim[0]*noiselim[1])[0],
                             (noiselim[0], noiselim[1]))
        noisefield = (pars.meanintensity
                    + pars.meanintensity*pars.bgcontrast*randnrs)

        texturebasic = snd.convolve(noisefield, gfilter)
        texturetiled = snd.convolve(np.tile(noisefield, [3, 3]), gfilter)

        texturebasic = self._normalizetexture(texturebasic, norm, meanintensity, filterstd, bgstixel)
        texturetiled = self._normalizetexture(texturetiled, norm, meanintensity, filterstd, bgstixel)

        self.texture = texturetiled
        self.texturebasic = texturebasic

        self.texture_flipped = np.flipud(texturetiled)
        self.texpars.noiselim = noiselim
        self.texpars.filterwidth = filterwidth


    def _generatetexture_withloops(self):
        pars = self.pars
        self.texpars_withloop = Parameters()
        bgstixel = int(pars.bgstixel)
        filterstd = pars.filterstd
        meanintensity = pars.meanintensity
        filterwidth = int(filterstd/bgstixel*3)
        noiselim = (np.ceil(np.array([pars.squareheight,
                                      pars.squarewidth])/bgstixel)
                    ).astype(int)
        gfilter = np.zeros((pars.squareheight, pars.squarewidth))
        seed = pars.bggenerationseed

        texture = np.zeros((pars.squareheight, pars.squarewidth))
        noisefield = np.zeros((pars.squareheight, pars.squarewidth))
        norm = 0
        for i in range(noiselim[0]):
            for j in range(noiselim[1]):
                rndnr, seed = randpy.gasdev(seed)
                noisefield[i, j] = meanintensity + meanintensity*pars.bgcontrast*rndnr
                if (i < filterwidth * 2 + 1 ) and (j < filterwidth * 2 + 1 ):
                    gfilter[i, j] = np.exp2(-((i - filterwidth)**2 + (j - filterwidth)**2) / (2 * (filterstd / bgstixel)**2))
                    norm += gfilter[i, j]
        for i in range(noiselim[0]):
            for j in range(noiselim[1]):
                helper = 0
                for ki in range(filterwidth*2 + 1):
                    for kj in range(filterwidth*2 + 1):
                        helper += (noisefield[(i - (ki - filterwidth) + noiselim[0]) % noiselim[0],
                                             (j - (kj - filterwidth) + noiselim[1]) % noiselim[1]]
                                    * gfilter[ki, kj])
#                c = 255 * ((helper / norm - meanintensity) * filterstd / bgstixel + meanintensity)
                c = self._normalizetexture(helper, norm, meanintensity, filterstd, bgstixel)

                for gi in range(bgstixel):
                    for gj in range(bgstixel):
                        texture[i*bgstixel + gi, j*bgstixel + gj] = c

        self.texturewithloops = texture


    def _normalizetexture(self, texture, norm, meanintensity, filterstd, bgstixel):
        texture = ((texture/norm-meanintensity)*filterstd/bgstixel + meanintensity)
        # Clip the contrast values at 0 and 1, then center around 0 so that
        # -1 is black and 1 is white.
        texture = np.clip(texture, 0, 1)*2-1
        return texture

    def _generatestimulus(self):
        self._generatetexture()
        self._generatetraj()

    def playstimulus(self, begin=0, frames=120, pause_duration=None):
        """
        Play the OMB stimulus in a Qt window

        pause_duration is in seconds.
        """
        import matplotlib.pyplot as plt
        import plotfuncs as plf

        if pause_duration is None:
            pause_duration = self.pars.nblinks/self.refresh_rate
        # Switch to qt backend if it is not active already.
        plf.check_interactive_backend()

        noiselim = self.texpars.noiselim
        fig = plt.figure()
        plt.imshow(self.texture_flipped, cmap='Greys_r', origin='lower')
        for i in range(begin, begin+frames):
            # Make sure we don't modify the actual bgtraj but make a copy of it
            coord = self.bgtraj[:, i].copy()
            # Flip the y movement for display purposes. Note that the random
            # numbers for the steps are already multiplied by -1 once.
            coord[1] *= -1
            # Center the coordinates
            coord = coord + noiselim*1.5
            plt.xlim(coord[0]+[-noiselim[0]/2, noiselim[0]/2])
            # The default order for ylim args is bottom, top. If the order
            # is reversed, the image is flipped.
            plt.ylim(coord[1]+[noiselim[1]/2, -noiselim[1]/2])
#            plt.ylim(coord[1]+[-noiselim[1]/2, noiselim[1]/2])
            plt.pause(pause_duration)
        plt.show()
        return fig


    def generatecontrast(self, coord, window=0):
        """
        Returns the contrast value for a particular coordinate throughout
        the whole experiment. The coordinates are rounded to be used as
        indices.
        """
        coord = np.array(coord)
        # Movement in x direction corresponds to translation in left/right
        # axis; and y to up/down axis. Since the first index is the row,
        # we need swap x and y if we want to keep the order (x,y).
        coord = np.flipud(coord)
        # Use the clipped trajectory in case the
        # texture goes out of the central region.
        traj = self.bgtraj_clipped
        contrast = np.zeros((window*2+1, window*2+1, self.ntotal))
        for i, ii in enumerate(range(-window, window+1)):
            for j, jj in enumerate(range(-window, window+1)):
                traj_loop = np.round(traj
                            + coord[..., None]
                            # HINT: center the texture
                            #- self.texpars.noiselim[:, None]*1.5
                            + np.array([ii, jj])[..., None]).astype(int)
                contrast[i, j] = self.texture_flipped[traj_loop[0], traj_loop[1]]
        return contrast

    def regioncontrast(self, coord, window):
        """
        Returns the average contrast values for a group of coordinates.
        """
        import miscfuncs as msc

        msc.cut_around_center(self.texture)

#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import animation
    exp, ombstimnr = '20180710', 8
    checkerstimnr = 6
    maxframes = 20000

    st = OMB(exp, ombstimnr, maxframes=maxframes)
    st.clusterstats()
    #import time; time.sleep(2)
    #st.playstimulus(frames=6, pause_duration=None)

    #%%
    from datetime import datetime
    import miscfuncs as msc
    startime = datetime.now()
    a = st.generatecontrast(st.texpars.noiselim/2, 100)
    # Capitalize name of variable to prevent it from slowing variable exp. down
    RW = asc.rolling_window(a, st.filter_length)

    all_spikes = np.zeros((st.nclusters, st.ntotal))
    for i in range(st.nclusters):
        all_spikes[i, :] = st.binnedspiketimes(i)

    # Add a cell that spikes at every bin to find the non-spike triggered average
    all_spikes = np.vstack((all_spikes, np.ones(all_spikes.shape[-1])))
    stas = np.einsum('abcd,ec->eabd', RW, all_spikes)
    del RW
    stas /= all_spikes.sum(axis=(-1))[:, np.newaxis, np.newaxis, np.newaxis]
    print(f'{msc.timediff(startime)} elapsed for contrast generation and STA calculation')
    #%%
#    fig1 = plt.figure(1)
#    fig2 = plt.figure(2)
    data = iof.load(exp, checkerstimnr)
    ckstas = np.array(data['stas'])
    ckstas /= np.nanmax(np.abs(ckstas), axis=0)[np.newaxis, ...]
    ckstas = ckstas[..., ::-1]
#    imshowkwargs_omb = dict(cmap='RdBu_r', vmin=stas.min(), vmax=stas.max())
#    imshowkwargs_chk = dict(cmap='RdBu_r', vmin=-np.nanmax(np.abs(ckstas)), vmax=np.nanmax(np.abs(ckstas)))


#    fig3, axes = plt.subplots(1, 2, num=3)
#    i = 32
#    ims = []
#    for j in range(20):
#        im_omb = axes[0].imshow(stas[i, :, :, j], **imshowkwargs_omb, animated=True)
#        im_chk = axes[1].imshow(ckstas[i, :, :, 2*j], **imshowkwargs_chk, animated=True)
#        ims.append([im_omb, im_chk])
#
#    #    plt.show()
#    animation.ArtistAnimation(fig3, ims, interval=200, blit=True, repeat_delay=200)

    #%%
    import plotfuncs as plf
    # Subtract the non-triggered average from all stas
    stas_corr = stas-stas[-1, :, :, :]
    # Combine corrected and non-corrected versions to see them side by side
    combined = np.empty((stas.shape[0]*2, *stas.shape[1:]))
    combined[::2] = stas
    combined[1::2] = stas_corr

    fig, sl = plf.multistabrowser(combined, st.frame_duration, cmap='Greys_r')
