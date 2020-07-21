import os

import numpy as np
import scipy.ndimage as snd

import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf

from randpy import randpy

from stimulus import Stimulus, Parameters


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
        if self.maxframes is None:
            frametimings = frametimings[:-1]
        self.frametimings = frametimings
        self.filter_length = filter_length
        self.frame_duration = np.ediff1d(frametimings).mean()

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
        frametimings = self.frametimings
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

        steps /= pars.bgstixel

        self.bgsteps = steps
        self.ntotal = ntotal

    def _generatetraj(self):
        self._generatesteps()
        self.bgtraj = np.cumsum(self.bgsteps, axis=1)
        self.bgtraj_clipped = np.fmod(self.bgtraj, self.texpars.noiselim[0])

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

        texturebasic = snd.convolve(noisefield, gfilter, mode='wrap')
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
                if (i < filterwidth * 2 + 1) and (j < filterwidth * 2 + 1):
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
        # This will be the full size 800x800 with repeats
        self.texturewithloopsmaxi = texture
        # Save a smaller version for compatibility
        self.texturewithloops = texture[::bgstixel, ::bgstixel]

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

    def generatecontrast(self, coord, window=0, pad_length=0):
        """
        Returns the contrast value for a particular coordinate throughout
        the whole experiment, padded with zeros in the beginning to fit
        the rolling_window function. The coordinates are rounded to be used as
        indices.

        Padding is needed because rolling window reduces the size of the
        temporal axis by it's "window" parameter. The option "preserve_dims"
        leads to doubling in memory so we solve it by generating the array
        with zeros already there.

        Parameters
        --------
        coords:
            The coordinates for the center of the contrast to be generated.
            Origin should be lower (or upper?) left corner. So the center
            would be [100, 100] for default OMB parameters.

            Should be a list or numpy array of two elements.

        pad_length:
            The length of the padding in the beginning, corresponds to the
            window parameter rolling window function, or filter_length
            for STA in general.

        window:
            The resulting array will extend this many pixels in both
            directions around coord. If zero, size will be a single pixel.

        """
        if isinstance(coord, int) or len(coord) != 2:
            raise ValueError('coord is expected to have two elements.')
        coord = np.array(coord)

        texture = self.texture_flipped
        # Use the clipped trajectory in case the
        # texture goes out of the central region.
        traj = self.bgtraj_clipped.copy()
        # Flip the y axis because texture is also flipped
        traj[1, :] *= -1
        # swap x and y
        traj = np.flipud(traj)

        contrast = np.zeros((window*2+1, window*2+1,
                             self.ntotal+pad_length), dtype=np.float32)
        if pad_length != 0:
            traj = np.concatenate((np.zeros((2, pad_length)), traj), axis=-1)

        for i, ii in enumerate(range(-window, window+1)):
            for j, jj in enumerate(range(-window, window+1)):
                traj_loop = np.round(-traj
                            + coord[..., None]
                            # HINT: center the texture
                            + self.texpars.noiselim[:, None]
                            + np.array([ii, jj])[..., None]).astype(int)
                traj_loop = np.fmod(traj_loop, texture.shape[0])
                contrast[i, j] = texture[traj_loop[0], traj_loop[1]]
        return contrast

    def generatecontrastmaxi(self, coord, window=0, pad_length=0):
        """
        Generate texture using each pixel, to minimize rounding errors

        although norma apparently does it based on stixels

        coordinate is the bottom (?) left corner
        window is the full size of the texture (not like radius around)
        """
        coord = np.array(coord)
        # Movement in x direction corresponds to translation in left/right
        # axis; and y to up/down axis. Since the first index is the row,
        # we need swap x and y if we want to keep the order (x,y).
        coord = np.flipud(coord)

        fieldsize = np.array([self.pars.squareheight, self.pars.squarewidth])
        # Use the clipped trajectory in case the
        # texture goes out of the central region.
        traj = np.fmod(self.bgtraj*self.pars.bgstixel, fieldsize[:, None])

        contrast = np.zeros((window, window, self.ntotal+pad_length))

        if pad_length != 0:
            traj = np.concatenate((np.zeros((2, pad_length)), traj), axis=-1)

        if not hasattr(self, 'texture_withloops'):
            self._generatetexture_withloops()
        texture = self.texturewithloopsmaxi

        for i, ii in enumerate(range(window)):
            for j, jj in enumerate(range(window)):
                traj_loop = np.round(traj
                            + coord[..., None]
                            # HINT: center the texture
                            + np.array([ii, jj])[..., None]).astype(int)
                traj_loop = np.fmod(traj_loop, fieldsize[:, None])
                contrast[i, j] = texture[traj_loop[0], traj_loop[1]]
#                contrast[i, j, :] = np.take(texture, traj_loop, mode='wrap')
        return contrast

    def regioncontrast(self, coord, window):
        """
        Returns the average contrast values for a group of coordinates.
        """
        import miscfuncs as msc

        msc.cut_around_center(self.texture)

    def read_texture_analysis(self):
        # In case of multiple texture data files, find the one with most
        # number of frames.
        files = list(self.stim_dir.glob(f'{self.stimnr}_texturesta_*'))

        newlist = []
        for file in files:
            newlist.append(file.stem.split('_')[-1].split('fr')[0])
        newlist = np.array(newlist, dtype=np.int)
        texturefile = files[np.argsort(newlist)[-1]]

        return np.load(self.stim_dir / texturefile)

    def contrast_signal_cell(self, cell_i, *args):
        """
        Returns the contrast signal for a cell based on location of
        the maximum pixel in its texture STA.
        """
        texture_maxi = self.read_texture_analysis()['texture_maxi']
        contrast_signal = self.generatecontrast(texture_maxi[cell_i], *args)
        return contrast_signal

    def show_texture_stas(self):
        """
        Returns an overview of all texture STAs with the detected receptive
        field centers marked.
        """
        texturedata = self.read_texture_analysis()
        fig, sl = plf.multistabrowser(texturedata['texturestas'], self.frame_duration,
                                      cmap='Greys_r')
        coords = texturedata['texture_maxi']
        for i in range(self.nclusters):
            ax = fig.axes[i]
            ax.plot(*coords[i][::-1], 'r+', markersize=10, alpha=.2)
        return fig, sl


#%%
if __name__ == '__main__':

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
    contrast = st.generatecontrast(st.texpars.noiselim/2, 100, 19)
    contrast_avg = contrast.mean(axis=-1)

    rw = asc.rolling_window(contrast, st.filter_length, preserve_dim=False)

    all_spikes = np.zeros((st.nclusters, st.ntotal))
    for i in range(st.nclusters):
        all_spikes[i, :] = st.binnedspiketimes(i)

    stas = np.einsum('abcd,ec->eabd', rw, all_spikes)
    stas /= all_spikes.sum(axis=(-1))[:, np.newaxis, np.newaxis, np.newaxis]

    # Correct for the non-informative parts of the stimulus
    stas = stas - contrast_avg[None, ..., None]

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
