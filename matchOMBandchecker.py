#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt

import analysis_scripts as asc
import iofuncs as iof



def read_and_match_pars(exp, omb_stimnr, chk_stimnr):
    OMB_DEFAULT_TEXTURESIZE = np.array([800, 800])

    _, metadata = asc.read_spikesheet(exp)

    scr_x, scr_y = metadata['screen_width'], metadata['screen_height']

    #datao = iof.load(exp, omb_stimnr)
    #datac = iof.load(exp, checker_stimnr)

    parso = asc.read_parameters(exp, omb_stimnr)
    parsc = asc.read_parameters(exp, chk_stimnr)

    chk_stxw, chk_stxh = [parsc[key] for key in ['stixelwidth', 'stixelheight']]

    if (not chk_stxw == chk_stxh or
       parsc['stimulus_type'] not in ['FrozenNoise', 'checkerflicker']):
        ValueError(f'{iof.getstimname(exp, chk_stimnr)} is not '
                     'checkerflicker!')

#    ckdim = scr_x//chk_stxw
#    omb_dim = np.array(OMB_DEFAULT_TEXTURESIZE)//parso['bgstixel']

    ombstx = parso['bgstixel']
    chkstx = chk_stxw

    # Calculate the difference between the dimensions for both stimuli in pixels
    px_diff = OMB_DEFAULT_TEXTURESIZE - np.array([scr_x, scr_y])

    # We divide by two because the texture is centered in the screen and
    # clipped.
    px_diff //= 2

    # HINT
    px_diff = px_diff[::-1]

    return ombstx, chkstx, px_diff


def coord_omb2chk(omb_coord, exp, omb_stimnr, chk_stimnr):
    """
    Convert OMB coordinates to checkerflicker to match the locations
    from both stimuli.

    Parameters
    ------
    omb_coord:
        (x, y) coordinates in the OMB texture

    omb_stimnr:
        The stimulus number for OMB
    chk_stimnr:
        The stimulus number for checkerflicker

    Returns
    ------
    chk_coord:
        Coordinates in the checkerflicker stimulus
    """
    ombstx, chkstx, px_diff = read_and_match_pars(exp, omb_stimnr, chk_stimnr)
    coord = omb_coord * ombstx
    coord -= px_diff
    coord = (np.round(coord / chkstx)).astype(int)
    return coord


def coord_chk2omb(chk_coord, exp, omb_stimnr, chk_stimnr):
    """
    Convert checkerflicker coordinates to OMB to match the locations
    from both stimuli.

    Parameters
    ------
    chk_coord:
        (x, y) coordinates in the checkerflicker stimulus
    omb_stimnr:
        The stimulus number for OMB
    chk_stimnr:
        The stimulus number for checkerflicker

    Returns
    ------
    chk_coord:
        Coordinates in the OMB texture
    """
    ombstx, chkstx, px_diff = read_and_match_pars(exp, omb_stimnr, chk_stimnr)
    coord = chk_coord * chkstx
    coord += px_diff
    coord = (np.round(coord / ombstx)).astype(int)
    return coord


def chkmax2ombcoord(cell_i, exp, omb_stimnr, chk_stimnr):
    """
    Starting from the index of the cell, return the coordinates of the
    maximum pixel of checkerflicker STA in OMB coordinates.

    Parameters
    ---------
    cell_i
        Cell index
    """
    data = iof.load(exp, chk_stimnr)
    maxinds = np.array(data['max_inds'])

    coord = coord_chk2omb(maxinds[cell_i][:-1], exp, omb_stimnr, chk_stimnr)
    return coord

#%%
if __name__ == '__main__':
    exp, omb_stimnr = '20180710_YE', 8
    chk_stimnr = 6

    omstx, ckstx, _ = read_and_match_pars(exp, omb_stimnr, chk_stimnr)
    #ckstx = parsc['stixelwidth']
    #omstx = parso['bgstixel']
    test_coords_omb = np.array([[400, 300], [600, 300], [400, 250],
                                [800, 600], [0, 0]])/omstx
    res_coords_chk = np.empty(test_coords_omb.shape)

    for i, test_coord in enumerate(test_coords_omb):
        tochk = coord_omb2chk(test_coord, exp, omb_stimnr, chk_stimnr)
        res_coords_chk[i] = tochk
        tochktoomb = coord_chk2omb(tochk, exp, omb_stimnr, chk_stimnr)
        print(test_coord*omstx, test_coord, tochk, tochk*ckstx, tochktoomb, tochktoomb*omstx)