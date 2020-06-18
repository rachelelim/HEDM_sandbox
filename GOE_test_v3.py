#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:21:40 2020

@author: rachel
"""

from __future__ import print_function

import time
import logging

import os

import glob

import multiprocessing

import numpy as np

from scipy import ndimage

import timeit

import argparse


try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml

from hexrd import constants as cnst
from hexrd import config
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd import instrument
from hexrd.findorientations import \
    generate_orientation_fibers, \
    run_cluster
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd.xrd import indexer

from hexrd import matrixutil as mutil

from hexrd.xrd          import rotations  as rot
from hexrd.xrd import indexer

import h5py

logger = logging.getLogger(__name__)


# images
def load_images(yml):
    return imageseries.open(yml, format="frame-cache", style="npz")


# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


def quaternion_ball(qref, radius, num=500):
    """
    Makes a ball of randomly sampled quaternions within the specified misorientation of the supplied reference.

    Parameters
    ----------
    qref : array_like, (4,)
        Unit quaternion defining the reference orientation.
    radius : scalar
        Maximum misorientation in degrees.
    num : int, optional
        The number of orientations to generate. The default is 500.

    Returns
    -------
    The (4, num) array of quaternions around qref.

    """
    # make random angle/axis pairs
    rand_angs = np.radians(radius)*np.random.rand(num)
    rand_axes = mutil.unitVector(np.random.randn(3, num))
   
    # form quats
    qball = rot.quatOfAngleAxis(rand_angs, rand_axes)
   
    # recenter around reference orientation
    qref_mat = rot.quatProductMatrix(
        np.atleast_1d(qref).reshape(4, 1), mult='right'
    ).squeeze()

    return np.dot(qref_mat, qball)



def test_GOE(cfg, quat, eta_ome, ome_tol=0.5, eta_tol=0.05):
    
    ncpus=cfg.multiprocessing

    compl = indexer.paintGrid(
        qball,
        eta_ome,
        etaRange=np.radians(cfg.find_orientations.eta.range),
        omeTol=np.radians(ome_tol),
        etaTol=np.radians(eta_tol),
        omePeriod=np.radians(cfg.find_orientations.omega.period),
        threshold=comp_thresh,
        doMultiProc=ncpus > 1,
        nCPUs=ncpus
        )
    
    return compl


def thresh_GOE(quats, test_compl, comp_thresh=0.5):
    keep = np.array(test_compl) > comp_thresh
    indices = np.arange(0,len(test_compl))[keep]
    GOE = quats[:,indices]
    compl = np.array(test_compl)[indices]
    return GOE,compl


def write_GOE(GOE_quats,compl,i, dump_dir):
    phi = 2*np.arccos(GOE_quats[0,:])
    n = xfcapi.unitRowVector(GOE_quats[1:,:])
    
    expmaps = phi*n
    
    ori_data = np.hstack([np.expand_dims(compl,axis=1),expmaps.T])
    
    delim = '   '
    header_items = ('# completeness', 'exp_map_c[0]', 'exp_map_c[1]', 'exp_map_c[2]')
    
    header = delim.join( np.tile('{:<12}', 4)
                ).format(*header_items)
    
    
    
    # print(output_str, file=
         
    
    fname = os.path.join(dump_dir,'grain_%04d.out'%i)
    
    
    
    f = open(fname, 'w+')
    print(header, file=f)
    for i in range(len(phi)):
        
        output_str = delim.join(['{:<12f}', '{:<12e}', '{:<12e}','{:<12e}']
             ).format(*ori_data[i,:])
        print(output_str, file=f)
    
    f.close()
    
    
    # np.savetxt(fname, ori_data, delimiter = '\t',header = 'Completeness \t\t xi[0] \t\t\t\t xi[1] \t\t\t\t xi[2] ')
    

#%%

cfg_file = '/Volumes/CalderaStorage/rachel/CHESS_Jun17/FF/2020-06-02/GOE/ti_init.yml'

instr_file = '/Volumes/CalderaStorage/rachel/CHESS_Jun17/FF/2020-06-02/ge_detector_06.yml'

mis_thresh = 1.0
comp_thresh = 0

ref_grains_out = '/Volumes/CalderaStorage/rachel/CHESS_Jun17/FF/2020-06-02/ti_index/grains.out'

data_dir = '/Volumes/CalderaStorage/rachel/CHESS_Jun17/FF/2020-06-02/'
fc_stem = 'ti7-1_scan00021_%s.npz'

dump_dir = '/Volumes/CalderaStorage/rachel/CHESS_Jun17/FF/2020-06-02/GOE/initial_GOE/'

active_hkls=[0,1,2,3,4]

#%%

cfg = config.open(cfg_file)[0]
pd = cfg.material.plane_data

instr = load_instrument(instr_file)
det_keys = instr.detectors.keys()

imsd = dict.fromkeys(det_keys)
for det_key in det_keys:
    fc_file = sorted(
        glob.glob(
            os.path.join(
                data_dir,
                fc_stem % det_key.lower()
            )
        )
    )
    if len(fc_file) != 1:
        raise(RuntimeError, 'cache file not found, or multiple found')
    else:
        ims = load_images(fc_file[0])
        imsd[det_key] = OmegaImageSeries(ims)


eta_ome = instrument.GenerateEtaOmeMaps(
        imsd, instr, pd,
        active_hkls=active_hkls,
        threshold=cfg.find_orientations.orientation_maps.threshold,
        ome_period=cfg.find_orientations.omega.period)


grains_data = np.loadtxt(ref_grains_out)
expmaps = grains_data[:, 3:6]
quats = rot.quatOfExpMap(expmaps.T)


#%%

for i in range(0,len(expmaps)):
    this_quat = quats[:,i]
    qball = quaternion_ball(this_quat, mis_thresh)
    test_compl = test_GOE(cfg, qball, eta_ome, eta_tol=0.25)
    GOE, compl = thresh_GOE(qball,test_compl, comp_thresh)
    
    print('Writing Grain %d' %i)
    write_GOE(GOE, compl, i, dump_dir)
    
    
    
    
    
    

