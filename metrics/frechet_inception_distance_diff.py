# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils
import os 
#----------------------------------------------------------------------------

def compute_diff(opts, max_real, num_gen, swav=False, sfid=False):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    # mu_real, sigma_real =  metric_utils.compute_feature_stats_for_generator(
    #     opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
    #     rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, swav=swav, sfid=sfid).get_mean_cov()

    
    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, swav=swav, sfid=sfid).get_mean_cov()

    output_path = opts.run_dir + '/back_fid.npz'
    # print(output_path)/
    # print(os.path.exists(output_path))
    if os.path.exists(output_path):
    #     print(" 我进来了")
        feature = np.load(output_path)
        
        mu_real = feature['mu']
        sigma_real = feature['sigma']
    else:
        mu_real = mu_gen
        sigma_real = sigma_gen
    np.savez_compressed(output_path, mu=mu_gen, sigma=sigma_gen)


    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    # fid = np.real(m)
    return float(fid)

#----------------------------------------------------------------------------
