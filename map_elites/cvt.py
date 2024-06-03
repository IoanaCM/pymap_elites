#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.

import math
import numpy as np
import torch
import torch.multiprocessing as mpt
import multiprocessing as mp
import os

# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

from . import common as cm



def __add_to_archive(s, centroid, archive, kdt):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche)
    s.centroid = n
    if n in archive:
        scalar = np.isscalar(s.fitness)
        comp = s.fitness > archive[n].fitness
        if scalar and comp or not scalar and comp[0]:
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s
        return 1


# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def __evaluate(x):
    z, f = x  # evaluate z with function f
    fit, desc = f(z)
    return cm.Species(z, desc, fit)

# map-elites algorithm (CVT variant)
def compute(dim_map, dim_x, f,
            n_niches=1000,
            max_evals=1e5,
            params=cm.default_params,
            log_dir=None,
            log_file=None,
            variation_operator=cm.variation):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

    """
    
    # create log folder
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_file)
    
    log_file = open(log_file, 'w')
    
    pool_size = params.get('pool_size', mp.cpu_count())
    print(f'Num Cores - {pool_size}')
    
    pool = None
    if params.get('gpu', False):
        print('Using gpu')
        pool = mpt.Pool(pool_size)
    else:
        pool = mp.Pool(pool_size)

    # create the CVT
    c = cm.cvt(n_niches, dim_map,
              params['cvt_samples'], params['cvt_use_cache'], log_dir=log_dir)
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c, log_dir=log_dir)

    archive = {} # init archive (empty)
    n_evals = 0 # number of evaluations since the beginning
    b_evals = 0 # number evaluation since the last dump

    # main loop
    while (n_evals < max_evals):
        log_file.write(f'Eval: {n_evals}\n')
        to_evaluate = []
        # random initialization
        if len(archive) <= params['random_init'] * n_niches:
            for _ in range(0, params['random_init_batch']):
                n = np.random.randint(1, dim_x)
                idx = np.sort(np.random.choice(range(0, dim_x), n, replace=False))
                x = np.zeros(shape=dim_x, dtype=int)
                x[idx] = 1
                to_evaluate += [(x, f)]
            log_file.write(f'Random init archive\n')
        else:  # variation/selection loop
            keys = list(archive.keys())
            log_file.write(f'Keys: {keys}\n')
            log_file.flush()
            fits = np.array([archive[k].fitness[0] for k in keys])
            min_fit = np.min(fits)
            to_add = 0 if min_fit > 0 else (1 - min_fit)
            fits = np.array([x + to_add for x in fits])
            tot_fit = np.sum(fits)
            fits = fits / tot_fit
            # we select all the parents at the same time because randint is slow
            rand1 = np.random.choice(range(len(keys)), size=params['batch_size'], p=fits)
            rand2 = np.random.choice(range(len(keys)), size=params['batch_size'], p=fits)
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[rand1[n]]]
                y = archive[keys[rand2[n]]]
                # copy & add variation
                z = variation_operator(x.x, y.x, params)
                to_evaluate += [(z, f)]
        # evaluation of the fitness for to_evaluate
        s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)
        log_file.write(f'Finished eval\n')
        # natural selection
        log_file.write(f'Adding to archive {len(s_list)} solutions: {[(s.desc, s.fitness) for s in s_list]}\n')
        for s in s_list:
            if s.desc is not None and s.fitness is not None:
                __add_to_archive(s, s.desc, archive, kdt)
        # count evals
        n_evals += len(to_evaluate)
        b_evals += len(to_evaluate)

        # write archive
        if b_evals >= params['dump_period'] and params['dump_period'] != -1:
            print("[{}/{}]".format(n_evals, int(max_evals)), end=" ", flush=True)
            cm.__save_archive(archive, n_evals, log_dir)
            b_evals = 0
        # write log
        if log_file != None:
            fit_list = np.array([x.fitness for x in archive.values()])
            log_file.write("{} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()),
                    fit_list.max(axis=0), np.mean(fit_list,axis=0), np.median(fit_list,axis=0),
                    np.percentile(fit_list, 5,axis=0), np.percentile(fit_list, 95,axis=0)))
            log_file.flush()
    cm.__save_archive(archive, n_evals, log_dir)
    return archive
