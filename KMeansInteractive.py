#!/usr/bin/python
# -*- coding: utf-8 -*-



def plot_kmeans_interactive(X, y, min_clusters=1, max_clusters=6):
    from IPython.html.widgets import interact
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.datasets.samples_generator import make_blobs
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
 
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        '''
        X, y = make_blobs(n_samples=300, centers=4,
                          random_state=0, cluster_std=0.60)
        '''
        def _kmeans_step(frame=0, n_clusters=2):

            rng = np.random.RandomState(2)
            labels = np.zeros(X.shape[0])
            centers = 100+50*rng.randn(n_clusters, 2)

            nsteps = frame // 3

            for i in range(nsteps + 1):
                old_centers = centers
                if i < nsteps or frame % 3 > 0:
                    dist = euclidean_distances(X, centers)
                    labels = dist.argmin(1)

                if i < nsteps or frame % 3 > 1:
                    centers = np.array([X[labels == j].mean(0)
                                        for j in range(n_clusters)])
                    nans = np.isnan(centers)
                    centers[nans] = old_centers[nans]


            # plot the data and cluster centers
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='RdYlGn',
                        vmin=0, vmax=n_clusters - 1);
            plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                        c=np.arange(n_clusters),
                        s=200, cmap='RdYlGn')
            plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                        c='black', s=50)

            # plot new centers if third frame
            if frame % 3 == 2:
                for i in range(n_clusters):
                    plt.annotate('', centers[i], old_centers[i], 
                                 arrowprops=dict(arrowstyle='->', linewidth=1))
                plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c=np.arange(n_clusters),
                            s=200, cmap='rainbow')
                plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c='black', s=50)

            plt.xlim(-50, 300)
            plt.ylim(0, 350)
            plt.xlabel(u'freies mtl. Einkommen [EUR]')
            plt.ylabel(u'angefragte mtl. Kreditrate [EUR]')

            if frame % 3 == 0:
                plt.title(u'1. Neuer Cluster Schwerpunkt')
            elif frame % 3 == 1:
                plt.title(u"2. Zuordnen aller Punkte zum nächstgelegendsten Schwerpunkt", size=14)
            elif frame % 3 == 2:
                plt.title(u"3. Verschiebe Schwerpunkt zum Cluster-Mittelwert", size=14)

    
    return interact(_kmeans_step, frame=[0, 50],
                    n_clusters=[min_clusters, max_clusters])