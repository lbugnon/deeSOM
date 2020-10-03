# =========================================================================
# sinc(i) - http://fich.unl.edu.ar/sinc/
# Leandro Bugnon, Cristian Yones, Diego Milone and Georgina Stegmayer
# lbugnon@sinc.unl.edu.ar
# =========================================================================
import numpy as np
from sompy import SOMFactory
import time
from sklearn.base import BaseEstimator
import pickle

class _SOM():
    def __init__(self, train_data, train_labels, n_jobs, train_len, max_train_len, max_map_size, nsize, elastic_factor,
                 visualization_neighbour, verbosity, mapsize=None):
        """
        Train a SOM with a different labeling scheme. A one-positive-sample scheme is used: if one positive sample is
        contained in a SOM unit, the unit will be a positive one.
        :param train_data: numpy array NxM
        :param train_labels: numpy array N
        :param mapsize:
        """
        self.verbosity = verbosity
        self.visualization_neighbour = visualization_neighbour
        self.n_jobs = n_jobs

        if mapsize is None:
            n = max(2, int(min(max_map_size, int(np.sqrt(5 * train_data.shape[0] ** 0.54321) * nsize) *
                              elastic_factor)))
            mapsize = [n, n]
        self.mapsize = mapsize
        som = SOMFactory.build(train_data, mapsize, normalization=None)
        som.train(n_job=n_jobs, train_len_factor=train_len, verbose=None, maxtrainlen=max_train_len)
        self.som_labels = self.set_neighbour(som, mapsize[0], train_data, train_labels)
        self.codebook = som.codebook

    def set_neighbour(self, som, n, data, labels):
        """
        Extend the positive units labels to a neighbourhood (positive region).
        :param som: sompy.SOM model
        :param n: map size
        :param data: data to find best matching units
        :param labels: labels for data.
        :return: new labels for data
        """
        visualization_neighbour = self.visualization_neighbour

        idx_mi = np.where(labels == 1)[0]
        # Find best matching units for positives
        bmus1 = som.find_bmu(data[idx_mi], self.n_jobs).astype(int)[0]

        som_labels = np.zeros((n * n,), dtype=np.bool)
        som_labels[bmus1] = 1
        # Positive labels are spread trough a neighbourhood of width visualization_neighbour
        if visualization_neighbour is None:
            # distance in units
            visualization_neighbour = max(0, np.floor(np.log(n) / np.log(7)) -
                                          np.floor(np.exp(-abs(len(idx_mi) / 1e3 - 6) + 0.4)))
        # gaussian distance
        visualization_neighbour **= 2
        if visualization_neighbour > 0:
            dist_matrix = som._distance_matrix
            pos_units = np.where(som_labels == 1)[0]
            for l in pos_units:
                som_labels[dist_matrix[l, :] <= visualization_neighbour] = 1

        return som_labels

    def predict(self, data):
        """
        Use a SOM to label new data
        :param data: data to label
        :return: new labels for data
        """
        som = SOMFactory.build(np.zeros((1,1)), self.codebook.mapsize, normalization=None)
        som.codebook = self.codebook
        bmus = som.find_bmu(data, 4)
        labels = self.som_labels[bmus[0].astype(int)]
        return labels


class _SOMLayer():
    def __init__(self, train_data, train_labels, idxtrn, n_jobs, train_len, max_train_len, max_map_size, nsize,
                   elastic_factor, visualization_neighbour, verbosity, h):
        n_in = len(idxtrn)
        starth_time = time.time()

        self.som = _SOM(train_data[idxtrn, :], train_labels[idxtrn], n_jobs, train_len, max_train_len, max_map_size,
                       nsize, elastic_factor, visualization_neighbour, verbosity)
        resh = self.som.predict(train_data[idxtrn, :])

        # Update candidates for the next layer
        idxtrn = idxtrn[np.where(resh == 1)[0]]
        n_out = len(idxtrn)

        if verbosity:
            print("Layer=%03d \t layer_size=%03d \t n_inputs=%06d \t n_outputs=%06d \t (layer_time=%0.1f min)" %
                  (h, self.som.mapsize[0], n_in, n_out, (time.time() - starth_time) / 60), flush=True)
        self.out_index = idxtrn

    def predict(self, test_data):
        return self.som.predict(test_data)

class _SOMEnsembleLayer():
    def __init__(self, train_data, train_labels, idxtrn, n_jobs, train_len, max_train_len, max_map_size, nsize,
                   elastic_factor, target_imbalance, visualization_neighbour, verbosity, h=0):
        """
        Train SOMs in parallel. Positive cases can be presented to all layers, and unknown cases are split en equal parts.
         Output is the OR operation for all layers.
        """

        self.soms = []

        starth_time = time.time()
        posind = np.where(train_labels[idxtrn] == 1)[0]
        negind = np.where(train_labels[idxtrn] == 0)[0]
        NP = len(posind)
        NN = len(negind)
        idxtrnh0 = []

        # Define ensemble size to achieve imbalance target_imbalance.
        negatives_wanted = NP * target_imbalance
        Nhp = max(int(np.round(NN / negatives_wanted)), 2)

        idxtrn_new = []
        resh = np.zeros(train_labels[idxtrn].shape, dtype=bool)
        for h0 in range(Nhp):
            negindh0 = np.s_[int(h0 * NN / Nhp):int((h0 + 1) * NN / Nhp)]
            posindh0 = range(len(posind))
            idxtrn_part = idxtrn[np.append(posind[posindh0], negind[negindh0])]

            trdat = train_data[idxtrn_part, :]
            trlab = train_labels[idxtrn_part]
            starth0_time = time.time()
            som = _SOM(trdat, trlab, n_jobs, train_len, max_train_len, max_map_size, nsize, elastic_factor,
                      visualization_neighbour, verbosity)
            self.soms.append(som)

            resh0 = som.predict(trdat)
            n_out0 = sum(resh0)
            idxtrn_part = idxtrn_part[np.where(resh0 == 1)[0]]
            if len(idxtrn_new) < 1:
                idxtrn_new = idxtrn_part
            else:
                idxtrn_new = np.concatenate((idxtrn_new, idxtrn_part))

            if verbosity:
                print("ensemble_member=%d_%03d \t layer_size=%03d \t n_inputs=%06d \t n_outputs=%06d \t (layer_time=%0.1f min)" %
                      (h, h0, som.mapsize[0], trdat.shape[0], n_out0, (time.time() - starth_time) / 60), flush=True)

        n_in = len(idxtrn)
        idxtrn = np.array(list(set(idxtrn_new)))
        n_out = len(idxtrn)

        if verbosity:
            print("ensemble_layer=%d, n_inputs=%06d, n_outputs=%06d, (htime=%0.1f min)" %
                  (h, n_in, n_out, (time.time() - starth_time) / 60), flush=True)

        self.out_index = idxtrn

    def predict(self, test_data):
        """
        """
        resh = []
        for h0 in range(len(self.soms)):
            resh0 = self.soms[h0].predict(test_data)
            if len(resh) == 0:
                resh = resh0
            else:
                resh[np.where(resh0 == 1)[0]] == 1
        return resh


class DeeSOM(BaseEstimator):
    """DeeSOM is a library for highly-umbalanced class recognition based on self-organizing maps (SOM). """

    def __init__(self, verbosity=False, max_map_size=150, max_number_layers=100, visualization_neighbour=None, n_jobs=4,
                 train_len_factor=1, map_size_factor=1, max_train_len=200, elastic_threshold=0.97,
                 positive_th=1.02, target_imbalance=1500, elastic=True):
        """
        Classifier based on several layers of self-organizing maps (SOMs). ElasticSOM and  Ensemble-elasticSOM are
        implemented, as described in:

        "Deep neural architectures for highly imbalanced data in bioinformatics, L. A. Bugnon, C. Yones, D. H. Milone,
        G. Stegmayer, IEEE Transactions on Neural Networks and Learning Systems (2019)".

        Several hyperparameters are provided in the interface to easily explore in research. However, the main
        hyperparameters are: visualization_neighbour and target_imb

        :param verbosity: If verbosity>0, it shows  evolution of training via standard output
        ;param elastic: True set elastic behaviour.
        :param max_map_size_esmapsize: Maximum number of layers.
        :param max_number_layers: Maximum size of each layer.
        :param visualization_neighbour: This controls the size of the cluster that selects the samples at each layer
        that pass to the next layer. If None, it is automatically determined.
        :param n_jobs: Number of jobs (multithread).
        :param train_len_factor: Multiplying factor in (0,1] applied to number of training steps
        :param map_size_factor: Multiplying factor applied to the map size. 
        :param max_train_len: Maximum number of iterations per layer.
        :param elastic_threshold: Elastic algorithm is triggered if data reduction < (1-elastic_threshold)*100%
        :param positive_th: Stop training when the number of candidates is positive_th * training_positives
        :param target_imbalance: If set None, no ensemble layers are created (as in the elasticSOM model). Otherwise,
        ensemble layers are built to approximate the data imbalance of each layer to target_imbalance (as in the eeSOM
        model). If input imbalance is less than target_imbalance, the model behaviour is the same as if elasticSOM.
        """

        self.max_map_size = max_map_size
        self.visualization_neighbour = visualization_neighbour
        self.n_jobs = n_jobs
        self.train_len_factor = train_len_factor
        self.max_train_len = max_train_len
        self.map_size_factor = map_size_factor
        self.max_number_layers = max_number_layers
        self.elastic_threshold = elastic_threshold
        self.positive_th = positive_th
        self.target_imbalance = target_imbalance
        self.elastic = True
        self.verbosity = verbosity

        self.elastic_factor = 1
        self.layers = []
        

    def save_model(self, fname):
        config = [self.max_map_size, self.visualization_neighbour, self.train_len_factor, self.max_train_len, self.map_size_factor, self.max_number_layers,
                  self.elastic_threshold, self.positive_th, self.target_imbalance]
        pickle.dump([self.layers, config], open(fname, "wb"))

    def load_model(self, fname):
        self.layers, config = pickle.load(open(fname, "rb"))
        self.max_map_size, self.visualization_neighbour, self.train_len_factor, self.max_train_len, \
        self.map_size_factor, self.max_number_layers, self.elastic_threshold, self.positive_th, \
        self.target_imbalance = config


    def fit(self, train_data, train_labels):
        """
        Train a deep-SOM model, using several layers of SOMs.
        :param train_data: numpy.array NxM
        :param train_labels: numpy.array N
        :return:
        """
        start_time = time.time()
        n_data = train_data.shape[0]
        idxtrn = np.array(range(n_data))
        n_out = n_data

        n_posh = len(np.where(train_labels == 1)[0])
        h = 0
        if self.verbosity:
            print("\nStart training: n_samples=%d, n_positives=%d" % (n_data, n_posh), flush=True)
        self.data_proba = np.zeros(train_labels.shape)
        while (h < self.max_number_layers) and (n_out >= self.positive_th * n_posh):
            n_in = len(idxtrn)
            n_pos = len(np.where(train_labels[idxtrn] == 1)[0])
            n_neg = n_in - n_pos

            # imbalance-driven ensemble layer
            if self.target_imbalance > 0 and n_neg / n_pos >= 2 * self.target_imbalance:
                layer = _SOMEnsembleLayer(train_data, train_labels, idxtrn, self.n_jobs, self.train_len_factor,
                                                 self.max_train_len, self.max_map_size, self.map_size_factor, self.elastic_factor,
                                                 self.target_imbalance, self.visualization_neighbour, self.verbosity, h)
            else:
                layer = _SOMLayer(train_data, train_labels, idxtrn, self.n_jobs, self.train_len_factor, self.max_train_len,
                                         self.max_map_size, self.map_size_factor, self.elastic_factor,
                                         self.visualization_neighbour, self.verbosity, h)

            idxtrn = layer.out_index
            self.data_proba[idxtrn] += 1  # candidates ranking
            self.layers.append(layer)

            n_out = len(idxtrn)
            # Define elastic behaviour
            self.get_deesom_height(n_out, n_in)
            h += 1

        self.elastic_factor = 1
        if self.verbosity:
            print("(Total time=%0.1f min)" % ((time.time() - start_time) / 60), flush=True)
        return


    def predict_proba(self, test_data=None):
        """
        :param test_data:
        :return:
        """
        if test_data is None:
            return self.data_proba/np.max(self.data_proba)

        n_data = test_data.shape[0]
        idxtst = np.array(range(n_data))
        scores = np.zeros(n_data, dtype=np.int)
        h = 0
        while h < len(self.layers) and len(idxtst) > 0:
            resh = self.layers[h].predict(test_data[idxtst, :])

            idxtst = idxtst[np.where(resh == 1)[0]]
            scores[idxtst] += 1
            # slab[i]==n means sample i was discarded in layer n (thus it was considered as positive up to layer n-1)
            h += 1
        return scores/np.max(scores)


    def get_deesom_height(self, n_out, n_in):
        """ Calculate next SOM layer height using the elastic algorithm"""
        if self.elastic and n_out > self.elastic_threshold * n_in:
            self.elastic_factor *= 1.2


