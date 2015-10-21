# -*- coding: utf-8 -*-
"""
Created on Tue Oct 06 18:44:49 2015

@author: Igor Stankevich, loknar@list.ru

The MIT License (MIT)

Copyright (c) 2015 Igor Stankevich, loknar@list.ru

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE

"""
import logging, threading, time, importlib, os, cPickle, math, ctypes
import numpy as np
import pandas as pd
from hashlib import sha1
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.externals import joblib
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from logging import handlers
from scipy.stats import randint, uniform
import multiprocessing as mp

class main_logger:

    def __init__(self, filename = 'blender.log', debug = True):
        self.log = logging.getLogger('/logs/' + filename)
        if debug:
            loglevel = logging.DEBUG
        else:
            loglevel = logging.INFO
        self.log.setLevel(loglevel)
        self.bus = None
        handler = handlers.TimedRotatingFileHandler(filename, when = 'midnight', utc = True)
        handler.setLevel(loglevel)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(threadName)-10s %(message)s')
        handler.setFormatter(formatter)
        self.log.addHandler(handler)
        self.log.propagate = False

    def __call__(self, log_type, module, message, *args, **kwargs):
        '''
        logs data
        log_type: info, debug, error, critical - methods of logger
        module: name of the module called the error
        message: error message
        '''
        method = getattr(self.log, log_type)
        method(module + ': ' + message, *args, **kwargs)

def cache_key(*args):
    key = ''
    for arg in args:
        key += str(arg)
    return sha1(key).hexdigest()

class Blender(object):
    '''
    Class for computing blended model, selecting predictors according to blended loss function.
    On initialization forms 2 major sets: 
        X_major_train - this will be used for cross validation training of all classifiers
        X_major_test - this will be used as predicting set for trained classifiers, to test 
        logistic regression predictions
    Inputs:
        X (features' vectors) and y (target vector)
        init_values - value first constant column for logistic regression, default 'auto'
            if 'auto' - uses most most frequent value from target vector
        classifiers - list of classifiers, each classifier is dict of the following format:
            'name': classificator_name to import, for example "sklearn.ensemble.RandomForestClassifier"
            'kwargs': arguments distributions dictionary, for example:
                 {
                     "n_estimators": randint(50, 200),
                     "criterion": ["gini", "entropy"],
                     "max_features": randint(1, 50),
                     "max_depth": [1, 3, 7, 10, None],
                     "min_samples_split": randint(1, 11),
                     "min_samples_leaf": randint(1, 11),
                     "bootstrap": [True, False]
                 }
                 distribution can be either scipy RVS method or list, to sample from
        test_split - ratio of train/test split
        verbose - False for no comments, True for some information within process
    '''
    
    def __init__(self, X, y, init_value = 'auto', classifiers = [], save_path = '',
                 test_split = 0.3, verbose = True, njobs = -1):
        self.logger = main_logger()
        self.verbose = verbose
        self.X = X
        self.y = y
        self.test_split = test_split
        self.classifiers = self.__init_classifiers(classifiers)
        self.save_path = save_path
        self.pickler = Pickler(save_path)
        self.evaluator = Evaluator(clf_path = save_path)
        self.best_score = 0.0
        self.njobs = mp.cpu_count() if njobs == -1 else njobs
        # splitting initial data for blend results train/test
        self.X_major_train, self.X_major_test, self.y_major_train, self.y_major_test = train_test_split(X, y, test_size=test_split, random_state=42)
        # setting up folds for classifier training
        self.blended_train_X, self.blended_test_X = self.__init_blend(init_value)
        # adding classifiers who has parallel mode
        self.parallel_classifiers = ['RandomForestClassifier', 'ExtraTreesClassifier', 'XGBClassifier']
        
    def __init_blend(self, init_value):
        return None, None

    def __init_blend_old(self, init_value):
        '''
        initializes Blend arrays with constants
        shape of train array is same as Major train array
        shape of test array is same as Major test array
        '''
        if init_value == 'auto':
            init_value = np.bincount(self.y.astype(np.int32)).argmax()
        elif type(init_value) == int:
            pass
        else:
            raise ValueError('init_value should be "auto" or INT')
        train = np.zeros((self.X_major_train.shape[0], 1))
        test = np.zeros((self.X_major_test.shape[0], 1))
        if init_value != 0:
            train.fill(init_value)
            test.fill(init_value)
        return train, test
        
    def __init_classifiers(self, clfs):
        if clfs:
            return clfs
        else:
            return self.load_classifiers()
        
    def add_classifier(self, clf):
        self.classifiers.append(clf)
        
    def random_classifier(self, clf_name = None):
        '''
        samples classifier and its parameters from saved distributions
        '''
        if clf_name is None:
            # generating weights distribution to sample classifier from
            dist = np.array([clf['weight'] for clf in self.classifiers])
            # normalizing weights to probability mass function
            dist = dist/dist.sum()
            # sampling classifier index from generated distribution
            choice = dist.cumsum().searchsorted(np.random.sample(1))[0]
            clf = self.classifiers[choice]
        else:
            for c in self.classifiers:
                if c['clf'] == clf_name:
                    clf = c
                    break
        kwargs = {}
        for arg in clf['kwargs']:
            if hasattr(clf['kwargs'][arg], 'rvs'):
                value = clf['kwargs'][arg].rvs()
            elif type(clf['kwargs'][arg]) == list:
                value = np.random.choice(clf['kwargs'][arg])
            else:
                raise ValueError('Incorrect argument distribution supplied for argument %s, classificator %s' % (arg, clf['clf']))
            # correcting maximum amount of features for Trees classifiers
            if arg == 'max_features' and type(value) == int and self.X.shape[1] < value and clf['clf'] != 'GradientBoostingClassifier':
                value = self.X.shape[1]
            # correcting subsample for GB classifier
            if (clf['clf'] == 'GradientBoostingClassifier' or clf['clf'] == 'XGBClassifier')\
            and (arg == 'subsample' or arg == 'colsample_bytree') and value > 1.0:
                value = 1.0
            kwargs[arg] = value
        # if classifier does not have parallel mode, returning True forcing Blender to run in parallel mode
        parallel = False if clf['clf'] in self.parallel_classifiers else True
        return clf['lib'], clf['clf'], kwargs, parallel
    
    def reinit_classifiers(self):
        '''
        scans for .ini file to check if there are new classifiers added
        not implemented yet
        '''
        pass

    def load_classifiers(self):
        '''
        loads classifiers
        loading from ini file not implemented yet
        each classifier have following parameters:
            lib - lib to be imported before classifier in case import lib+classifier doesn't work for you
            clf - classifier name
            weight - weight to be used in random selection, compared with weights of other classifiers
            kwargs - classifier arguments to be randomly sampled
        '''
        clfs = [
                {
                 'lib': 'sklearn.ensemble',
                 'clf': 'RandomForestClassifier',
                 'weight': 8.,
                 'kwargs': {
                            "n_estimators": randint(2000, 10000),
                            "criterion": ["gini", "entropy"],
                            "max_features": ['auto'],
                            "max_depth": [None],
                            "min_samples_split": [2],
                            "min_samples_leaf": [1],
                            "bootstrap": [True, False],
                            "n_jobs": [-1],
                           }
                },
                {
                 'lib': 'xgboost',
                 'clf': 'XGBClassifier',
                 'weight': 24.,
                 'kwargs': {
                            "max_depth": randint(10, 25),
                            "learning_rate": uniform(loc = 0.005, scale = 0.2),
                            "n_estimators": randint(4000, 15000),
                            #"gamma": uniform(loc = 0., scale = 0.2),
                            "subsample": uniform(loc = 0.4, scale = 0.8),
                            "colsample_bytree": uniform(loc = 0.4, scale = 0.8),
                            "seed": randint(1, 10000),
                            "min_child_weight": [3],
                            "silent": [True],
                           }
                },
                {
                 'lib': 'sklearn.ensemble',
                 'clf': 'ExtraTreesClassifier',
                 'weight': 8.,
                 'kwargs': {
                            "n_estimators": randint(2000, 10000),
                            "criterion": ["gini", "entropy"],
                            "max_features": ['auto'],
                            "max_depth": [None],
                            "min_samples_split": [2],
                            "min_samples_leaf": [1],
                            "bootstrap": [True, False],
                            "n_jobs": [-1],
                           }
                },
                {
                 'lib': 'sklearn.ensemble',
                 'clf': 'GradientBoostingClassifier',
                 'weight': 2.,
                 'kwargs': {
                            "loss": ["deviance", "exponential"],
                            "learning_rate": uniform(loc = 0.05, scale = 0.2),
                            "n_estimators": randint(50, 200),
                            "max_depth": randint(1, 11),
                            "min_samples_split": [2],
                            "min_samples_leaf": [1],
                            "subsample": uniform(loc = 0.1, scale = 1.5),
                            "max_features": ["sqrt", "log2"],
                           }
                },
                {
                 'lib': 'sklearn.ensemble',
                 'clf': 'AdaBoostClassifier',
                 'weight': 1.,
                 'kwargs': {
                            "learning_rate": [1.],
                            "n_estimators": randint(30, 100),
                            "algorithm": ["SAMME", "SAMME.R"],
                           }
                },
        ]
        return clfs
       
    def run(self):
        self.running = True
        def worker():
            while self.running:
                try:
                    self.process()
                except Exception as e:
                    self.logger('error', 'main loop', str(e), exc_info = True)
                self.reinit_classifiers()
                time.sleep(1)
            self.logger('info', 'main worker', 'main thread finished')
        self.thread = threading.Thread(target = worker)
        self.thread.start()
        
    def train_clfs(self, clfs):
        '''
        method uses supplied classifiers to use in blender
        classifiers should have following methods:
            fit - to train classifier, 
            predict - to predict classes,
            predict_proba - to predict probabilities of classes
        '''
        self.folds = list(StratifiedKFold(self.y_major_train, 5, shuffle=True))
        for clf in clfs:
            self.blend(clf)
    
    def stop(self):
        self.running = False
        self.thread.join()
        
    def __del__(self):
        self.stop()
        
    def process(self):
        '''
        methods chooses one of classifiers randomly, fits classifier on all folds,
        cross validates the folds and use predictions to train blender
        then blender being tested for new score, if new score is better than old one
        new classifier and its results are added to blender
        '''
        # randomly selecting classififer and its init parameters
        clf_lib, clf_name, kwargs, parallel = self.random_classifier()
        # initializing training folds
        self.folds = list(StratifiedKFold(self.y_major_train, 5, shuffle=True))
        clf = self.init_clf(clf_lib, clf_name, kwargs)
        self.blend(clf, parallel)
    
    def init_clf(self, clf_lib, clf_name, kwargs):
        # importing classifier's class
        if clf_lib:
            lib = importlib.import_module(clf_lib)
            clf_class = getattr(lib, clf_name)
        else:
            clf_class = importlib.import_module(clf_name)
        # instantiating and returning classifier
        return clf_class(**kwargs)
        
    def blend(self, clf, parallel = False):
        if self.verbose:
            print 'Training base predictors...'
            number = self.blended_train_X.shape[1] if self.blended_train_X is not None else 0
            print number, clf
        if parallel:
            if self.verbose:
                print 'Classifier %s is using single CPU, starting %s processes for it' % (clf.__class__.__name__, str(self.njobs))
            self.parallel_blend(clf)
        else:
            blend_train_X, blend_test_X, trained_clfs = form_blend_data(self.X_major_train, self.X_major_test,
                                                                        self.y_major_train, self.folds, clf, self.verbose)
            # stacking saved blended train and test data with new column
            temp_train_X, temp_test_X = self.stack_blends(blend_train_X, blend_test_X)
            self.check_classifier_result(temp_train_X, temp_test_X, trained_clfs)
            
    def stack_blends(self, blend_train_X, blend_test_X):
        if self.blended_train_X is not None:
            temp_train_X = np.hstack((self.blended_train_X, blend_train_X.reshape(blend_train_X.shape[0], 1)))
        else:
            temp_train_X = blend_train_X.reshape(blend_train_X.shape[0], 1)
        if self.blended_test_X is not None:
            temp_test_X = np.hstack((self.blended_test_X, blend_test_X.reshape(blend_test_X.shape[0], 1)))
        else:
            temp_test_X = blend_test_X.reshape(blend_test_X.shape[0], 1)
        return temp_train_X, temp_test_X
            
    def parallel_blend(self, clf):
        '''
        launches few same classifiers in parallel
        then collects their data
        '''
        Q = mp.Queue()
        processes = []
        for i in xrange(self.njobs):
            if i > 0:
                clf_lib, clf_name, kwargs, parallel = self.random_classifier(clf.__class__.__name__)
                clf = self.init_clf(clf_lib, clf_name, kwargs)
            p = mp.Process(target = mp_worker, args = (Q, self.X_major_train, self.X_major_test,
                                                       self.y_major_train, self.folds, clf, self.verbose))
            p.start()
            processes.append(p)
        t_start = round(time.time(), 0)
        informed = False
        divider = 60
        while True:
            if Q.qsize() == len(processes):
                if self.verbose:
                    print 'Parallel training finished, collecting data and checking classifiers...'
                for i in xrange(len(processes)):
                    blend_train_X, blend_test_X, trained_clfs = Q.get()
                    temp_train_X, temp_test_X = self.stack_blends(blend_train_X, blend_test_X)
                    self.check_classifier_result(temp_train_X, temp_test_X, trained_clfs)
                if self.verbose:
                    print 'Joining processes...'
                for p in processes:
                    p.join()
                break
            t_now = round(time.time(), 0)
            time_passed = t_now - t_start
            live_processes = len(processes) - Q.qsize()
            if self.verbose and not informed and time_passed % divider == 0 and time_passed > 0:
                mins = math.floor(time_passed / 60.0)
                secs = time_passed - mins * 60
                print '%s processes left running, training for %imin %isec...' % (str(live_processes), int(mins), int(secs))
                informed = True
                if divider <= 1200:
                    divider *= 1.5
                else:
                    divider = 1200
            if time_passed % divider != 0:
                informed = False
            time.sleep(0.5)
        # pausing to correctly display everything from processes on windows
        time.sleep(1)

    def check_classifier_result(self, train_X, test_X, trained_clfs):
        model_index = self.blended_train_X.shape[1] if self.blended_train_X is not None else 0
        if self.verbose:
            print 'Training blender...'
        blender = LogisticRegression()
        # fitting blend classifier with collected train data
        blender.fit(train_X, self.y_major_train)
        if self.verbose:
            print 'Logistic regression coefs: %s' % str(blender.coef_)
        # predicting with blender on blended TEST data
        Y_test_predict = blender.predict(test_X)
        Y_test_predict_proba = blender.predict_proba(test_X)[:,1]
        # comparing predicted and original targets
        score = metrics.accuracy_score(self.y_major_test, Y_test_predict)
        roc_auc_score = metrics.roc_auc_score(self.y_major_test, Y_test_predict_proba)
        print 'Current score: %s (ROC AUC: %s), previous best score: %s' % (str(score), str(roc_auc_score), str(self.best_score))
        # if current score is better than previous - saving model and expanding current prediction results for training
        if roc_auc_score > self.best_score:
            if self.verbose:
                print '>>>>>>>>> Saving new score, classifiers, blender...'
            self.best_score = roc_auc_score
            self.pickler.save(trained_clfs, 'classifiers_' + str(model_index))
            self.pickler.save(blender, 'blender_' + str(model_index))
            self.pickler.save(self.folds, 'folds_' + str(model_index))
            self.blended_train_X = train_X
            self.blended_test_X = test_X

def form_blend_data(X_major_train, X_major_test, y_major_train, folds, clf, verbose = False):
    '''
    Trains classifier with cross validation on Major train set (X_major_train and y_major_train)
    Forms and returns 3 lists:
        1. cross validated results from classifiers on Major train set (X_major_train) as train data for logistic regression
        2. mean predictions on Major train set (X_major_test) as test data for logistic regression
        3. list of trained classifiers to save aterwards if they are good
    '''
    scores, aucs, clfs = [], [], []
    blend_train_X = np.zeros(X_major_train.shape[0])
    blend_test_X = np.zeros((X_major_test.shape[0], len(folds)))
    # if XGB - suppling matrix, doesnt accept pandas with not alphanumeric feature names
    if clf.__class__.__name__ == 'XGBClassifier' and isinstance(X_major_train, pd.DataFrame):
        X_major_train, X_major_test = X_major_train.as_matrix(), X_major_test.as_matrix()
    for j, (tri, cvi) in enumerate(folds):
        if isinstance(X_major_train, pd.DataFrame):
            X_train, X_test = X_major_train.iloc[tri], X_major_train.iloc[cvi]
        else:
            X_train, X_test = X_major_train[tri], X_major_train[cvi]
        y_train, y_test = y_major_train[tri], y_major_train[cvi]
        # cloning bulk classifier
        clf_ = clone(clf)
        # training classifier on train indexes from Major train set
        clf_.fit(X_train, y_train)
        if verbose:
            if not clf.__class__.__name__ == 'XGBClassifier' and isinstance(X_train, pd.DataFrame):
                print 'Top 10 features: %s' % str(X_train.columns[np.argsort(clf_.feature_importances_)[::-1][:10]].format())
        # predicting on test indexes from Major train set, asssigning predicted values in folds for blend training
        blend_train_X[cvi] = clf_.predict_proba(X_test)[:,1]
        # predicting with trained classifier on Major TEST set, 1 fold per column, to be averaged before return
        blend_test_X[:, j] = clf_.predict_proba(X_major_test)[:,1]
        # evaluating score of current classifier
        score = clf_.score(X_test, y_test)
        # evaluating ROC AUC score
        roc_auc_score = metrics.roc_auc_score(y_test, clf_.predict_proba(X_test)[:,1])
        scores.append(score)
        aucs.append(roc_auc_score)
        if verbose:
            print 'Fold #%s, Score: %s, AUC: %s' % (str(j), str(score), str(roc_auc_score))
        # saving classifier to return list
        clfs.append(clf_)
    print 'Classifier %s mean score: %s, mean auc: %s' % (str(clf.__class__.__name__), np.mean(scores), np.mean(aucs))
    return blend_train_X, blend_test_X.mean(1), clfs

def mp_worker(q, X_major_train, X_major_test, y_major_train, folds, clf, verbose = False):
    blend_train_X, blend_test_X, trained_clfs = form_blend_data(X_major_train, X_major_test,
                                                                y_major_train, folds, clf, verbose)
    q.put((blend_train_X, blend_test_X, trained_clfs))

class Evaluator(object):
    '''
    loads classifiers and saves predictions for supplied test set
    '''
    
    def __init__(self, clf_path = '', verbose = True, njobs = -1):
        self.logger = main_logger('evaluator.log')
        self.clfs = []
        self.verbose = verbose
        self.blender = None
        self.njobs = mp.cpu_count() if njobs == -1 else njobs
        self.pickler = Pickler(clf_path)
        
    def process(self, X, y = None, predict_probability = True):
        blender = self.pickler.load('blender_' + str(self.count_saved_classifiers()-1))
        blend_inputs = self.predictors_data(X)
        predictions_prob = blender.predict_proba(blend_inputs)[:,1]
        predictions = blender.predict(blend_inputs)
        if y is not None:
            score = metrics.accuracy_score(y, predictions)
            roc_auc_score = metrics.roc_auc_score(y, predictions_prob)
            print 'Calculated score: %s' % str(score)
            print 'Calculated ROC AUC score: %s' % str(roc_auc_score)
        return predictions_prob if predict_probability else predictions
        
    def get_top_features(self, top = 10, clf_i = 0, top_features = {}):
        group = self.pickler.load('classifiers_' + str(clf_i))
        for i, clf in enumerate(group):
            args = np.argsort(clf.feature_importances_)[-top:][::-1]
            for arg in args:
                if arg not in top_features:
                    top_features[arg] = 1
                else:
                    top_features[arg] += 1
        print 'Best features and occurences:'
        print sorted(top_features.items(), key=lambda x: x[1], reverse = True)
        return top_features
            #print 'Top %s features: %s, args: %s' % (str(top), str(clf.feature_importances_[args]), str(args))

    def predictors_data(self, X):
        '''
        launches few same classifiers in parallel
        then collects their data
        '''
        clfs_amount = self.count_saved_classifiers()
        if self.verbose:
            print 'Sharing X...'
        # Defining shared memory array
        shared_array_base = mp.Array(ctypes.c_double, X.shape[0]*X.shape[1])
        shared_X = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_X = shared_X.reshape(X.shape)
        shared_X[:,:] = X
            
        def generate_data(clfs_amount, clf_path, verbose = True):
            for i in range(clfs_amount):
                yield (i, verbose, clf_path)

        if self.verbose:
            print 'Processing %s predictors with size %s parallel pool...' % (str(clfs_amount), str(self.njobs))
        iterable = generate_data(clfs_amount, self.pickler.clf_path, self.verbose)
        blend_inputs = np.zeros((X.shape[0], clfs_amount))
        pool = mp.Pool(self.njobs, initializer=_init, initargs=(shared_X, ))
        results = pool.map(eval_worker, iterable)
        pool.close()
        for i in range(clfs_amount):
            blend_inputs[:,i] = results[i]
        if self.verbose:
            print 'Predictors ready, returning results to blender'
        return blend_inputs
    
    def count_saved_classifiers(self):
        blenders = [ f for f in os.listdir(self.pickler.clf_path) if 'blender' in f and len(f.rsplit('.')) == 2 ]
        return len(blenders)

def _init(arr_to_eval_worker):
    """ Each pool process calls this initializer. Load the array to be populated into that process's global namespace """
    global arr
    arr = arr_to_eval_worker

def eval_worker(data):
    i, verbose, clf_path = data
    pickler = Pickler(clf_path)
    if verbose:
        print 'Loading %s' % 'classifiers_' + str(i)
    group = pickler.load('classifiers_' + str(i))
    X = arr
    if verbose:
        print 'Loaded %s' % group[0].__class__.__name__
    group_inputs = np.zeros((X.shape[0], len(group)))
    for j, clf in enumerate(group):
        group_inputs[:,j] = clf.predict_proba(X)[:,1]
    if verbose:
        print 'Calculated %s' % str(i)
    return group_inputs.mean(1)


class Pickler(object):
    '''
    helper class for parallel classificator loading
    '''
    def __init__(self, clf_path):
        self.clf_path = clf_path
    
    def file_exists(self, key):
        return os.path.isfile(self.form_filename(key))

    def form_filename(self, key, encoder = 'cPickle'):
        if encoder == 'joblib':
            return self.clf_path + key + '.pkl'
        elif encoder == 'cPickle':
            return self.clf_path + key + '.pickle'

    def save(self, obj, name, encoder = 'cPickle'):
        filename = self.form_filename(name)
        if encoder == 'joblib':
            joblib.dump(obj, filename)
        elif encoder == 'cPickle':
            with open(filename, 'wb') as f:
                cPickle.dump(obj, f)

    def load(self, name, encoder = 'cPickle'):
        filename = self.form_filename(name)
        if encoder == 'joblib':
            return joblib.load(filename)
        elif encoder == 'cPickle':
            with open(filename, 'rb') as f:
                return cPickle.load(f)


if __name__ == '__main__':
    pass

