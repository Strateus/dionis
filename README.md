# Dionis
This package can be used to construct and blend randomly sampled models for Machine Learning.
Only predictors who help to raise final blender AUC score on test set are selected and saved, others skipped.

This package can be used for example for Kaggle competitions.

Dependencies:

	numpy, scikit-learn, scipy

Usage:

	from dionis import Blender
  	import numpy as np
  	
  	X = np.random.random((30000,1000)) # (n_samples, n_features)
  	y = np.random.random_integers(low = 0, high = 1, size = (30000,)) # targets
  	
  	blender = Blender(X, y, save_path = 'save/')

At start blender will split X and y to Major train and test sets. <b>Major train set</b> will be used for cross-validation training of all predictors, <b>Major test set</b> will be used to train blender itself.

	blender.run()

At this point blender will generate Cross Validation (CV) folds and randomly selects one of predefined models,
then it will randomly sample parameters of model distribution and start fitting model to targets.
After CV procedure is finished, it will save models in save_path directory.

To evaluate results or to generate submission, Evaluator class can be used:

	from dionis import Evaluator
	
	X = np.random.random((30000,1000)) # (n_samples, n_features)
	e = Evaluator(clf_path = 'save/', njobs = 4)

used without y (y = None) to generate predictions on test data

	predictions = e.process(X)

or can be used with y (targets) to check AUC score of predictions (for now only printed in stdout)

	predictions = e.process(X, y)

To submit results to Kaggle you can do something like this:

	IDs = np.random.random_integers(low = 0, high = 1, size = (30000,))
	res = np.vstack((IDs.values, predictions)).T
	np.savetxt('submission.csv', res, fmt=['%i', '%0.9f'], delimiter = ',', header = 'ID,target', comments = '')


Models, used in library, supplied by sklearn:

	RandomForestClassifier
	ExtraTreesClassifier
	GradientBoostingClassifier
	AdaBoostClassifier

GradientBoostingClassifier and AdaBoostClassifier are not parallel by nature, so Dionis will do parallelization himself
by launching separate training processes.

You can add your own classifiers and their parameters.
You can also change the weights of classifiers in the code.
Explicit parameters definition for classifiers is not yet implemented.
