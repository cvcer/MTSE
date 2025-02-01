Multi-Target Linear Shrinkage estimator

In the folder, there are several Python files:
	* MTSE.py:
		Implementation of the MTSE oracle and estimator presented in the paper. 
		The docstrings of the functions "MTSE_estimator" and "MTSE_estimator_oracle" detail how to use them.
	* hub.py:
		Central file to reproduce the experiment. 
		Runing this file will run and plot/print the results of the 4 experiments of the paper 
		(as it is, it can take 48h, change the hyperparameters to run it quicker but less precise).
		The paths are designed to work under Windows. Change the "\\" into "/" for Unix.
	* QIS.py, GIS.py, LIS.py, analytical_shrinkage.py, LWO_estimator.py:
		Estimators implemented to compare in the GMV experiment. 
		The original articles are cited in the corresponding section (GIS and LIS come from the same paper as QIS).
	* dataio.py, dataloader.py, markowitz.py, MC.py, plot.py:
		Utilitary functions tu run the experiments.

The dataset of prices of SP500 for the GMV experiment are in the folder "data". 
dataloader.py is in charge of leading it properly in the according experiment.