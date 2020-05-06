#!/usr/bin/bash
conda create -n spark -c anaconda-cluster python=3.7 spark \
	accelerate ipython-notebook
source activate spark
#TODO: fix SPARK_HOME not found)
# PYSPARK_DRIVER_PYTHON=ipython pyspark
# IPYTHON_OPTS="notebook" ./bin/pyspark #starts jupyter notebook