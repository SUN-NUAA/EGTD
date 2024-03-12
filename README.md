# EGTD
This repository contains the source code for the algorithm proposed in the
article "EGTD: Entropy Guided Truth Discovery for Time Series Data". Below
are the instructions for running it on a Windows OS.

DATA PREPARATION
----------------
  1. Download the benchmark datasets from www.cs.ucr.edu/~eamonn/time_series_data/.
  2. Extract the zip file with the password "attempttoclassify" to a directory of
     your choice, for example, "D:\Data\UCR_TS_Archive_2015". In this directory,
     you will find 85 sub-datasets, each containing two data files, "XXXX_TRAIN"
     and "XXXX_TEST".

SOFTWARE INSTALLATION
---------------------
  1. Download Python 3.x from [python.org](https://www.python.org/ python-3.12.2-amd64.exe).
  2. Install the Python runtime environment by executing the downloaded .exe file.
     Ensure that the checkbox "Add python.exe to path" is checked during
     installation to enable easy access to Python from the command line.
  3. Install the required modules (pandas, numpy, and quad) by executing the
     following commands:
       pip install pandas
       pip install numpy
       pip install quad

RUN THE PROGRAM
---------------
  1. Modify the location of the datasets (i.e., the variable setPath) in "main.py" to match the
     directory specified in the "DATA PREPARATION" step, i.e., "D:\\Data\\UCR_TS_Archive_2015".
  2. Open the command line, navigate to the directory containing this source code.
  3. Execute "python main.py" to run the program on the 85 sub-datasets.
  4. Each line of the output displays the overall Mean Absolute Error (MAE) and
     Root Mean Square Error (RMSE) for the corresponding sub-dataset.
