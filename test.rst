===============
Cynet
===============


.. image:: http://zed.uchicago.edu/logo/logozed1.png
   :height: 400px
   :scale: 50 %
   :alt: alternate text
   :align: center

|
:Info: See <https://arxiv.org/abs/1406.6651> for theoretical background
:Author: ZeD@UChicago <zed.uchicago.edu>
:Description: Implementation of the Deep Granger net inference algorithm, described
    in https://arxiv.org/abs/1406.6651, for learning spatio-temporal stochastic processes
    (*point processes*). **cynet** learns a network of generative local models, without assuming
    any specific model structure.


**Introduction:**
    Cynet is a python wrapper for a C++ implementation of the Deep Granger network
    inference algorithm. This package seeks to assist in the parsing of raw data
    into appropriate formats and then building predictive models from them. This
    document will go through an example of how to use this package with the
    Chicago crime dataset to build and evaluate predictive models.
|

**Dataset:**
    <https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2>

|

**Installaion:**

.. code-block:: bash

    pip install cynet
|

**Contents:**
    1. Processing raw data into xGenESeSS friendly format.
    2. Generating models using xGenESeSS.
    3. Running Cynet binary to get predictions files.
    4. Evaluating Predictions.
    5. Using predictions to generate predictive heat maps.

|

**Section 1: Processing raw data into xGenESeSS friendly format.**
    The goal of this section is to show how to turn the raw Chicago crime data into
    a format appropriate for xGenESeSS.

|
**1.1: The Crime dataset.**
    The Chicago crime dataset is a large csv which can be downloaded from the link
    above. It is frequently updated and contains reported crime events from 2001
    to present. There are about 6.8 million rows, one for each recorded event.
    There are 22 different columns, one for a different variable. For our purposes,
    the important variables are:

    * Date: Contains the date and time at which the event took place.
    * Primary Type: the type of the crime. Includes but not limited to: Battery, Assault, Theft, Criminal Damage, Burglary, Motor Vehicle Theft.
    * Arrest: Inidicates if an arrest was made.
    * Latitude: Latitude coordinate of the event.
    * Longitude: Longitude coordinate of the event.
    |
    We will use these above variables to help parse the data into the appropriate format.
|
**1.2: The desired file formats and time series table.**

    To generate the Xgenesis models, we need three types of files. These three files
    constitutes a time series table. Each row in the table will describe a tile in our
    grid. Tiles are defined by coordinate boundaries and a variable type. That is,
    tiles with the same latitude and longitude boundaries but with different variables
    will count as separate tiles in this table. The column headers in this case will be
    time slices. The time slices in our example will be days. Each value in the table
    will be an integer describing the number of events that took place at that
    particular tile, within that particular time slice.

|
**Files and examples:**


Column file. The columns (time slices) in our table. In this example, they are one day
long.

.. code-block:: bash

    2014-01-01T00:00:00.000000000
    2014-01-02T00:00:00.000000000
    2014-01-03T00:00:00.000000000
    2014-01-04T00:00:00.000000000
    2014-01-05T00:00:00.000000000
    ...

Coordinate file. The rows (tiles) in our table:

.. code-block:: bash

    42.0196#42.02236#-87.66784#-87.66432#VAR
    42.0196#42.02236#-87.66784#-87.66432#BURGLARY-THEFT-MOTOR_VEHICLE_THEFT
    42.0196#42.02236#-87.66784#-87.66432#HOMICIDE-ASSAULT-BATTERY
    41.74874#41.75151#-87.57286#-87.56935#VAR
    41.74874#41.75151#-87.57286#-87.56935#BURGLARY-THEFT-MOTOR_VEHICLE_THEFT
    41.74874#41.75151#-87.57286#-87.56935#HOMICIDE-ASSAULT-BATTERY
    ...

Csv file. The actual timeseries:

.. code-block:: bash

    0 1 1 0 2 0 1 ...
    0 0 0 1 0 2 0 ...
    0 1 1 0 0 0 1 ...
    0 0 0 1 1 2 0 ...
    ...


If these examples are taken together, then the table implies that for the the
tile **42.0196#42.02236#-87.66784#-87.66432#VAR**, 0 events took place on 1/1/2014,
1 on 1/2/2014, 1 on 1/3/2014, 0 on 1/4/2014, 2 on 1/5/2014, etc.

|
|
**1.3: Intermediate Time Series Tables.**

        Here we begin processing the csv into the desired formats. The spatioTemporal
    class is used for this. This step will take a bit of time to run. We will fit the
    data from 2001 to 2018. We will group the various types in the **Primary Type**
    column into three groups. For each of these groups, we will produce an intermediate
    timeseries table. In these csv files, the columns are the dates and the rows will
    start with a tile followed by the time series on that tile.

    .. code-block:: python

        import cynet.cynet as cn
        import numpy as np

        EPS = 200
        STOREFILE='crime.p'
        CSVFILE='crime.csv'

        grid={'Latitude':np.around(np.linspace(41.5,42.05,EPS),decimals=5),
              'Longitude':np.around(np.linspace(-87.9,-87.2,EPS),decimals=5),
              'Eps':EPS}

        tiles=list([[grid['Latitude'][i],grid['Latitude'][i+1],grid['Longitude'][j], grid['Longitude'][j+1]]
                    for i in np.arange(len(grid['Latitude'])-1)
                    for j in np.arange(len(grid['Longitude'])-1)])

        S0=cn.spatioTemporal(log_file=CSVFILE,
                             log_store=STOREFILE,
                             types=[['BURGLARY','THEFT','MOTOR VEHICLE THEFT']],
                             value_limits=None,
                             grid=tiles,
                             init_date='2001-01-01',
                             end_date='2018-12-31',
                             freq='D',
                             EVENT='Primary Type',
                             threshold=0.05)
        S0.fit(csvPREF='CRIME-')
        tiles=S0.getGrid()

        with open("tiles.txt", "wb") as tiles_pickle:
            pickle.dump(tiles,tiles_pickle)

    In the above code:
        * **tiles** is generated using grid and EPS. 
