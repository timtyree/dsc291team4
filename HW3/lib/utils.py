import os
import subprocess
import findspark
from pyspark import SparkContext, SparkConf
import numpy as np
import pandas as pd
from numpy import linalg as LA
from statistics import mean
import ipyleaflet
from ipyleaflet import (
    Map,
    Marker,
    TileLayer, ImageOverlay,
    Polyline, Polygon, Rectangle, Circle, CircleMarker,
    GeoJSON,
    DrawControl
)
from matplotlib.colors import rgb2hex
from matplotlib.pyplot import xticks, yticks, figure
import pylab as plt

from ipyleaflet import Heatmap, WidgetControl, FullScreenControl
from ipywidgets import IntSlider, jslink

class leaflet_old:
    """
    Plots circles on a map (one per station) whose size is proportional to the number of feature measurements, and whose color is equal to their aggregated value (min, max, avg)  
    
    :param featureStr: the feature you'd like to visualize given as a string, eg. 'SNWD'
    :param aggregateType: 'avg', 'min', 'max'
    """
    
    def __init__(self, sqlctxt, featureStr):
        self.feat = featureStr
        self.pdfMaster = None
        self.sqlctxt = sqlctxt
        
        self.maxLong = None
        self.minLong = None
        self.maxLat = None
        self.minLat = None
        
        self.minAggVal = None
        self.maxAggVal = None
        
        
        self.cmap = plt.get_cmap('jet_r')
        self.m = None
        
    def add(self, dataframe):
        """
        Adds (or concatenates) a dataframe to the instance's master dataframe. The dataframe to be added must have atleast
        the following columns: station, latitude, longitude, featureStr, value
        """
        
        self.sqlctxt.registerDataFrameAsTable(dataframe, "temp")
        query = f"""
        SELECT Station, latitude, longitude, COUNT(Measurement), MEAN(Values)
        FROM temp
        WHERE Measurement=='SNWD'
        GROUP BY Station, latitude, longitude
        """
        tempdf = self.sqlctxt.sql(query).toPandas()
        
        if (self.pdfMaster is None):
            # set as new pdfMaster
            self.pdfMaster = tempdf
            
        else:            
            # append to existing dfMaster
            self.pdfMaster = pd.concat([self.pdfMaster, tempdf])
            
    def plot_all(self):
        # update internals
        self.maxLong = self.pdfMaster['longitude'].max()
        self.minLong = self.pdfMaster['longitude'].min()
        self.maxLat = self.pdfMaster['latitude'].max()
        self.minLat = self.pdfMaster['latitude'].min()
        self.minAggVal = self.pdfMaster['avg(Values)'].min()
        self.maxAggVal = self.pdfMaster['avg(Values)'].max()
        
        # update center of map
        self.center = [(self.minLat + self.maxLat)/2, (self.minLong + self.maxLong)/2]
        self.zoom = 6
        self.m = Map(default_tiles=TileLayer(opacity=1.0), center=self.center, zoom=self.zoom)
        
        # loop over all the points in given dataframe, adding them to self.map
        circles = []
        for index,row in self.pdfMaster.iterrows():
            _lat=row['latitude']
            _long=row['longitude']
            _count=row['count(Measurement)']
            _coef=row['avg(Values)']
#             pdb.set_trace()
            # taking sqrt of count so that the  area of the circle corresponds to the count
            c = Circle(location=(_lat,_long), radius=int(1200*np.sqrt(_count+0.0)), weight=1,
                    color='#AAA', opacity=0.8, fill_opacity=0.4,
                    fill_color=self.get_color(_coef))
            circles.append(c)
            self.m.add_layer(c)
        self.m
        
    
    def color_legend(self):
        self.cfig = figure(figsize=[10,1])
        ax = plt.subplot(111)
#         vals = self.cmap(np.arange(0,1,.005))[:,:3]
        vals = self.cmap(np.arange(0,1,.005))[:,:3]
        vals3 = np.stack([vals]*10)
        vals3.shape
        ax.imshow(vals3)
        midpoint = 200. * -self.minAggVal/(self.maxAggVal - self.minAggVal)
        xticks((0,midpoint,200),["%4.1f"%v for v in (self.minAggVal,0.,self.maxAggVal)])
        yticks(());

    def get_color(self, val):
        x = (val - self.minAggVal)/(self.maxAggVal - self.minAggVal)
        return(rgb2hex(self.cmap(x)[:3]))

class leaflet_eig:
    """
    Plots circles on a map (one per station) whose size is proportional to the number of feature measurements, and whose color is equal to their aggregated value (min, max, avg)  
    
    :param featureStr: the feature you'd like to visualize given as a string, eg. 'SNWD'
    :param aggregateType: 'avg', 'min', 'max'
    """
    
    def __init__(self, sqlctxt, featureStr):
        self.feat = featureStr
        self.pdfMaster = None
        self.sqlctxt = sqlctxt
        
        self.maxLong = None
        self.minLong = None
        self.maxLat = None
        self.minLat = None
        
        self.minAggVal = None
        self.maxAggVal = None
        
        
        self.cmap = plt.get_cmap('jet_r')
        self.m = None
        
    def add(self, dataframe):
        """
        Adds (or concatenates) a dataframe to the instance's master dataframe. The dataframe to be added must have atleast
        the following columns: station, latitude, longitude, featureStr, value
        """
        
        self.sqlctxt.registerDataFrameAsTable(dataframe, "temp")
        query = f"""
        SELECT Station, latitude, longitude, COUNT(Year) NumMeas, MEAN(Values)
        FROM temp
        GROUP BY Station, latitude, longitude
        """
        tempdf = self.sqlctxt.sql(query).toPandas()
        
        if (self.pdfMaster is None):
            # set as new pdfMaster
            self.pdfMaster = tempdf
            
        else:            
            # append to existing dfMaster
            self.pdfMaster = pd.concat([self.pdfMaster, tempdf])
            
    def plot_all(self):
        # update internals
        self.maxLong = self.pdfMaster['longitude'].max()
        self.minLong = self.pdfMaster['longitude'].min()
        self.maxLat = self.pdfMaster['latitude'].max()
        self.minLat = self.pdfMaster['latitude'].min()
        self.minAggVal = self.pdfMaster['avg(Values)'].min()
        self.maxAggVal = self.pdfMaster['avg(Values)'].max()
        
        # update center of map
        self.center = [(self.minLat + self.maxLat)/2, (self.minLong + self.maxLong)/2]
        self.zoom = 6
        self.m = Map(default_tiles=TileLayer(opacity=1.0), center=self.center, zoom=self.zoom)
        
        # loop over all the points in given dataframe, adding them to self.map
        circles = []
        for index,row in self.pdfMaster.iterrows():
            _lat=row['latitude']
            _long=row['longitude']
            _count=row['NumMeas']
            _coef=row['avg(Values)']
#             pdb.set_trace()
            # taking sqrt of count so that the  area of the circle corresponds to the count
            c = Circle(location=(_lat,_long), radius=int(2000*np.sqrt(_count+0.0)), weight=1,
                    color='#AAA', opacity=0.8, fill_opacity=0.4,
                    fill_color=self.get_color(_coef))
            circles.append(c)
            self.m.add_layer(c)
        self.m.add_control(FullScreenControl())
        
    
    def color_legend(self):
        self.cfig = figure(figsize=[10,1])
        ax = plt.subplot(111)
#         vals = self.cmap(np.arange(0,1,.005))[:,:3]
        vals = self.cmap(np.arange(0,1,.005))[:,:3]
        vals3 = np.stack([vals]*10)
        vals3.shape
        ax.imshow(vals3)
        midpoint = 200. * -self.minAggVal/(self.maxAggVal - self.minAggVal)
        xticks((0,midpoint,200),["%4.1f"%v for v in (self.minAggVal,0.,self.maxAggVal)])
        yticks(());

    def get_color(self, val):
        x = (val - self.minAggVal)/(self.maxAggVal - self.minAggVal)
        return(rgb2hex(self.cmap(x)[:3]))
     
    
class leaflet:
    """
    Plots circles on a map (one per station) whose size is proportional to the number of feature measurements, and whose color is equal to their aggregated value (min, max, avg)  
    
    :param featureStr: the feature you'd like to visualize given as a string, eg. 'SNWD'
    :param aggregateType: 'avg', 'min', 'max'
    """
    
    def __init__(self, sqlctxt, featureStr):
        self.feat = featureStr
        self.pdfMaster = None
        self.sqlctxt = sqlctxt
        
        self.maxLong = None
        self.minLong = None
        self.maxLat = None
        self.minLat = None
        
        self.minAggVal = None
        self.maxAggVal = None
        
        
        self.cmap = plt.get_cmap('jet')
        self.m = None
        
    def add(self, dataframe):
        """
        Adds (or concatenates) a dataframe to the instance's master dataframe. The dataframe to be added must have atleast
        the following columns: station, latitude, longitude, featureStr, value
        """
        
        self.sqlctxt.registerDataFrameAsTable(dataframe, "temp")
        query = f"""
        SELECT Station, latitude, longitude, NumMeas, MEAN(Values)
        FROM temp
        GROUP BY Station, latitude, longitude, NumMeas
        """
        tempdf = self.sqlctxt.sql(query).toPandas()
        
        if (self.pdfMaster is None):
            # set as new pdfMaster
            self.pdfMaster = tempdf
            
        else:            
            # append to existing dfMaster
            self.pdfMaster = pd.concat([self.pdfMaster, tempdf])
            
    def plot_all(self):
        # update internals
        self.maxLong = self.pdfMaster['longitude'].max()
        self.minLong = self.pdfMaster['longitude'].min()
        self.maxLat = self.pdfMaster['latitude'].max()
        self.minLat = self.pdfMaster['latitude'].min()
        self.minAggVal = self.pdfMaster['avg(Values)'].min()
        self.maxAggVal = self.pdfMaster['avg(Values)'].max()
        
        # update center of map
        self.center = [(self.minLat + self.maxLat)/2, (self.minLong + self.maxLong)/2]
        self.zoom = 6
        self.m = Map(default_tiles=TileLayer(opacity=1.0), center=self.center, zoom=self.zoom)
        
        # loop over all the points in given dataframe, adding them to self.map
        circles = []
        for index,row in self.pdfMaster.iterrows():
            _lat=row['latitude']
            _long=row['longitude']
            _count=row['NumMeas']
            _coef=row['avg(Values)']
            # taking sqrt of count so that the  area of the circle corresponds to the count
            c = Circle(location=(_lat,_long), radius=int(1200*(_count+0.0)), weight=1,
                    color='#AAA', opacity=0.8, fill_opacity=0.4,
                    fill_color=self.get_color(_coef))
            circles.append(c)
            self.m.add_layer(c)
        self.m
    
    def heatmap(self):
        # update internals
        self.maxLong = self.pdfMaster['longitude'].max()
        self.minLong = self.pdfMaster['longitude'].min()
        self.maxLat = self.pdfMaster['latitude'].max()
        self.minLat = self.pdfMaster['latitude'].min()
        self.minAggVal = self.pdfMaster['avg(Values)'].min()
        self.maxAggVal = self.pdfMaster['avg(Values)'].max()
        
        # update center of map
        self.center = [(self.minLat + self.maxLat)/2, (self.minLong + self.maxLong)/2]
        self.zoom = 4
        self.m = Map(default_tiles=TileLayer(opacity=1.0), center=self.center, zoom=self.zoom)
        
        # 
        locations = [[self.pdfMaster.iloc[i]['latitude'], self.pdfMaster.iloc[i]['longitude'], (self.pdfMaster.iloc[i]['avg(Values)'])*1000/(self.maxAggVal)] for i in range(len(self.pdfMaster)) if self.pdfMaster.iloc[i]['avg(Values)'] >= 0]
        hm_hot = Heatmap(locations=locations,
                     radius=10,
                     blur=5,
                     max=.8,
                     max_zoom=100,
                     gradient={0.0: 'yellow', .5: 'orange', 1.0: 'red'}
                    )
        self.m.add_layer(hm_hot)
        
        # 
        locations = [[self.pdfMaster.iloc[i]['latitude'], self.pdfMaster.iloc[i]['longitude'], (abs(self.pdfMaster.iloc[i]['avg(Values)']))*1000/(abs(self.minAggVal))] for i in range(len(self.pdfMaster)) if self.pdfMaster.iloc[i]['avg(Values)'] <= 0]
        hm_cold = Heatmap(locations=locations,
                     radius=10,
                     blur=5,
                     max=.8,
                     max_zoom=100,
                     gradient={0.0: 'lime', 0.5: 'cyan', 1.0: 'blue'}
                    )
        self.m.add_layer(hm_cold)
        
        radius_slider = IntSlider(description='Saturation', min=1, max=25, value=12)
        jslink((radius_slider, 'value'), (hm_hot, 'radius'))
        jslink((radius_slider, 'value'), (hm_cold, 'radius'))
        radius_control = WidgetControl(widget=radius_slider, position='topright')
        self.m.add_control(radius_control)
        
        self.m.add_control(FullScreenControl())
        
    
    def color_legend(self):
        self.cfig = figure(figsize=[10,1])
        ax = plt.subplot(111)
#         vals = self.cmap(np.arange(0,1,.005))[:,:3]
        vals = self.cmap(np.arange(0,1,.005))[:,:3]
        vals3 = np.stack([vals]*10)
        vals3.shape
        ax.imshow(vals3)
        midpoint = 200. * -self.minAggVal/(self.maxAggVal - self.minAggVal)
        xticks((0,midpoint,200),["%4.1f"%v for v in (self.minAggVal,0.,self.maxAggVal)])
        yticks(());

    def get_color(self, val):
        x = (val - self.minAggVal)/(self.maxAggVal - self.minAggVal)
        return(rgb2hex(self.cmap(x)[:3]))
    
def decadeMeasurementDelta(featureStr, states, data_dir, sqlContext):
    deltaDFs = []
    # create the sparksql context
    for s in states:
        parquet = s + '.parquet'
        parquet_path = data_dir + '/' + parquet
        df = sqlContext.read.parquet(parquet_path)
        sqlContext.registerDataFrameAsTable(df,f'table_{s}')
        
        # 70s
        
        ###
        Query = f"""
        SELECT Station, Measurement, Values, longitude, latitude, Year
        FROM table_{s}
        WHERE Measurement=={featureStr} and (Year >= 1970 and Year < 1980)
        """
        query70s = sqlContext.sql(Query)
        rdd70s = query70s.rdd.map(lambda x: replaceSNWD(x, 'Values'))
        dfs70 = sqlContext.createDataFrame(rdd70s)
        sqlContext.registerDataFrameAsTable(dfs70, f'table_{s}_70s')
        
        ###
        Query = f"""
        SELECT Station, COUNT(Measurement) as NumMeas70s, MEAN(Values) as AvgVal70s, longitude, latitude
        FROM table_{s}_70s
        GROUP BY Station, Measurement, longitude, latitude
        ORDER BY Station ASC
        """
        groupbystation70s = sqlContext.sql(Query)
        sqlContext.registerDataFrameAsTable(groupbystation70s, "meas70s")
        ###
        
        # 00s
        
        ###
        Query = f"""
        SELECT Station, Measurement, Values, longitude, latitude, Year
        FROM table_{s}
        WHERE Measurement=={featureStr} and (Year >= 2000 and Year < 2010)
        """
        query00s = sqlContext.sql(Query)
        rdd00s = query00s.rdd.map(lambda x: replaceSNWD(x, 'Values'))
        dfs00 = sqlContext.createDataFrame(rdd00s)
        sqlContext.registerDataFrameAsTable(dfs00, f'table_{s}_00s')
        
        ###
        Query = f"""
        SELECT Station, COUNT(Measurement) as NumMeas00s, MEAN(Values) as AvgVal00s, longitude, latitude
        FROM table_{s}_00s
        GROUP BY Station, Measurement, longitude, latitude
        ORDER BY Station ASC
        """
        groupbystation00s = sqlContext.sql(Query)
        sqlContext.registerDataFrameAsTable(groupbystation00s, "meas00s")
        ###
        
        ### Combined
        
        ###
        Query = f"""
        SELECT meas70s.Station, meas70s.longitude, meas70s.latitude, AvgVal70s, AvgVal00s, NumMeas70s, NumMeas00s
        FROM meas70s
        INNER JOIN meas00s on meas70s.Station = meas00s.Station
        ORDER BY Station ASC
        """
        combined = sqlContext.sql(Query)
        rddcombined = combined.rdd.map(lambda x: computeDelta(x, 'AvgVal70s', 'AvgVal00s'))
        deltaDFs.append(sqlContext.createDataFrame(rddcombined))
        print(f"Done adding {s}")
    
    return(deltaDFs)


def decadeMeasurementDelta_states(featureStr, states, data_dir, sqlContext):
    deltaDFs = []
    # create the sparksql context
    for s in states:
        parquet = s + '.parquet'
        parquet_path = data_dir + '/' + parquet
        df = sqlContext.read.parquet(parquet_path)
        sqlContext.registerDataFrameAsTable(df,f'table_{s}')
        
        # 70s
        
        ###
        Query = f"""
        SELECT state, Station, Measurement, Values, longitude, latitude, Year
        FROM table_{s}
        WHERE Measurement=={featureStr} and (Year >= 1970 and Year < 1980)
        """
        query70s = sqlContext.sql(Query)
        rdd70s = query70s.rdd.map(lambda x: replaceSNWD(x, 'Values'))
        dfs70 = sqlContext.createDataFrame(rdd70s)
        sqlContext.registerDataFrameAsTable(dfs70, f'table_{s}_70s')
        
        ###
        Query = f"""
        SELECT state, Station, COUNT(Measurement) as NumMeas70s, MEAN(Values) as AvgVal70s, longitude, latitude
        FROM table_{s}_70s
        GROUP BY state, Station, Measurement, longitude, latitude
        ORDER BY Station ASC
        """
        groupbystation70s = sqlContext.sql(Query)
        sqlContext.registerDataFrameAsTable(groupbystation70s, "meas70s")
        ###
        
        # 00s
        
        ###
        Query = f"""
        SELECT state, Station, Measurement, Values, longitude, latitude, Year
        FROM table_{s}
        WHERE Measurement=={featureStr} and (Year >= 2000 and Year < 2010)
        """
        query00s = sqlContext.sql(Query)
        rdd00s = query00s.rdd.map(lambda x: replaceSNWD(x, 'Values'))
        dfs00 = sqlContext.createDataFrame(rdd00s)
        sqlContext.registerDataFrameAsTable(dfs00, f'table_{s}_00s')
        
        ###
        Query = f"""
        SELECT state, Station, COUNT(Measurement) as NumMeas00s, MEAN(Values) as AvgVal00s, longitude, latitude
        FROM table_{s}_00s
        GROUP BY state, Station, Measurement, longitude, latitude
        ORDER BY Station ASC
        """
        groupbystation00s = sqlContext.sql(Query)
        sqlContext.registerDataFrameAsTable(groupbystation00s, "meas00s")
        ###
        
        ### Combined
        
        ###
        Query = f"""
        SELECT meas70s.state, meas70s.Station, meas70s.longitude, meas70s.latitude, AvgVal70s, AvgVal00s, NumMeas70s, NumMeas00s
        FROM meas70s
        INNER JOIN meas00s on meas70s.Station = meas00s.Station
        ORDER BY Station ASC
        """
        combined = sqlContext.sql(Query)
        rddcombined = combined.rdd.map(lambda x: computeDelta(x, 'AvgVal70s', 'AvgVal00s'))
        deltaDFs.append(sqlContext.createDataFrame(rddcombined))
        print(f"Done adding {s}")
    
    return(deltaDFs)
    
def computeDelta(x, key70s, key00s):
    d = x.asDict()
    if (key70s in d) and (key00s in d):
        delta = d[key00s] - d[key70s]
        d['Values'] = delta
        d['NumMeas'] = d['NumMeas70s']+d['NumMeas00s']
    d.pop('NumMeas70s')
    d.pop('NumMeas00s')
    d.pop(key70s)
    d.pop(key00s)
    return(d)

def replaceSNWD(x, key):
    d = x.asDict()
    if key in d:
        odd_indices = list(x.Values)[1::2]
        winter_odd_indices = odd_indices[:90]
        d[key] = float(mean(winter_odd_indices))
    return(d)

def remove0sAndAverage(x, key):
    d = x.asDict()
    if key in d:
        vals = list(x.Values)
        vals = [v for v in vals if v > 0]
        if len(vals)==0:
            vals = [0]
        d[key] = float(mean(vals))
    return(d)

def getStateData(state, data_dir, tarname, parquet):

    shell = f'''
    rm -rf {data_dir}/{tarname}
    curl https://mas-dse-open.s3.amazonaws.com/Weather/by_state/{tarname} > {data_dir}/{tarname}
    ls -lh {data_dir}/{tarname}
    pwd
    cd {data_dir}
    tar -xzf {tarname}
    du ./{parquet}
    '''

    out = subprocess.getoutput(shell)
    print(out)

def getStations():
    data_dir = '../DataHW3'
    tarname = 'Weather_Stations.tgz'
    parquet = 'stations.parquet'
    shell = f'''
    curl https://mas-dse-open.s3.amazonaws.com/Weather/{tarname} > {data_dir}/{tarname}
    cd {data_dir}
    tar -xzf {tarname}
    du ./{parquet}    
    '''
    out = subprocess.getoutput(shell)
    print(out)
    
    
def create_sc(pyFiles):
    sc_conf = SparkConf()
    sc_conf.setAppName("Weather_PCA")
    sc_conf.set('spark.executor.memory', '3g')
    sc_conf.set('spark.executor.cores', '1')
    sc_conf.set('spark.cores.max', '4')
    sc_conf.set('spark.default.parallelism','10')
    sc_conf.set('spark.logConf', True)
    print(sc_conf.getAll())

    sc = SparkContext(conf=sc_conf,pyFiles=pyFiles)

    return sc 


def outerProduct(X):
    """Computer outer product and indicate which locations in matrix are undefined"""
    O=np.outer(X,X)
    N=1-np.isnan(O)
    return (O,N)

def sumWithNan(M1,M2):
    """Add two pairs of (matrix,count)"""
    (X1,N1)=M1
    (X2,N2)=M2
    N=N1+N2
    X=np.nansum(np.dstack((X1,X2)),axis=2)
    return (X,N)

### HW: Replace the RHS of the expressions in this function (They need to depend on S and N.
def HW_func(S,N):
    E=      np.ones([365]) # E is the sum of the vectors
    NE=     np.ones([365]) # NE is the number of not-nan antries for each coordinate of the vectors
    Mean=   np.ones([365]) # Mean is the Mean vector (ignoring nans)
    O=      np.ones([365,365]) # O is the sum of the outer products
    NO=     np.ones([365,365]) # NO is the number of non-nans in the outer product.
    return  E,NE,Mean,O,NO

def computeCov(RDDin):
    """computeCov recieves as input an RDD of np arrays, all of the same length, 
    and computes the covariance matrix for that set of vectors"""
    RDD=RDDin.map(lambda v:np.array(np.insert(v,0,1),dtype=np.float64)) # insert a 1 at the beginning of each vector so that the same 
                                           #calculation also yields the mean vector
    OuterRDD=RDD.map(outerProduct)   # separating the map and the reduce does not matter because of Spark uses lazy execution.
    (S,N)=OuterRDD.reduce(sumWithNan)

    E,NE,Mean,O,NO=HW_func(S,N)

    Cov=O/NO - np.outer(Mean,Mean)
    # Output also the diagnal which is the variance for each day
    Var=np.array([Cov[i,i] for i in range(Cov.shape[0])])
    return {'E':E,'NE':NE,'O':O,'NO':NO,'Cov':Cov,'Mean':Mean,'Var':Var}

def computeStatistics(sqlContext,df):
    """Compute all of the statistics for a given dataframe
    Input: sqlContext: to perform SQL queries
            df: dataframe with the fields 
            Station(string), Measurement(string), Year(integer), Values (byteArray with 365 float16 numbers)
    returns: STAT, a dictionary of dictionaries. First key is measurement, 
             second keys described in computeStats.STAT_Descriptions
    """

    sqlContext.registerDataFrameAsTable(df,'weather')
    STAT={}  # dictionary storing the statistics for each measurement
    measurements=['TMAX', 'SNOW', 'SNWD', 'TMIN', 'PRCP', 'TOBS']
    
    for meas in measurements:
        t=time()
        Query="SELECT * FROM weather\n\tWHERE measurement = '%s'"%(meas)
        mdf = sqlContext.sql(Query)
        print(meas,': shape of mdf is ',mdf.count())

        data=mdf.rdd.map(lambda row: unpackArray(row['Values'],np.float16))

        #Compute basic statistics
        STAT[meas]=computeOverAllDist(data)   # Compute the statistics 

        # compute covariance matrix
        OUT=computeCov(data)

        #find PCA decomposition
        eigval,eigvec=LA.eig(OUT['Cov'])

        # collect all of the statistics in STAT[meas]
        STAT[meas]['eigval']=eigval
        STAT[meas]['eigvec']=eigvec
        STAT[meas].update(OUT)

        print('time for',meas,'is',time()-t)
    
    return STAT

# Compute the overall distribution of values and the distribution of the number of nan per year
def find_percentiles(SortedVals,percentile):
    L=int(len(SortedVals)/percentile)
    return SortedVals[L],SortedVals[-L]
  
def computeOverAllDist(rdd0):
    UnDef=np.array(rdd0.map(lambda row:sum(np.isnan(row))).sample(False,0.01).collect())
    flat=rdd0.flatMap(lambda v:list(v)).filter(lambda x: not np.isnan(x)).cache()
    count,S1,S2=flat.map(lambda x: np.float64([1,x,x**2]))\
                  .reduce(lambda x,y: x+y)
    mean=S1/count
    std=np.sqrt(S2/count-mean**2)
    Vals=flat.sample(False,0.0001).collect()
    SortedVals=np.array(sorted(Vals))
    low100,high100=find_percentiles(SortedVals,100)
    low1000,high1000=find_percentiles(SortedVals,1000)
    return {'UnDef':UnDef,\
          'mean':mean,\
          'std':std,\
          'SortedVals':SortedVals,\
          'low100':low100,\
          'high100':high100,\
          'low1000':low100,\
          'high1000':high1000
          }

# description of data returned by computeOverAllDist
STAT_Descriptions=[
('SortedVals', 'Sample of values', 'vector whose length varies between measurements'),
 ('UnDef', 'sample of number of undefs per row', 'vector whose length varies between measurements'),
 ('mean', 'mean value', ()),
 ('std', 'std', ()),
 ('low100', 'bottom 1%', ()),
 ('high100', 'top 1%', ()),
 ('low1000', 'bottom 0.1%', ()),
 ('high1000', 'top 0.1%', ()),
 ('E', 'Sum of values per day', (365,)),
 ('NE', 'count of values per day', (365,)),
 ('Mean', 'E/NE', (365,)),
 ('O', 'Sum of outer products', (365, 365)),
 ('NO', 'counts for outer products', (365, 365)),
 ('Cov', 'O/NO', (365, 365)),
 ('Var', 'The variance per day = diagonal of Cov', (365,)),
 ('eigval', 'PCA eigen-values', (365,)),
 ('eigvec', 'PCA eigen-vectors', (365, 365))
]
