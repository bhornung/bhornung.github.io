---
layout: post
title:  "Is it Cfa or Cfb?"
date:   2019-01-29 21:20:06 +0000
categories: python
---

## Goals

We attempt to construct a pipeline to predict the Koppen climate of a geographical location based on the observed temperatures and precipitation.

The raw notebook can be found [here](https://github.com/bhornung/bhornung.github.io/blob/master/assets/climate-classification/notebook/climate-classification.ipynb).

## The data

### Source

The temperature and precipitation data were obtained from the [WorldClim - Global Climate Data](http://www.worldclim.org/current) database. The resolution is 10' on an equidistant latitude and longitude grid. Please note, this leads to increasing weight of points advancing towards the polar regions. The Koppen classification can be found at WORLD MAPS OF [Koppen--Geiger Climate Classification page](http://koeppen-geiger.vu-wien.ac.at/) of Universitat Wien. The data cover most of the landmass save for the Antarctica.

The two datasets were projected onto a common grid and saved to a csv store.

### Structure

The loaded from the store have roughly 64,00 points:


```python
path_to_store = r'path/to/temperature-precipitation-cls.csv'
df = pd.read_csv(path_to_store, index_col = 0 )

print("Number of records: {0}".format(len(df)))
```

    Number of records: 64409
    

The fields in the column header have the following meaning:

* `Lat` : lattitude in degrees
* `Lon` : longitude in degrees
* `Cls` : Koppen climate label
* `2-13` : monthly average in temperature in $10 \cdot \text{C}^{\circ}$  from January to December.
* `14-25` : monthly average precipitation in millimetres from January to December.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lat</th>
      <th>Lon</th>
      <th>Cls</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-55.75</td>
      <td>-67.75</td>
      <td>ET</td>
      <td>86.0</td>
      <td>83.0</td>
      <td>76.0</td>
      <td>56.0</td>
      <td>36.0</td>
      <td>20.0</td>
      <td>15.0</td>
      <td>...</td>
      <td>86.0</td>
      <td>81.0</td>
      <td>75.0</td>
      <td>66.0</td>
      <td>67.0</td>
      <td>63.0</td>
      <td>51.0</td>
      <td>51.0</td>
      <td>75.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-55.75</td>
      <td>-67.25</td>
      <td>ET</td>
      <td>83.0</td>
      <td>80.0</td>
      <td>73.0</td>
      <td>53.0</td>
      <td>32.0</td>
      <td>17.0</td>
      <td>12.0</td>
      <td>...</td>
      <td>72.0</td>
      <td>70.0</td>
      <td>61.0</td>
      <td>53.0</td>
      <td>52.0</td>
      <td>55.0</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>58.0</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-55.25</td>
      <td>-70.75</td>
      <td>ET</td>
      <td>88.0</td>
      <td>86.0</td>
      <td>76.0</td>
      <td>57.0</td>
      <td>40.0</td>
      <td>25.0</td>
      <td>21.0</td>
      <td>...</td>
      <td>145.0</td>
      <td>132.0</td>
      <td>121.0</td>
      <td>113.0</td>
      <td>105.0</td>
      <td>107.0</td>
      <td>103.0</td>
      <td>92.0</td>
      <td>105.0</td>
      <td>107.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-55.25</td>
      <td>-70.25</td>
      <td>ET</td>
      <td>88.0</td>
      <td>86.0</td>
      <td>76.0</td>
      <td>57.0</td>
      <td>39.0</td>
      <td>24.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>131.0</td>
      <td>119.0</td>
      <td>110.0</td>
      <td>104.0</td>
      <td>95.0</td>
      <td>97.0</td>
      <td>90.0</td>
      <td>80.0</td>
      <td>95.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-55.25</td>
      <td>-69.75</td>
      <td>ET</td>
      <td>71.0</td>
      <td>69.0</td>
      <td>59.0</td>
      <td>39.0</td>
      <td>20.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>120.0</td>
      <td>108.0</td>
      <td>99.0</td>
      <td>94.0</td>
      <td>86.0</td>
      <td>88.0</td>
      <td>80.0</td>
      <td>72.0</td>
      <td>89.0</td>
      <td>88.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



The latitude and longitude columns are removed for they will not be used in the following:


```python
df.drop(['Lat', 'Lon'], axis = 1, inplace = True)
```

### Preliminary checks

A quick sanity check is worth the time:


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>...</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>...</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
      <td>64409.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-19.496421</td>
      <td>-7.391420</td>
      <td>27.436694</td>
      <td>76.984102</td>
      <td>124.579748</td>
      <td>160.791535</td>
      <td>178.200174</td>
      <td>170.462063</td>
      <td>139.925818</td>
      <td>92.115248</td>
      <td>...</td>
      <td>55.060628</td>
      <td>53.122390</td>
      <td>56.857893</td>
      <td>63.541989</td>
      <td>73.573740</td>
      <td>73.104815</td>
      <td>64.611576</td>
      <td>58.248350</td>
      <td>54.097471</td>
      <td>54.739369</td>
    </tr>
    <tr>
      <th>std</th>
      <td>221.942406</td>
      <td>218.333765</td>
      <td>199.267312</td>
      <td>164.706627</td>
      <td>126.037252</td>
      <td>99.320742</td>
      <td>88.278280</td>
      <td>91.803491</td>
      <td>111.517567</td>
      <td>146.584784</td>
      <td>...</td>
      <td>75.628189</td>
      <td>67.982472</td>
      <td>70.617548</td>
      <td>79.706175</td>
      <td>90.766774</td>
      <td>84.688932</td>
      <td>72.526402</td>
      <td>66.720411</td>
      <td>66.163412</td>
      <td>74.047716</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-512.000000</td>
      <td>-469.000000</td>
      <td>-442.000000</td>
      <td>-357.000000</td>
      <td>-206.000000</td>
      <td>-116.000000</td>
      <td>-82.000000</td>
      <td>-95.000000</td>
      <td>-165.000000</td>
      <td>-275.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-219.000000</td>
      <td>-199.000000</td>
      <td>-130.000000</td>
      <td>-41.000000</td>
      <td>35.000000</td>
      <td>94.000000</td>
      <td>119.000000</td>
      <td>107.000000</td>
      <td>56.000000</td>
      <td>-22.000000</td>
      <td>...</td>
      <td>11.000000</td>
      <td>13.000000</td>
      <td>14.000000</td>
      <td>14.000000</td>
      <td>19.000000</td>
      <td>20.000000</td>
      <td>19.000000</td>
      <td>17.000000</td>
      <td>13.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-29.000000</td>
      <td>-12.000000</td>
      <td>34.000000</td>
      <td>88.000000</td>
      <td>135.000000</td>
      <td>160.000000</td>
      <td>180.000000</td>
      <td>172.000000</td>
      <td>142.000000</td>
      <td>94.000000</td>
      <td>...</td>
      <td>24.000000</td>
      <td>28.000000</td>
      <td>34.000000</td>
      <td>43.000000</td>
      <td>52.000000</td>
      <td>53.000000</td>
      <td>44.000000</td>
      <td>36.000000</td>
      <td>30.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>214.000000</td>
      <td>221.000000</td>
      <td>229.000000</td>
      <td>230.000000</td>
      <td>232.000000</td>
      <td>243.000000</td>
      <td>249.000000</td>
      <td>248.000000</td>
      <td>242.000000</td>
      <td>238.000000</td>
      <td>...</td>
      <td>64.000000</td>
      <td>62.000000</td>
      <td>67.000000</td>
      <td>78.000000</td>
      <td>88.000000</td>
      <td>89.000000</td>
      <td>80.000000</td>
      <td>71.000000</td>
      <td>67.000000</td>
      <td>63.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>337.000000</td>
      <td>332.000000</td>
      <td>331.000000</td>
      <td>339.000000</td>
      <td>359.000000</td>
      <td>384.000000</td>
      <td>392.000000</td>
      <td>382.000000</td>
      <td>357.000000</td>
      <td>325.000000</td>
      <td>...</td>
      <td>609.000000</td>
      <td>707.000000</td>
      <td>759.000000</td>
      <td>1471.000000</td>
      <td>1728.000000</td>
      <td>1232.000000</td>
      <td>903.000000</td>
      <td>919.000000</td>
      <td>802.000000</td>
      <td>705.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 24 columns</p>
</div>



The precipitation data exhibit a distribution heavily skewed by outliers. This is best visualised by plotting the density and cumulative distribution functions along with the (0.1, 0.2, 0.3, ..., 0.9, 0.99) quantiles in the top row of the figure below. 


```python
df_deciles = df.describe(percentiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]).drop(['count', 'mean', 'std'])
```

The box plots of the monthly data in the middle row also emphasise the presence of large number outliers. In addition, the temperature and precipitation data are on different scales 


![png]({{"/assets/climate-classification/images/output_15_0.png"}})


We are foremost interested in how the observations are distributed within a class. For this end, the monthly mean temperature and precipitation and standard deviation thereof are calculated for all climates to see where they are positioned on a common scale. It is apparent that the temperature data encompass a range five to ten times larger than that of the the precipitation data 


![png]({{"/assets/climate-classification/images/output_17_0.png"}})


Ideally all of the features should be on similar scales, because it would make the distances in each feature subspace comparable. An other desirable property that each feature of each class is normally distributed. It would enable us to use methods require normally distributed data, such as discriminant analysis. Thus we will attempt to:

* Bring the temperature and precipitation data to similar scales
* Further transform them to approximately normal distribution for each feature.

`StandardScaler` and `MinMaxScaler` are out of question for they cannot handle outliers. The outliers have two effects. Firstly, using the minmax scaler, the ranges where the bulk of the data lays would be squashed due to the large value of the outlying points. Secondly, the variance would be two large, which would again result in constraining the meaningful data to a small range. Secondly, the variance would be inflated by the spread of the outliers, and it thus would not reflect the variance of the bulk of the data. Therefore performing any variance based transformation, such as PCA, or fitting, e.g. discriminant analysis, would potentially lead to information loss.

`RobustScaler` uses specific quantiles to scale the data. Choosing the quantile range carefully, say 0.10 to 0.90, would avoid squashing the data, but they still would be riddled with outliers.

The transform that we are apply to the data circumvents both problems. It is the Yeo--Johnson power transform. It rescales the datum individually to follow an approximate normal distribution. The transformed distribution is then scaled by the variance. In this manner the temperature and precipitation data sets are moved to similar ranges. It is worth noting, the individual features will not be normally distributed. Although, they are expected to be less riddled with outliers.


```python
from sklearn.preprocessing import power_transform

temp_trf = power_transform(df.loc[:, '2' : '13'].values.reshape(-1, 1), method = 'yeo-johnson') 
                       # with_centering = False, quantile_range = (00.0, 99.0)) 

precip_trf = power_transform(df.loc[:, '14' : '25'].values.reshape(-1, 1), method = 'yeo-johnson')
                               #        with_centering = False, quantile_range = (0.0, 99.0))

# create new dataframe of the transformed values
df_trf = pd.DataFrame(data = np.concatenate([temp_trf.reshape(-1, 12), precip_trf.reshape(-1, 12)], axis = 1),   
                      columns = [str(x) for x in range(2, 26)])   
 
df_trf['Cls'] = df['Cls'].values
```

As we can see in the top row of the plot below, the 0th and 99th percentiles of the temperature and precipitation distributions are positioned close to each other. The number and distance of the outliers are greatly reduced as indicated by the middle row. These changes are further confirmed by the scatter plots of the monthly data in the bottom row. 


![png]({{"/assets/climate-classification/images/output_21_0.png"}})


The monthly precipitation and temperature values are now neatly positioned within the same range for all climates as shown in the plot below.


![png]({{"/assets/climate-classification/images/output_23_0.png"}})


### Feature engineering

In this section, we further transform the data to generate features can optimally used to establish classification rules.

Firstly, the Yeo--Johnson transformed data are saved in variable `X`:


```python
X = df_trf.iloc[:, :-1].values
```

#### Label encoding

The literal class labels are converted to integer ones and stored in the vector `y`:


```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(df_trf['Cls'].values)
label_codes = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
```

#### Dimensionality reduction

Before we proceed to select classifiers it is the most useful to investigate the cardinality of various classes. 


```python
from collections import Counter
", ".join( "({0} : {1})".format(label_codes[x[0]], x[1]) for x in Counter(y).most_common())
```




    '(Dfc : 10894), (ET : 7219), (BWh : 7193), (Aw : 5894), (Dfb : 4406), (BSk : 3273), (BSh : 3107), (Cfa : 2855), (Af : 2440), (Cfb : 2234), (EF : 1980), (BWk : 1838), (Am : 1778), (Cwa : 1501), (Dfd : 1410), (Dwc : 1315), (Csa : 996), (Dfa : 756), (Csb : 677), (Dwb : 651), (Cwb : 560), (Dwa : 313), (As : 276), (Dsc : 261), (Dsb : 207), (Cfc : 175), (Dwd : 100), (Dsa : 79), (Csc : 11), (Cwc : 10)'



There are four--five classes in which the number of samples are comparable to that of the dimension. Clearly, a dimensionality reduction would help here. We invoke our old friend principal component analysis (PCA).

It is readily seen that the first six principal components account for almost 99% of the total variance. This implies there is a chance to cluster the data in a projected subspace whose dimension is sufficiently smaller than the number of samples of almost all classes. 


```python
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X)
variance_ratio = pca.explained_variance_ratio_
```

![png]({{"/assets/climate-classification/images/output_32_0.png"}})

## Classification

We now proceed to select a suitable method process the climate data from the plethora of classification algorithms.

### Nearest means

Because it is too hard to choose, we create our classification method based on the class means. The set of labels are denoted by $L$. The cluster label, $l_{i}$ of the datum $i$ is that of the class whose mean $\mu_{j}$ is closest to the point in question:

$$
l_{i} = \underset{j \in L}{\operatorname{argmin}} ||\mu_{j} - x_{i}||_{2}\, .
$$

The partition above results in the Voronoi decomposition of the data around the class means. This method is implemented in the `NearestMeanClassifier` class


```python
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances_argmin

class NearestMeanClassifier(BaseEstimator):
    
    def __init__(self):
        
        self._is_fitted = False
    
    def fit(self, X, y):
        """
        Calculates the class centres on a training sample.
        
        Parameters:
            X np.ndarray, (n_samples, n_features) : training observations
            y (n_samples) :  training class labels
        
        Returns:
            self
        """
        
        labels = np.unique(y)
        n_clusters = labels.size
        
        self.cluster_centers_ = np.zeros((n_clusters, X.shape[1]))
        self.cluster_labels_ = np.zeros(n_clusters, dtype = np.int)
        
        for idx, label in enumerate(labels):
            self.cluster_centers_[idx] = np.mean(X[y == label], axis = 0)
            self.cluster_labels_[idx] = label
            
        self._is_fitted = True
         
    def predict(self, X):
        """
        Returns the predicted class labels on a test sample.
        Parameters:
            X np.ndarray, (n_samples, n_features) : observations
            
        Returns:
            labels_ np.ndarray, (n_samples) : predicted labels
        """
        
        if not self._is_fitted:
            raise ValueError("Classifier must be fit before prediction")
            
        label_positions = pairwise_distances_argmin(X, self.cluster_centers_)
        self.labels_ = self.cluster_labels_[label_positions]
        
        return self.labels_
```

The classification is carried out in the projected PCA space. A fivefold cross validation is invoked to determine the optimum number of principal components. Accuracy and adjusted rand score -- to account for the chance matches -- will measure the goodness of the classifier.


```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, make_scorer

cv = KFold(n_splits = 5, shuffle = True)
pca = PCA()
classifier = NearestMeanClassifier()

pipeline = Pipeline([('pca', pca), ('classifier', classifier)])

param_grid_nmc = [{'pca__n_components' : [2, 4, 6, 8, 10, 12, 14]}]

scoring = scoring = {'ars': make_scorer(adjusted_rand_score), 'accuracy': make_scorer(accuracy_score)}

grid_nmc = GridSearchCV(pipeline, cv = cv, param_grid = param_grid_nmc, 
                        scoring = scoring, 
                        refit = False,
                        return_train_score = False)
_ = grid_nmc.fit(X, y)
```


![png]({{"/assets/climate-classification/images/output_38_0.png"}})


The classifier is surprisingly poor. What could the possible reasons be? We tacitly required
*  the distributions being pairwise symmetric with respect to the dividing hyperplane.
* The distribution of each class is confined within its Voronoi boundary

These are really strict requirements which are not granted to have met.


### Structure of data

It is high time to delve into the structure of the data deeper. The projections of the error ellipses onto the 2D hyperplanes spanned by the first six principal components are shown in the figure below. The labels in the top left corner are the indices of the hyperplanes, 0 being the largest. Only the first most populous classes are shown for sake of clarity. It is clear, the distributions are far from symmetric. Also there is a considerable amount of overlap between various classes.


![png]({{"/assets/climate-classification/images/output_43_0.png"}})


#### Linear discriminant analysis

One can rightly point out that the binary combination of the PCA vectors (unit or zero coefficients) are not necessarily those that maximally separate the classes.

Linear Discriminant Analysis (LDA) assumes all classes are identically distributed according to a normal distribution. Using this assumption it finds those projections of the data along which the classes are most separated. Using less than the all principal components posing the danger that some variables being discarded that might be important components of the separating planes. It is however reasonable to assume the separating planes are correlated with the principal components given the large number of classes. (On the other hand, it is not at all difficult to draw an example which would prove otherwise, but that would be a really contrived one.)

To use LDA, the data are required to be normal and identically distributed. If any of these criteria grossly violated, this classification method shall not be used, or at least can result in poor performance. Nevertheless, the projections onto the 2D LDA hyperplanes are plotted below. The figure reassures us as to our assumption, that linear combinations of the vectors do not separate well the classes. (A cross validated LDA fit would result in approximately 80% accuracy.)


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()
X_lda = lda.fit_transform(X, y)
```


![png]({{"/assets/climate-classification/images/output_46_0.png"}})


### k-nearest neighbours

Separating planes, thus must be of higher order than linear. Unfortunately, quadratic discriminant analysis cannot be invoked for the low membership of certain classes. An other option is to use support vector machines with nonlinear kernels. 

Since the data are mingled, why not to use them directly to decide membership? A point is more likely to have a label identical to those of its neighbours. The k-nearest neighbour method utilises this idea.

#### Space to search in

The cost of running a KNN classifier depends on the dimension. It is thus advantageous the reduce the dimensionality of the data. LDA creates a series of projections along which the class separation is enhanced. There we use a succession of them to build the KNN classifier. 


```python
from sklearn.neighbors import KNeighborsClassifier

lda = LDA()
pca = PCA()

knn = KNeighborsClassifier()

pipeline_lda = Pipeline([('lda', lda), ('knn', knn)])
pipeline_pca = Pipeline([('pca', pca), ('knn', knn)])

cv = KFold(n_splits = 5, shuffle = True)

param_grid_lda = [{'lda__n_components' : [4, 6, 8, 10]}]
param_grid_pca = [{'pca__n_components' : [4, 6, 8, 10]}]

grid_knn_lda = GridSearchCV(pipeline_lda, cv = cv, param_grid = param_grid_lda, 
                        scoring = scoring, 
                        refit = False,
                        return_train_score = False)

grid_knn_pca = GridSearchCV(pipeline_pca, cv = cv, param_grid = param_grid_pca, 
                        scoring = scoring, 
                        refit = False,
                        return_train_score = False)

_1 = grid_knn_lda.fit(X, y)
_2 = grid_knn_pca.fit(X, y)
```

The fact that PCA outperforms LDA at lower dimensions reminds us the fact the many assumptions of LDA is not met. At higher dimensions, both method perform equally well. In the following we choose PCA for feature engineering.


![png]({{"/assets/climate-classification/images/output_50_0.png"}})

#### Hyperparameter search

KNN can be percieved as a voting algorithm. The label of a datum is decided upon a handful its closest neighbours. Two notions ought to be specified, namely, _handful_ and _distance_.

Increasing the number of neighbours, decreases the locality of the model, for further points are included in casting a vote on the class membership. The parameter `n_neighbors` specifies how many neighbours should be taken into account. The _distance_ between points is irrelevant when using the default settings of the KNN estimator. However, one might expect that data closer to a specific datum are more likely to have labels identical to that of the datum in question. If the `weights` parameter  is set to _distance_, points closer the contribute with larger weight to the decision on the class membership.


```python
param_grid = {'pca__n_components' : [6, 8, 10, 12],
              'knn__n_neighbors' : [5, 10, 15],
              'knn__weights' : ['uniform', 'distance']}


grid = GridSearchCV(pipeline_pca, param_grid = param_grid, 
                    cv = cv,
                    scoring = scoring, 
                    refit = False, 
                    return_train_score = False)

_2 = grid.fit(X, y)
```

The variation of accuracy as a response to the hyperparameters is shown below. It plateaus off beyond the tenth principal component in all settings. Taking into account the distance of the neighbours leads to an increase of 1%. It is curious that the accuracy depends negatively on the number of neighbours. This can be attributed to classes overlapping heavily.


![png]({{"/assets/climate-classification/images/output_55_0.png"}})

## Summary

We have created a pipeline to classify climates based on the observed precipitation and temperature data. Using dimensionality reduction techniques and an optimised k-nearest neighbours classifier we have managed to achieve 90% accuracy.

### Further improvements

Considerable amount of time has been sacrificed to choose to generate features, and again some labour was needed to optimise the classifier. In a following post we are going to merge these two steps in a neural network.
