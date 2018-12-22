---
layout: post
title:  "Clustering GDPs"
date:   2018-12-22 07:00:47 +0000
categories: python
---


In this blog post, we are going to have a look at GDPs of various countries and try to group them according to a suitable measure.

The raw notebook can be found [here](https://github.com/bhornung/bhornung.github.io/blob/master/assets/clustering-gdp/notebook/clustering-gdp.ipynb).


```python
from collections import Counter
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score 
from scipy.spatial.distance import squareform, pdist
```


## Data

The purchasing power parity gross domestic product (GDP (PPP)) have been obtained from the paper of [James and Gakidou](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3487911/). The reference currency is the 2005 USD defined according to the IMF.

Time series spanning between 1950 and 2018 contains points for 210 countries or territories (countries for short in the following). 

### Data format

The time series' are loaded to a `pandas` dataframe where the countries are encoded by their three letter abbreviations as row indices. Each year is stored in a separate column.


```python
path_to_db = r'path/to/spreadsheet'
sheet_name = 'Sheet1'

df = pd.read_excel(path_to_db, sheet_name = sheet_name, header = None)

# split columns by countries
groups = df.groupby(0)
df = pd.DataFrame.from_dict({n : g[1].values for n, g in groups}, orient = 'index')

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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
      <th>64</th>
      <th>65</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ABW</th>
      <td>1266.2420</td>
      <td>1327.6650</td>
      <td>1393.9080</td>
      <td>1467.9210</td>
      <td>1546.2200</td>
      <td>1623.0460</td>
      <td>1717.7520</td>
      <td>1792.5970</td>
      <td>1867.8290</td>
      <td>1943.5910</td>
      <td>...</td>
      <td>20333.6900</td>
      <td>20143.8800</td>
      <td>19444.6300</td>
      <td>15729.5800</td>
      <td>16451.7000</td>
      <td>17275.2300</td>
      <td>18179.3200</td>
      <td>19139.9100</td>
      <td>20157.2100</td>
      <td>21228.3800</td>
    </tr>
    <tr>
      <th>AFG</th>
      <td>179.4909</td>
      <td>182.1864</td>
      <td>185.5338</td>
      <td>192.9397</td>
      <td>194.3511</td>
      <td>195.4881</td>
      <td>200.4814</td>
      <td>198.2413</td>
      <td>204.9097</td>
      <td>207.1753</td>
      <td>...</td>
      <td>263.0708</td>
      <td>292.6143</td>
      <td>294.5654</td>
      <td>351.5893</td>
      <td>372.7821</td>
      <td>387.9407</td>
      <td>404.6445</td>
      <td>419.7369</td>
      <td>436.9097</td>
      <td>455.4637</td>
    </tr>
    <tr>
      <th>AGO</th>
      <td>1164.6930</td>
      <td>1192.4080</td>
      <td>1220.7880</td>
      <td>1248.8910</td>
      <td>1212.4560</td>
      <td>1281.8420</td>
      <td>1252.5360</td>
      <td>1338.9390</td>
      <td>1385.0590</td>
      <td>1377.6910</td>
      <td>...</td>
      <td>2289.3820</td>
      <td>2675.2520</td>
      <td>2945.2890</td>
      <td>2878.3360</td>
      <td>2958.8980</td>
      <td>3076.2210</td>
      <td>3174.6280</td>
      <td>3268.8260</td>
      <td>3347.4790</td>
      <td>3384.9220</td>
    </tr>
    <tr>
      <th>AIA</th>
      <td>1443.0830</td>
      <td>1502.3340</td>
      <td>1566.0870</td>
      <td>1637.5250</td>
      <td>1712.6170</td>
      <td>1784.9380</td>
      <td>1875.6700</td>
      <td>1943.4890</td>
      <td>2010.6670</td>
      <td>2077.3580</td>
      <td>...</td>
      <td>13845.4000</td>
      <td>15838.2700</td>
      <td>15682.0600</td>
      <td>11612.8400</td>
      <td>12059.6800</td>
      <td>12573.3900</td>
      <td>13137.4100</td>
      <td>13733.3200</td>
      <td>14360.5000</td>
      <td>15016.1900</td>
    </tr>
    <tr>
      <th>ALB</th>
      <td>778.0425</td>
      <td>809.2832</td>
      <td>814.2191</td>
      <td>845.1029</td>
      <td>869.4968</td>
      <td>912.0341</td>
      <td>923.9318</td>
      <td>976.7833</td>
      <td>1017.5170</td>
      <td>1056.7130</td>
      <td>...</td>
      <td>2731.5010</td>
      <td>2889.3540</td>
      <td>3093.5480</td>
      <td>3179.8590</td>
      <td>3246.2210</td>
      <td>3333.4310</td>
      <td>3436.3720</td>
      <td>3562.8720</td>
      <td>3711.6290</td>
      <td>3877.9590</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>



### General trends

Firstly, we investigate how GDPs have changed over time. To this end, a histogram is created in which each bin corresponds to 2% of the maximum GDP of all years.


```python
def create_histogram(X, bins = None):
    """
    Creates a histogram for all columns.
    Parameters:
        X (np.ndarray (n_row, n_col)) : raw data
        bins (None or np.histogram bins) : the bins. If None, 50 equal width bins are created in 
        the [0, np,max(X)] range
    Returns:
        hst (np.ndarray (50, n_col)) : histogram
    """
    
    if bins is None:
        bins = np.linspace(0, np.max(X), num = 50)
    
    hst = np.array([np.histogram(x, bins = bins, density = True)[0] for x in X.T])
    hst = hst * np.diff(bins)
    return hst
```


```python
histo_year = create_histogram(df.values)
```

![png]({{"/assets/clustering-gdp/images/output_8_0.png"}})


It is readily seen the overall GDP has increased over the years. The distribution has become more spread out.

#### Moments

The median and the first four moments of the GDP distribution are plotted below. It is worth comparing the median and the mean. Both of them steadily increase, (apart from the years following the collapse of the Eastern Block in 1990--1991). However, the median shows an increase of 450% whereas the mean only tripled. This means the countries, in general, became richer with respect to the whole distribution. This is also seen in the overall reduction of skew. The median, however, decreased by ~20% after 1990 compared to the dent of ~2% in the mean. This implies those countries with lower GDPs were hit harder than those with higher GDPs. 


```python
df_year_stats = pd.DataFrame({'mean' : df.mean(axis = 0),
                              'median' : df.median(axis = 0),
                              'std' : df.std(axis = 0), 
                              'skew' : df.skew(axis = 0)})
```


![png]({{"/assets/clustering-gdp/images/output_12_0.png"}})


### Relative GDP

It is revealing to compare the GDPs to the maximum in each year. In doing so, the relative wealth distribution can be assessed.  The `df_scaled_year_stats` lists the median, mean and standard deviation for relative to the highest GDP in each year. 


```python
def scale_by_max(X):
    maxval = np.max(X)
    
    if maxval == 0.0:
        maxval = 0.0
    scaled = X / maxval

    return scaled
```


```python
df_scaled = df.apply(scale_by_max, axis = 0)

df_scaled_year_stats = pd.DataFrame({'maximum' : df.max(axis = 0),
                                     'mean' : df_scaled.mean(axis = 0),
                                     'median' : df_scaled.median(axis = 0),
                                     'std' : df_scaled.std(axis = 0)})
```


```python
histo_scaled_year = create_histogram(df_scaled.values)
```


![png]({{"/assets/clustering-gdp/images/output_17_0.png"}})


The wealth of the countries relative to the wealthiest one become higher as time progresses. In the year of 1950, 70% percent has a GDP at most 2% of the corresponding maximum.  By 2010, this ratio becomes 40%.  

Further signs of equalisation, that the mean doubles and the median triples. It is still astounding, though, that half of the countries only have at most 3% of the maximum GDP. It is also interesting to observe, that the separation between the wealthiest country and the rest has been increasing since the middle of previous decade. This effect was moderated by the crisis in 2008, as shown by the less negative slope of the _mean_ and _std_ curves.


![png]({{"/assets/clustering-gdp/images/output_19_0.png"}})
![png](output_19_0.png)


#### Entropy

Entropy can measure how uniformly the values are distributed between various quantiles. It does not, however, states anything about the values themselves. Whether 90% is in the first and 10% in the last quantiles or _vica versa_, they would both result in the same entropy. Nevertheless, entropy can provide us with a single number indicator, how uniform the GDP distribution is. The entropy increases over the decades, indicating the GDPs becoming less concentrated around one value i.e. countries diverge. (Please note, the bins are defined with respect to the maximum GDP of a given year, ii) if all countries had equal wealth, the entropy would be zero - all values in one slot.) The entropy thus is not suitable to measure inequality. To that end we will invoke the Gini coefficient. 


```python
entropy = calc_entropy(histo_year)
entropy_scaled = calc_entropy(histo_scaled_year)
```

![png]({{"/assets/clustering-gdp/images/output_22_0.png"}})


### Inequality - Lorenz curve

The [Lorenz curve](https://en.wikipedia.org/wiki/Lorenz_curve) is the cumulative wealth as a function of population ratio which possesses that wealth. If the wealth is equally distributed, the curve is a linear between the points (0, 0) and (1, 1) which is also an upper bound of it. The more it deviates from the diagonal, the less equal the wealth distribution is. The Gini coefficient is related to the integral of the Lorenz curve. It expresses the distance from equality; 0 meaning total equality, 1 complete inequality.

The coefficient decreases over time, by a modest 0.1 point. (Remember, the median in terms of the highest GDP was still 3% in 2015!) The late fall of communism leaves behind a number of poor countries with economic struggles which results in the increase of the coefficient for a good twenty years from the early eighties. By 2000, the Gini coefficient starts to decrease again, showing some countries had started to recover.


```python
lorenz_curve = calculate_lorenz_curve(df.values)
gini_coeff = calculate_gini(df.values)
```


![png]({{"/assets/clustering-gdp/images/output_25_0.png"}})


## Uncovering groups of countries

We have so far characterised the distribution. A logical next step is to uncover what components form it; whether there are groups that evolve similarly over the time.

### Temporal characteristics

If there are countries that move together the correlation between rankings in two subsequent should show this. We are going to use Spearman's rho and Kendall's tau correlations. The former takes into account the magnitude of changes in two consecutive rankings, whereas the latter one calculate the number changes in rankings regardless their magnitude. Therefore Kendall's tau is expected to be more sensitive to any variation of the ranking.

The correlation coefficients are calculated for a number increasing delays, namely 1, 5, 10 and 20 years and are shown below. The almost constant nature of Spearman's rho indicates that there are no major reorderings in the countries relative positions. However, the minor - or local, if you wish - changes are magnified by Kendall's coefficient.

The country rankings barely change between two subsequent years. Correlation coefficients at larger gaps are indicative of longer term dynamics. E.g. the five year forward coefficient rapidly decreases from 1985 reaching its minimum at 1989, showing that GDP rankings fluctuated between the years 1990 and 1994. From 1989 the correlation increases as th new order stabilises. The 10 year forward correlation exhibits a small but noticeable slump starting at 1990. This means the rankings starting at 2000 differ more and more to those ten years earlier. The correlation begins to increase at 1995 so there is again a period of stabilization after 1995.


```python
year_spearman_corr = df.corr(method = 'spearman')
year_kendall_corr = df.corr(method = 'kendall')
```


![png]({{"/assets/clustering-gdp/images/output_28_0.png"}})


#### Clustering

It would be interesting to see whether there are groups of GDPs that move together and close to each other over time. Provided there are, finding the GDP most characteristic of the group. A number of clustering methods can achieve these goals. To name a few, [k-medoids](https://en.wikipedia.org/wiki/K-medoids), [Markov cluster algorithm (MCA)](https://micans.org/mcl/), [affinity propagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation). For MCA has been used on numerous occasions in the previous posts, we are going to choose the k-medoids method. A slim implementation thereof can be found in the bottom of this post.



![png]({{"/assets/clustering-gdp/images/output_30_0.png"}})


What distance should be used in the clustering algorithm? Correlation is insensitive to the proximity of the time series. The $L_{2}$ distance ignores the relative magnitude (mean) of the series. E.g. $L_{2}$ distance between series (a) and (b) is the same as between series (b) and (c). However, (c) is only 1.5 greater than (b) whereas (b) is twice the magnitude of (a). 

The overlap, defined below, takes into account both the similarity of the time dependence as correlation does. The spatial distance is also accounted for in the denominator:

$$
D_{ij} = \frac{\int\limits_{t} (f_{i}(t) - f_{j}(t))^{2} \mathrm{d}t}{\int\limits_{t} f_{i}(t)^{2}\mathrm{d}t +  \int\limits_{t} f_{j}(t)^{2}\mathrm{d}t} = 1 - 2 \cdot \frac{\int\limits_{t} f_{i}(t) \cdot f_{j}(t)\mathrm{d}t}{\int\limits_{t} f_{i}(t)^{2}\mathrm{d}t +  \int\limits_{t} f_{j}(t)^{2}\mathrm{d}t}
$$

This measure is implemented as the `overlap` function below.


```python
overlap = lambda u, v: 1 - 2 * (u * v).sum() / ((u * u).sum() + (v * v).sum())
X = squareform(pdist(df.values, overlap))
```

The optimal number of clusters is unknown thus a range of them is tried out. The corresponding results are then saved in the `results` list of dictionaries.


```python
clusterer = KMedoids()

results = []

for n_clusters in range(2, 30):
    
    clusterer.set_params(n_clusters = n_clusters)
    labels = clusterer.fit_predict(X)
    score = clusterer.score
    sil_score = silhouette_score(X, labels)
    
    results.append({'n_clusters' : n_clusters,
                    'labels' : labels,
                    'sil_score' : sil_score,
                    'score' : score,
                    'medoids' :  clusterer.medoids})
```


```python
for idx in range(len(results)):
    
    medoids = results[idx]['medoids'] + 0
    labels = results[idx]['labels'] + 0
    
    i_sort = np.argsort(df.values[medoids].mean(axis = 1))
    labels_ = np.full_like(labels, -1)

    for label, medoid in enumerate(medoids[i_sort]):
        labels_[labels == i_sort[label]] = label
    
    medoids = medoids[i_sort]
    
    results[idx].update({'medoids' : medoids, 'labels' : labels_})
```

#### Determining the number of clusters

There are a manifold of ways to ascertain the optimal number of clusters. One can plot the intra-cluster variance as a function of the number of clusters. This number can be the point where the variance decreases only by small amount compared to the previous values. This method would suggest a five as the optimum number of clusters (left panel of figure below). One can alternatively choose the maximum of the [silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering)). This measure inversely proportional to the intra cluster variance bat also penalises the increasing number of clusters. There is a maximum at $n_{clusters} = 3$ and a secondary maximum at $n_{clusters} = 11$.


![png]({{"/assets/clustering-gdp/images/output_38_0.png"}})


Before making a decision, the first twelve clusters are shown in the figure below. The exemplars of the clusters are drawn with bold lines. The other members of the clusters appear in half tone of the corresponding colors.


```python
# TO HIDE
def plot_line_cluster(lines, medoids, labels, cmap, alpha = 0.1, ax = None):
    
    if ax is None:
        fix, ax = plt.subplots(1, 1)
      
    x = np.arange(lines.shape[1])
    
    lm = np.max(labels)
    legend = r'$n_{0}$ = {1}'.format("{clusters}", lm + 1)
    ax.annotate(legend, (35, 80000))
    
    for label, medoid in enumerate(medoids):
        color = cmap(label / lm)
        
        for line in lines[labels == label]:
            ax.plot(x, line, color = color, alpha = alpha, linewidth = 0.8)
          
        ax.plot(lines[medoid], color = color)
        
    ax.set_xlim((0, x.shape[0] - 1))
    ax.set_ylim((0,100000))
```


![png]({{"/assets/clustering-gdp/images/output_41_0.png"}})


It is instructive to compare the cluster size distributions. The clusters are ordered according to the mean of their exemplars.


```python
sel = [1, 3, 9]
for result in [results[idx] for idx in sel]:
    print("n_clusters:", result['n_clusters'], "cluster sizes:", sorted(Counter(result['labels']).most_common()))
```

    n_clusters: 3 cluster sizes: [(0, 67), (1, 80), (2, 63)]
    n_clusters: 5 cluster sizes: [(0, 46), (1, 48), (2, 42), (3, 36), (4, 38)]
    n_clusters: 11 cluster sizes: [(0, 17), (1, 8), (2, 23), (3, 7), (4, 14), (5, 28), (6, 33), (7, 23), (8, 23), (9, 28), (10, 6)]
    

$n_{clusters} = \{3, 5\}$ splits the time series into groups of roughly equal sizes. These groups are more likely to represent a geometric partitioning rather arising from the underlying structure. At $n_{clusters} = 3$, the third (brown) exemplar clearly falls between two strands of time series in a low density region. When $n_{clusters} = 11$, the clusters have strongly unequal cardinalities which suggests some of the structure is captured. For instance, the tenth - red - exemplar lies in the middle of a band of time series.

The primary separating dimension is the average value of the GDPs as shown in the figure below, where the clusters plotted against the mean and the average annual return (AAR). The AAR was chosen to represent the dynamics of the GDP evolution. 


```python
df_stats = pd.DataFrame(index = df.index)
df_stats['mean'] = df.mean(axis = 1)
df_stats['AR'] = np.mean(df.values[:,1:] / df.values[:, :-1], axis = 1) - 1
df_stats['label'] = results[9]['labels']
```

![png]({{"/assets/clustering-gdp/images/output_46_0.png"}})


A brief statistics summary can be found in the table below.


```python
df_stats.groupby('label')['mean'].describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.0</td>
      <td>233.378319</td>
      <td>61.292314</td>
      <td>115.757428</td>
      <td>193.469405</td>
      <td>246.473200</td>
      <td>289.756198</td>
      <td>301.355706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>367.356715</td>
      <td>79.051810</td>
      <td>237.693017</td>
      <td>335.371645</td>
      <td>347.178652</td>
      <td>431.422191</td>
      <td>487.426953</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23.0</td>
      <td>496.100780</td>
      <td>101.647850</td>
      <td>328.885636</td>
      <td>434.220634</td>
      <td>483.382926</td>
      <td>559.378722</td>
      <td>698.761802</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.0</td>
      <td>785.946523</td>
      <td>85.731001</td>
      <td>665.262229</td>
      <td>734.874499</td>
      <td>792.391648</td>
      <td>827.860350</td>
      <td>918.502082</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.0</td>
      <td>964.386829</td>
      <td>144.046074</td>
      <td>787.202458</td>
      <td>878.532934</td>
      <td>928.248209</td>
      <td>1035.975318</td>
      <td>1321.382727</td>
    </tr>
    <tr>
      <th>5</th>
      <td>28.0</td>
      <td>1592.629372</td>
      <td>271.377049</td>
      <td>1156.011977</td>
      <td>1408.440369</td>
      <td>1530.787083</td>
      <td>1792.837482</td>
      <td>2226.875030</td>
    </tr>
    <tr>
      <th>6</th>
      <td>33.0</td>
      <td>2963.937692</td>
      <td>622.499451</td>
      <td>2086.484017</td>
      <td>2472.128227</td>
      <td>2803.479300</td>
      <td>3375.052118</td>
      <td>4514.176818</td>
    </tr>
    <tr>
      <th>7</th>
      <td>23.0</td>
      <td>5821.337975</td>
      <td>986.966698</td>
      <td>4427.090924</td>
      <td>5039.400379</td>
      <td>5857.104652</td>
      <td>6520.650795</td>
      <td>7589.093045</td>
    </tr>
    <tr>
      <th>8</th>
      <td>23.0</td>
      <td>11364.604139</td>
      <td>2527.604081</td>
      <td>7438.311127</td>
      <td>9628.625083</td>
      <td>10928.945500</td>
      <td>13380.125742</td>
      <td>15862.491985</td>
    </tr>
    <tr>
      <th>9</th>
      <td>28.0</td>
      <td>26226.232545</td>
      <td>6705.804712</td>
      <td>14681.687576</td>
      <td>23062.626447</td>
      <td>24142.139288</td>
      <td>27714.116970</td>
      <td>45418.616364</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6.0</td>
      <td>51998.265131</td>
      <td>22192.680205</td>
      <td>17583.078364</td>
      <td>43456.326061</td>
      <td>51149.112879</td>
      <td>64787.399242</td>
      <td>81882.728182</td>
    </tr>
  </tbody>
</table>
</div>



The countries are as follows in each cluster:


```python
import json
with open(r'path/to/country/codees.json' , 'r') as fproc:
    ccodes = json.load(fproc)
    
df_stats['cname'] = df_stats.index.map(lambda x: ccodes.get(x, x))

for name, group in df_stats.groupby('label')['cname']:
    print(name, group.values)
```

    0 ['Afghanistan' 'Burundi' 'Burkina Faso' 'Bangladesh' 'Eritrea' 'Ethiopia'
     'Guinea' 'Cambodia' "Lao People's Democratic Republic" 'Myanmar'
     'Mozambique' 'Malawi' 'Nepal' 'Rwanda' 'Timor-Leste'
     'Tanzania, United Republic of' 'Uganda']
    1 ['Central African Republic' 'Congo, Democratic Republic of the' 'Liberia'
     'Madagascar' 'Niger' 'Sierra Leone' 'Somalia' 'Togo']
    2 ['Benin' 'Comoros' 'Ghana' 'Gambia' 'Guinea-Bissau' 'Haiti' 'India' 'Kenya'
     'Kyrgyzstan' 'Lesotho' 'Mali' 'Mauritania' 'Nigeria' 'Pakistan' 'Sudan'
     'Senegal' 'Sao Tome and Principe' 'Chad' 'Tajikistan' 'Uzbekistan'
     'Viet Nam' 'Yemen' 'Zimbabwe']
    3 ['Armenia' 'Bhutan' 'China' 'Egypt' 'Indonesia' 'Sri Lanka' 'Mongolia']
    4 ['Bolivia (Plurinational State of)' "CÃ´te d'Ivoire" 'Cameroon' 'Honduras'
     'Kiribati' 'Moldova, Republic of' 'Nicaragua' 'Philippines'
     'Papua New Guinea' "Korea (Democratic People's Republic of)" 'Paraguay'
     'Palestine, State of' 'Solomon Islands' 'Zambia']
    5 ['Angola' 'Albania' 'Azerbaijan' 'Bosnia and Herzegovina' 'Congo'
     'Cabo Verde' 'Djibouti' 'Ecuador' 'Micronesia (Federated States of)'
     'Georgia' 'Guatemala' 'Guyana' 'Iran (Islamic Republic of)' 'Iraq'
     'Jordan' 'Morocco' 'Maldives' 'El Salvador' 'Eswatini'
     'Syrian Arab Republic' 'Thailand' 'Tonga' 'Tunisia' 'Tuvalu' 'Ukraine'
     'Soviet Union' 'Vanuatu' 'Samoa']
    6 ['Argentina' 'Bulgaria' 'Belarus' 'Belize' 'Brazil' 'Botswana' 'Colombia'
     'Costa Rica' 'Cuba' 'Dominica' 'Dominican Republic' 'Algeria' 'Fiji'
     'Equatorial Guinea' 'Grenada' 'Jamaica' 'Kazakhstan' 'Saint Lucia'
     'Marshall Islands' 'Macedonia, the former Yugoslav Republic of'
     'Montenegro' 'Mauritius' 'Malaysia' 'Namibia' 'Panama' 'Peru' 'Romania'
     'Serbia' 'Suriname' 'Turkmenistan' 'Uruguay'
     'Saint Vincent and the Grenadines' 'South Africa']
    7 ['Anguilla' 'Antigua and Barbuda' 'Chile' 'Cook Islands' 'Estonia' 'Gabon'
     'Croatia' 'Hungary' 'Saint Kitts and Nevis' 'Lebanon' 'Lithuania' 'Latvia'
     'Mexico' 'Montserrat' 'Palau' 'Poland' 'Puerto Rico' 'Russian Federation'
     'Slovakia' 'Seychelles' 'Trinidad and Tobago' 'Turkey'
     'Venezuela (Bolivarian Republic of)']
    8 ['Aruba' 'Netherlands Antilles' 'Bahrain' 'Bahamas' 'Barbados' 'Cyprus'
     'Czechia' 'Spain' 'Greece' 'Hong Kong' 'Israel' 'Korea, Republic of'
     'Libya' 'Macao' 'Malta' 'Oman' 'Portugal' 'French Polynesia'
     'Saudi Arabia' 'Singapore' 'Slovenia' 'Turks and Caicos Islands'
     'Taiwan, Province of China']
    9 ['Andorra' 'Australia' 'Austria' 'Belgium' 'Bermuda' 'Brunei Darussalam'
     'Canada' 'Switzerland' 'Cayman Islands' 'Germany' 'Denmark' 'Finland'
     'France' 'United Kingdom of Great Britain and Northern Ireland'
     'Greenland' 'Ireland' 'Iceland' 'Italy' 'Japan' 'Luxembourg'
     'New Caledonia' 'Netherlands' 'Norway' 'New Zealand' 'San Marino' 'Sweden'
     'United States of America' 'Virgin Islands (British)']
    10 ['United Arab Emirates' 'Kuwait' 'Liechtenstein' 'Monaco' 'Nauru' 'Qatar']
    

## Appendix

### Misc functions


```python
def calculate_lorenz_curve(X):
    """
    Calculates the Lorenz curve.
    Parameters:
        X (np.ndarray [pop_size, n_records]) : the wealth distribution over time.
    Returns:
        X_trf (np.ndarray [pop_size, n_records]) : Lorenz curve
    """

    X_trf = X.copy()
    X_trf = np.nancumsum(np.sort(X_trf, axis = 0), axis = 0) / np.nansum(X_trf, axis = 0)

    return X_trf
```


```python
def calculate_gini(X):
    """
    Calculates the Gini coefficient.
    Parameters:
        X (np.ndarray [pop_size, n_records]) : the wealth distribution over time.
    Returns:
        gini_coeff (float) : Gini coefficient.
    """
    
    lc = calculate_lorenz_curve(X)
    pop_size = lc.shape[0]
    
    auc = np.sum(lc, axis = 0) / pop_size
    
    gini_coeff = 1.0 - 2.0 * auc
    
    return gini_coeff
```


```python
def calc_entropy(X):
    """
    Calculates the normalised entropy along the last dimension.
    Parameters:
        X (np.ndarray [n_records, n_categories]) : observations
    Returns:
        entropy (np.ndarray [n_records]) : the normalised entropy
    """
    
    normed = normalize(X)
    
    logp = np.zeros(X.shape, dtype = np.float) 
    logp[normed != 0.0] = np.log(normed[normed != 0.0])
    
    entropy = -np.sum(logp * normed, axis = 1)
    entropy /= np.log(X.shape[1])
    
    return entropy
```

### k-medoids clustering


```python
class KMedoids:
    """
    A slim implementation of the k-medoids algorithm.
    """
    
    def __init__(self, n_clusters = 8, n_init = 10, maxiter = 100):
        """
        Parameters:
            n_clusters (int) : number of clusters
            n_init (int) : number of restarts
            maxiter (int) : maximum number of iterations
        """
        
        self._n_clusters = n_clusters
        self._n_init = n_init
        self._maxiter  = maxiter
        self._labels = None
        self._medoids = None
         
    @property
    def labels_(self):
        return self._labels
    
    @property
    def medoids(self):
        return self._medoids
    
    @property
    def score(self):
        return self._score
        
    def fit(self, X):
        """
        Performs a clustering.
        
        Parameters:
            x (np.ndarray) : distance matrix
        Returns:
            self
        """
        
        # input check
        n_dim = X.ndim
        if n_dim != 2:
            raise ValueError("2D arrays expected. Got: {0}".format(n_dim))
            
        m, n = X.shape
        if n != m:
            raise ValueError("Square distance matrix expected.")
        
        # clear up from possible previous fit
        self._labels = None
        self._medoids = None
        
        # it can only better than this
        score_best = X.sum()
        
        # choose best from batch of fits
        for idx in range(self._n_init): 
            score_, labels_, medoids_ = self._single_fit(X)
            
            if score_ < score_best:
                score_best = score_
                self._medoids = medoids_
                self._labels = labels_
                self._score = score_best
        
        # relabel clusters
        labels = np.full_like(self._labels, -1)
        
        for idx, label in enumerate(self._medoids):
            labels[self._labels == label] = idx
        
        self._labels = labels
        
        return self
      
    def fit_predict(self, X):
        """
        Performs a clustering.
        Parameters:
            X (np.ndarray) : distance matrix
        Returns:
            self.labels_ (np.ndarray of int) : list of cluster labels
        """
        
        return self.fit(X).labels_
     
    def set_params(self, **params):
        """
        Sets the named parameters of the estimator.
        """
        
        for k, v in params.items():
            if k == "maxiter":
                self._maxiter = v
            
            elif k == "n_init":
                self._n_init = v
            
            elif k == "n_clusters":
                self._n_clusters = v
            
    def _single_fit(self, X):
        """
        Performs a single clustering.
        """
        
        medoids  = np.sort(np.random.choice(X.shape[0], self._n_clusters, replace = False))
        lookup = np.arange(X.shape[0])
        
        i_iter = 0
        
        while i_iter < self._maxiter:
            
            i_iter += 1
            d_min = X[medoids[0]] + 0 # deepcopy
            labels = np.full(X.shape[0], medoids[0])
            
            # maximisation step
            # assign datum to closest medoid
            for idx in medoids[1:]:
                mask = d_min > X[idx]
                d_min[mask] = X[idx][mask] + 0 # deepcopy
                labels[mask] = idx

            # expectation step
            # find new medoids

            is_converged = True

            for idx in range(self._n_clusters):
                mask = labels == medoids[idx]
                minidx = np.argmin(X[np.ix_(mask, mask)].sum(axis = 1))
                
                old_medoid = medoids[idx] + 0
                medoids[idx] = lookup[mask][minidx]
        
                is_converged &= medoids[idx] == old_medoid
        
            if is_converged:
                break
    
        # calculate cost function
        cost = d_min.sum()
            
        return cost, labels, medoids 
```
