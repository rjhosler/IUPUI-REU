# Ambulance Allocation Using Wasserstein Clustering of Hawkes Process Intensities
We have developed a clustering and prediction method that forms the back end of an app designed with the Indianapolis EMS department to decrease their overall response time to medical emergencies. Our method uses an online Hawkes process estimation algorithm to model and predict the probability of medical events in specified grid cells. We use calculations of Wasserstein Barycenters to place ambulances in optimal locations during predicted peak event times. Moreover, our method is extendable to other spatial-temporal processes and available as an API for public use. 

### Instillation
1. Requirements
    * Python 3
    * Matplotlib
    * Numpy
    * Scipy
    * Sklearn
    * Pandas
    * Haversine
    * Simplejson, urllib, urllib.request, json, bson, bson_util, pymongo
    * Flask
2. Clone to your local repository

### Usage
#### Point Process
![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/load_train.png )
![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/examine.png )
![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/hotspots.png )
![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/testpredict.png )
![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/update_csv.png )
![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/ex.png )
![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/locs_for_wasserstein.png )

#### Wasserstein

Example:
* cluster = Cluster (data,n)
    * data: what to cluster over
    * n: number of clusters
* cluster.set_centers (centers[:,0:2], len(centers))
    * centers: points to initialize the centers with
* lam = cluster.learn_lam(n_iter = 5, rand_centers =  False)
    * n_iter:  number of iterations
    * rand_centers: setting this to false will restore the centers back to where they were before cluster computation.


#### Brain API
Coming soon :)

### Authors
* marches
* rjhosler

### References
* Mohler, George, and P. Jeffrey Brantingham. “Privacy Preserving, Crowd Sourced Crime Hawkes Processes.” 2018 International Workshop on Social Sensing (SocialSens), 28 May 2018, doi:10.1109/socialsens.2018.00016.
* Cuturi, Marco, and Arnaud Doucet. “Fast Computation of Wasserstein Barycenters.” Proceedings of the 31st International Conference on Machine Learning, JMLR W&CP 32 (2) 2014, 14 June 2014, doi:arXiv:1310.4375v3 [stat.ML] . 
