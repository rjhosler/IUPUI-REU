# Ambulance Allocation Using Wasserstein Clustering of Hawkes Process Intensities
We have developed a clustering and prediction method that forms the back end of an app designed with the Indianapolis EMS department to decrease their overall response time to medical emergencies. Our method uses an online Hawkes process estimation algorithm to model and predict the probability of medical events in specified grid cells. We use calculations of Wasserstein Barycenters to place ambulances in optimal locations during predicted peak event times. Moreover, our method is extendable to other spatial-temporal processes and available as an API for public use. 

### Instillation
* Requirements
    * Python 3
    * Matplotlib
    * Numpy
    * Scipy
    * Sklearn
    * Pandas
    * Haversine
    * Simplejson, urllib, urllib.request, json, bson, bson_util, pymongo
    * Flask
* Clone to your local repository

### Usage
#### Point Process
* Begin by loading in correctly data in format of Pandas dataframe with labels of DATE_TIME: datetime string,'%Y-%m-%d %H:%M:%S', XCOORD: longitude coordinate, YCOORD: latitude coordinate. Then train the model.
    * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/load_train.png )
* After training, param_examine() shows convergence of Hawkes process theta values as well as variance of day and hour probability vectors used to add periodic trends to the model.
    * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/examine.png )
* Use hotspot_examine() to assess whether the trained model's intensity values produce a reasonable amount of events in reasonable grid locations. (This is not a prediction, just a comparison to data points used for training.)
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/hotspots.png )
* Initialize PointProcessRun class and predict over future events.
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/testpredict.png )
* Update PointProcessRun with new events by loading in a csv with labels of DATE_TIME: datetime string,'%Y-%m-%d %H:%M:%S', XCOORD: longitude coordinate, YCOORD: latitude coordinate.
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/update_csv.png )
* Predictions can be generated through the following methods.
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/ex.png )
* Locations for Wasserstein clustering can also be generated.
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/locs_for_wasserstein.png )

#### Wasserstein
* Here is an example of Wasserstein clustering using locations passed in from PointProcessRun class: 
   * ![alt_text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/wasser.png )

* Running Wasserstein over predicted intensities yields results shuch as this:
   * ![alt_txt](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/wasserstein_graph.png )

### Authors
* marches
* rjhosler

### References
* Mohler, George, and P. Jeffrey Brantingham. “Privacy Preserving, Crowd Sourced Crime Hawkes Processes.” 2018 International Workshop on Social Sensing (SocialSens), 28 May 2018, doi:10.1109/socialsens.2018.00016.
* Cuturi, Marco, and Arnaud Doucet. “Fast Computation of Wasserstein Barycenters.” Proceedings of the 31st International Conference on Machine Learning, JMLR W&CP 32 (2) 2014, 14 June 2014, doi:arXiv:1310.4375v3 [stat.ML] . 
