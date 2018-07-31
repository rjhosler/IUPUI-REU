# Ambulance Allocation Using Wasserstein Clustering of Hawkes Process Intensities
We have developed a clustering and prediction method that forms the back end of an app designed with the Indianapolis EMS department to decrease their overall response time to medical emergencies. Our method uses an online Hawkes process estimation algorithm to model and predict the probability of medical events in specified grid cells. We use calculations of Wasserstein Barycenters to place ambulances in optimal locations during predicted peak event times. Moreover, our method is extendable to other spatial-temporal processes and available as an API for public use. 

### Instillation
* Requirements
   * click==6.7
   * cycler==0.10.0
   * Flask==1.0.2
   * haversine==0.4.5
   * itsdangerous==0.24
   * Jinja2==2.10
   * kiwisolver==1.0.1
   * MarkupSafe==1.0
   * matplotlib==2.2.2
   * numpy==1.15.0
   * pandas==0.23.3
   * pymongo==3.7.1
   * pyparsing==2.2.0
   * python-dateutil==2.7.3
   * pytz==2018.5
   * scikit-learn==0.19.2
   * scipy==1.1.0
   * simplejson==3.16.0
   * six==1.11.0
   * sklearn==0.0
   * Werkzeug==0.14.1
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
   
#### API Usage
* The Amazon Web Service link is brain-env.jezptdbkpz.us-east-2.elasticbeanstalk.com 
##### GET
* The 2 required query parameters are the following: "start_time" and "interval_count". "start_time" is a timestamp formated time to start the point intensity prediction. "interval_count" is the amount of 15 minute intervals to predict. This request will return the 70 by 50 grid of point intensites (3500 points per "interval_count"). An example URL string will yeild the following results:
   * ![alt_txt](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/GET.png )

### Authors
* marches
* rjhosler

### References
* Mohler, George, and P. Jeffrey Brantingham. “Privacy Preserving, Crowd Sourced Crime Hawkes Processes.” 2018 International Workshop on Social Sensing (SocialSens), 28 May 2018, doi:10.1109/socialsens.2018.00016.
* Cuturi, Marco, and Arnaud Doucet. “Fast Computation of Wasserstein Barycenters.” Proceedings of the 31st International Conference on Machine Learning, JMLR W&CP 32 (2) 2014, 14 June 2014, doi:arXiv:1310.4375v3 [stat.ML] . 
