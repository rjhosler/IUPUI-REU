# RaspBary: Hawkes Point Process Wasserstein Barycenters as a Service
We have developed a clustering and prediction method that forms the back end of an app designed with the Indianapolis EMS department to decrease their overall response time to medical emergencies. Our method uses an online Hawkes process estimation algorithm to model and predict the probability of medical events in specified grid cells. We use calculations of Wasserstein Barycenters to place ambulances in optimal locations during predicted peak event times. Moreover, our method is extendable to other spatial-temporal processes and available as an API for public use. 
![alt_text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/allocation_looped.gif)

### Installation 
* [Requirements](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/requirements.txt): Runs on Python 3
* Clone to your local repository

### Usage
#### Point Process
* Begin by loading in correctly data in format of Pandas dataframe with labels of DATE_TIME: datetime string,'%Y-%m-%d %H:%M:%S' (or UNIX timestamp), XCOORD: longitude coordinate, YCOORD: latitude coordinate. Then train the model.
    * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/load_train.png )
* After training, param_examine() suggests parameters to examine for evaluating convergence of Hawkes process values as well as variance of day and hour probability vectors used to add periodic trends to the model.
    * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/examine.png )
* Use hotspot_examine() to assess whether the trained model's intensity values produce a reasonable amount of events in reasonable grid locations. (This is not a prediction, just a comparison to data points used for training.)
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/hotspots.png )
* Initialize PointProcessRun class and run a test prediction.
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/testpredict_fixed.png )
* Update PointProcessRun with dataframe containing labels DATE_TIME: datetime string,'%Y-%m-%d %H:%M:%S' (or UNIX timestamp), XCOORD: longitude coordinate, YCOORD: latitude coordinate.
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/update.png)
* Predictions can be generated through the following methods.
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/predictions.png )
* Locations for Wasserstein clustering can also be generated.
   * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/wasser_locs.png )

#### Wasserstein
* Here is an example of Wasserstein clustering using locations passed in from PointProcessRun class: 
   * ![alt_text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/wasser.png )

* Running Wasserstein over predicted intensities yields results shuch as this:
   * ![alt_txt](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/ex_proj.png )
   
#### API Usage
* We have configured our code to run on Amazon Web Services.
* NOTE: timestamps will be converted to UTC.
* This is the main idea:
    * ![alt text](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/system.png )
##### Intensity Projections
* The 2 required query parameters are the following: "start_time" and "interval_count". "start_time" is a timestamp formated time to start the point intensity prediction. "interval_count" is the amount of 15 minute intervals to predict. This request will return the 70 by 50 grid of point intensites (3500 points per "interval_count"). An example URL string will yeild the following results:
   * URL: https://server_address/emergencies?start_time=1485798259&interval_count=1
   * ![alt_txt](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/GET.png )
#### Vehicle Locations
* Utilizing the post request is easy to use in [Postman](https://www.getpostman.com). Here you will need to send in a json file of the current truck locations. The Postman environment should look like this:
   * ![alt_txt](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/Post_Usage_nolink.png)
* The "start_time" parameter will specify when the point process will predict point intensities. This trucks will allocate over this data. "interval_count" determines the length of time to predict while "interval_time" sets the length of each interval (NOTE: this parameter is ignored and the default 15 will always be used). For the current truck positions, "trucks" contains a list of each trucks' geographical location. The "virtual" parameter determines if the truck is allocatable.
* Sending the POST request will yeild results like this:
   * ![alt_txt](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/Post_Result.png )
* This returns a list of objects called "TruckSchema." Each object has the current location and the assigned location.
#### Update Model
##### CSV Upload
* The model can be updated through CSV upload using the login.html file.
* Simply load the correctly formatted CSV: 
  * ![alt_txt](https://github.com/rjhosler/IUPUI-REU/blob/repository_images/csv.png)
##### GET
* The model can also be updated through a get request with URL string parameters xcoord (longitude), ycoord (latitude) and unix timestamp.
* Example URL: http://server_address/SingleProcessUpdate?xcoord=-86.43&ycoord=39.14&timestamp=1532959162.


### Authors
* marches
* rjhosler

### References
* Mohler, George, and P. Jeffrey Brantingham. “Privacy Preserving, Crowd Sourced Crime Hawkes Processes.” 2018 International Workshop on Social Sensing (SocialSens), 28 May 2018, doi:10.1109/socialsens.2018.00016.
* Cuturi, Marco, and Arnaud Doucet. “Fast Computation of Wasserstein Barycenters.” Proceedings of the 31st International Conference on Machine Learning, JMLR W&CP 32 (2) 2014, 14 June 2014, doi:arXiv:1310.4375v3 [stat.ML] . 

### Citation
* [Please cite this paper](https://github.com/rjhosler/IUPUI-REU/blob/master/RaspBary.pdf) if you plan on using this software in an academic project.
