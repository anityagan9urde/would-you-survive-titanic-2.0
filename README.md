# <h1 id="top">Would You Survive The Titanic --version 2.0</h1>
- This is a Flask API developed by me to determine if an onboarder on the infamous Titanic would survive provided their details.
- This version is built upon the previous version which was quite basic in operation.
- ### New features include:
    - New models and model selection options
    - New features for more accurate prediction
    - New form format
- ### Contents:
  - <a href="#dataset">Dataset</a>
  - <a href="#model">Model Used</a>
  - <a href="#api">API</a>
  - <a href="#deployment">Deployment</a>
  - <a href="#learn">What did I learn</a>
> Deploy the API on Heroku by clicking the button below.<br><br> 
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://would-you-survive-titanic-2.herokuapp.com/)

| :exclamation:  Some models are not working currently. Specifically: Random Forest, KNN and Decision Trees. The solution is being worked on.   |
|-----------------------------------------|

### Running the project locally:

##### - Clone this repository. Open CMD. Ensure that you are in the project home directory. Create the machine learning model by running models.py as such:

`python models.py`

##### - This will create a serialized version of our models into files with an extension `.pkl` or use the previously pretrained models saved in the `./models` folder.

##### - Now, run api.py using below command to start Flask API

`python api.py`

##### - Open any browser and paste this URL: `http://localhost:5000` to run the file as an app.
<hr>

#### Following images show how the API will look when run properly:<br>
![]("https://github.com/AnityaGan9urde/would-you-survive-titanic-2.0/index.gif")
<hr>

### <h3 id="dataset">Dataset:</h3>
- The dataset used for training was taken from Kaggle.
- Link: Titanic Dataset: https://www.kaggle.com/c/titanic/data
- #### Features:

| Variable | Definition	| Key|
|---------|--------|-------|
|pclass |	Ticket class |	1 = 1st, 2 = 2nd, 3 = 3rd|
|sex |	Sex 	|
|Age |	Age in years 	|
|sibsp |	# of siblings / spouses aboard the Titanic |	
|parch |	# of parents / children aboard the Titanic |	
|ticket| 	Ticket number 	|
|fare |	Passenger fare |	
|cabin |	Cabin number |	
|embarked| 	Port of Embarkation |	C = Cherbourg, Q = Queenstown, S = Southampton|
- #### Target: 
  - 0 : Not Survived, 
  - 1 : Survived<br>
### <h3 id="model">Models Used:</h3>
- The dataset shows that this is clearly a **classification** task and can be solved by a myriad of classification algorithms such as Logistic Regression, Decision Trees and even Random Forests.
- I chose 6 algorithms to train the dataset on because why not.
- The models which were selected were: Logistic Regression, K-Nearest Neighbours, Gaussian Naive Bayes, Decision trees, Random Forest and Support Vector Machines.
- Model Performances:

>Model |	Score
>------|------
>Random Forest |	86.76
>Decision Tree |	86.76
>KNN |	84.74
>Logistic Regression |	80.36
>Support Vector Machines 	|78.23
>Naive Bayes 	|72.28

### <h3 id="api">API:</h3>
- I made an *API for all the models* so that users can interact and use the Machine Learning models with ease. User can select which model they would like to use during prediction. 
- To make the API work I have used the **Flask** library which are mostly used for such tasks.
- I have also connected a **HTML** form to the flask app to take in user input and a **CSS** file to decorate it.<br>
### <h3 id="deployment">Deployment:</h3><hr>
- The Flask API was deployed on the **Heroku** cloud platform so that anyone with the link to the app can access it online.
- I have connected this GitHub repository to Heroku so that it can be run on the Heroku dyno.
- I have used the **Gunicorn** package which lets Python applications run on any web server. The `Procfile` and `requirements.txt` should be defined with all the details required before the deployment.<br>
### <h3 id="learn">What did I learn:</h3>
- *Data Wrangling* using **Pandas**
- *Feature Engineering* to fit our data to our model
- Saving the model and using it again with **Pickle**
- Making a flask app
- A little frontend web development
- Making the app live by deploying it on cloud platforms 

<button><a href="#top">Back to Top</a></button>
