# Would You Survive The Titanic --version 2.0
## - *Run it directly by clicking here:* [![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://would-you-survive-titanic-2.herokuapp.com/)

## New features include:
- New models and model selection options
- New features for more accurate prediction
- New form format

### **Running the project locally:**

>Open CMD. Ensure that you are in the project home directory. Create the machine learning model by running below command -

    python train.py

This would create a serialized version of all the models into .pkl files. Save them in a folder called `models` in the project directory.

>Run app.py using below command to start Flask API

    python app.py

By default, flask will run on port 5000.

>Navigate to URL http://localhost:5000 on a browser.

You should be able to view the homepage.

Enter all valid values in all input boxes and hit Predict.

If everything goes well, you should be able to see the prediction on the web page and you will know your answer.
