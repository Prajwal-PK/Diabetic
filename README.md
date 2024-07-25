## Diabetic
Ever wondered if glucometer is the only way of knowing weather you are slipping in the circle of being **diabetic** or not?

If yes, then here a possible solution to your question.

Here is a simple machine learning project which can very well predict the possibilities of you being diabetic or not on the basis of some basic information regarding your health.

The dataset used for the project includes following columns:
* No of pregnancies
* Plasma Glucose Level
* Diastolic Blood Pressure (mm Hg)
* Skin Thickness (mm)
* Insulin (U/ml)
* BMI
* Diabetes Pedigree Function
* Age

### ML model
Logistic regression is used for the purpose of predicting the possibility of being diabetic.

Since there were only two classes for the output : **Diabetic** or  **non-diabetis**, that's why logistic regression was used.


### Deployment

Model is deployed using Streamlit framework which also adds simple and clear frontend to the web app.

Check it live: https://diabetic-model.streamlit.app/