# ML-classification-mealkit-DTC
This repo is based on the second project of a business case elaborated for the Machine Learning course at Hult International Business School by the professor Chase Kusterer.

The business case is based on Apprentice Chef, Inc. - a fictitious meal kit delivery company.

This project involved developing a predictive model and understanding what drives customers to purchase Apprentice Chef's newest product - Halfway There. Halfway There is a cross-selling promotion where subscribers receive a half bottle of wine from a local California vineyard every Wednesday (halfway through the workweek). This promotion is measured as the success of promoting Halfway There (1 = SUCCESS, 0 = FAIL).

I approached this problem by focusing on acquiring Domain Knowledge and focusing on engineering rich features that captured the variance of the outcome variable (successful cross-sell of Halfway There).

# Files in the repo
<a href="https://github.com/yvetteyyuan/ML-classification-mealkit-DTC/blob/master/Yi_Yuan_A2_Analysis.ipynb">Yi_Yuan_A2_Analysis.ipynb</a> : Jupyter Notebook with the analysis. This analysis has a heavy focus in feature engineering. Engineered features were essential for the final model, as well as for the the data-driven insights. I focused on developing a simple, explainable model in order to have more interpretability.

<a href="https://github.com/yvetteyyuan/ML-classification-mealkit-DTC/blob/master/Yi%20Yuan%20A2_Write_Up.pdf">Yi Yuan A2_Write_Up.pdf</a> : Contains the data-driven insights based on the business problem.

<a href="https://github.com/yvetteyyuan/ML-classification-mealkit-DTC/blob/master/Apprentice_Chef_Dataset.xlsx">Apprentice_Chef_Dataset.xlsx</a> : Contains the data provided by the data science team.

<a href="https://github.com/yvetteyyuan/ML-classification-mealkit-DTC/blob/master/Apprentice_Chef_Data_Dictionary.xlsx">Apprentice_Chef_Data_Dictionary.xlsx</a> : Metadata of each feature found in the dataset.

# Context about the dataset
Apprentice Chef, Inc. is an innovative company with a unique spin on cooking at home. Developed for the busy professional that has little to no skills in the kitchen, they offer a wide selection of daily-prepared gourmet meals delivered directly to your door. Each meal set takes at most 30 minutes to finish cooking at home and also comes with Apprentice Chef's award- winning disposable cookware (i.e. pots, pans, baking trays, and utensils), allowing for fast and easy cleanup. Ordering meals is very easy given their user-friendly online platform and mobile app.

# Business problem
In an effort to diversify their revenue stream, Apprentice Chef, Inc. has launched Halfway There, a cross-selling promotion where subscribers receive a half bottle of wine from a local California vineyard every Wednesday (halfway through the work week). The executives at Apprentice Chef also believe this endeavor will create a competitive advantage based on its unique product offering of hard to find local wines.

Halfway There has been exclusively offered to all of the customers in the dataset you received, and the executives would like to promote this service to a wider audience. They have tasked you with analyzing their data, developing your top insights, and building a machine learning model to predict which customers will subscribe to this service.
