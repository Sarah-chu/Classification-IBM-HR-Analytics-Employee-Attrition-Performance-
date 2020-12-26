# Classification--IBM HR Analytics Employee Attrition Performance
IBM HR Analytics Employee Attrition & Performance
Classification problem on Employee Attrition 
Based on the fictional dataset provided by IBM found on Kaggle :
https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

In this project, we evaluate our model based on **recall(sensitivity)** as we want to know how well the model detect the employee quitted(TP) from who actually quitted (TP + FN).
It is the proportion of positive result that were correctly classified from all the actual positive.
It is more useful when the class is *imbalanced*, as true negative is not included in the calculation and thus recall is not affected the imbalance.
In this dataset, we have lots of 'No Attrition' relative to 'Yes' and we would like to predict 'Yes' (i.e leave the company) precisely.
With recall , we can evaluate the models by looking at how well they predict an actual 'Yes' from all the actual positive (True positive and false negative)

We select **Gaussian Naive Bayes** as the best model from Logistic Regression, KNN, Decision Tree and Random Forest, based on sensitivity(recall).

In short, the key findings can be found below:


## Summary   


###### Influential Factors of Attrition
Our study finds out the employees’ major concerns as following : 
 1) Employees’ background 
    a) Age 
    b) Working experience
 2) Employees’ working conditions
    a) Working overtime
    b) Job role
    c) Job levels
 3) Employees’ benefits
    a) Salary
    b) Stock options


###### Recommendations 
 We suggest the following measures to improve the employees morale and the attrition rate :
 1) Career counseling and coaching for new employees
 2) Stock Options for junior employees
 3) Working time or workload adjustment
 4) Employees' caring scheme
 5) Company promotion system
 6) Working environment improvement
