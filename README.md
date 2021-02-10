# Classification--IBM HR Analytics Employee Attrition Performance<br/>
## IBM HR Analytics Employee Attrition & Performance<br/> Classification problem on Employee Attrition <br/>
Based on the fictional dataset provided by IBM found on Kaggle :<br/>
https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset<br/>

In this project, we evaluate our model based on **recall(sensitivity)** as we want to know how well the model detect the employee quitted(TP) from who actually quitted (TP + FN).<br/>
It is the proportion of positive result that were correctly classified from all the actual positive.
It is more useful when the class is *imbalanced*, as true negative is not included in the calculation and thus recall is not affected the imbalance.<br/>
In this dataset, we have lots of 'No Attrition' relative to 'Yes' and we would like to predict 'Yes' (i.e leave the company) precisely.
With recall , we can evaluate the models by looking at how well they predict an actual 'Yes' from all the actual positive (True positive and false negative).<br/>

We select **Gaussian Naive Bayes** as the best model from Logistic Regression, KNN, Decision Tree and Random Forest, based on sensitivity(recall).<br/>

In short, the key findings can be found below:<br/>


## Summary   <br/>


###### Influential Factors of Attrition<br/>
Our study finds out the employees’ major concerns as following : <br/>
 1) Employees’ background <br/>
    a) Age <br/>
    b) Working experience<br/>
 2) Employees’ working conditions<br/>
    a) Working overtime<br/>
    b) Job role<br/>
    c) Job levels<br/>
 3) Employees’ benefits<br/>
    a) Salary<br/>
    b) Stock options<br/>


###### Recommendations 
 We suggest the following measures to improve the employees morale and the attrition rate :<br/>
 1) Career counseling and coaching for new employees<br/>
 2) Stock Options for junior employees<br/>
 3) Working time or workload adjustment<br/>
 4) Employees' caring scheme<br/>
 5) Company promotion system<br/>
 6) Working environment improvement <br/>
