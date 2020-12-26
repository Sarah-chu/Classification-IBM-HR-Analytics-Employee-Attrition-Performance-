########################################################
## IBM HR Analytics Employee Attrition & Performance ##
########################################################

# Employee attrition has always been a human resources problem as it leads to
# higher training and recruitment cost, as well as the loss of talents. With the advance of 
# machine learning, employee attrition becomes foreseeable. Human resources department can 
# have a deeper understanding in the possible attrition reasons and try to make a retention plan
# for those who are potentially leaving.
# In this study, we will try to predict if an employee would resign
# in the company based on four aspects : employees' background, employees' working condition, 
# employees' satisfaction and employees' benefits. We will also try to identify the possible
# attributes of the attrition decision and give the corresponding insights.


### Table of Contents :
# 1) Data Preparation
# 2) Exploratory Data Analysis (EDA)
#    a) Employees' Background
#    b) Employees' Working Condition
#    c) Employees' Satisfaction
#    d) Employees' Benefits
# 3) Feature Engineering
# 4) Model Building
#    a) Logistic Regression
#    b) Naive Bayes 
#    c) K-Nearest Neighbors (KNN)
#    d) Decision Tree 
#    e) Random Forest
# 5) Model Evaluation and Selection
# 6) Key Findings
# 7) Summary



#------------------------#
#     Import the data    #
#------------------------#

## Load the package needed
library(ggplot2)
library(data.table)
library(magrittr)
library(dplyr)
library(car)
library(caTools)
library(corrplot)
library(e1071)
library(caret)
library(class)
library(randomForest)
library(vip)
library(naivebayes)
library(rpart) 
library(rpart.plot)
library(cowplot)

## Import data
data_original <- read.csv("Employee-Attrition.csv")

## Check data type and missing value
str(data_original)
table(is.na(data_original)) 
# No missing value is found

## Data cleaning: Delete unnecessary columns
data <- subset(data_original, select = -c(DailyRate,HourlyRate,EmployeeCount,
                                          EmployeeNumber,Over18,StandardHours))
str(data)

## Rename Age column
data <- data %>% rename(Age = `ï..Age`)


#-------------------------------------------#
####   Exploratory Data Analysis (EDA)   ####
#-------------------------------------------#

## Overview of the dataset
summary(data)
# We can find that 1233 employees didn't leave the company while 237 did quit.
# The company has three departments, with 882 men, 588 women and an average age of 36.
# Most of the staff are life science and medical graduates

## How many employees have left?
table(data$Attrition)
prop.table(table(data$Attrition))
# 83.9% of employees didn't quit the company while 16.1% did leave.

### Employees' background ###

## Age
ggplot(data, aes(x = Attrition, y =Age, fill = Attrition)) +
  geom_violin(alpha = 0.7) +
  labs(
    title = "Attrition by Age",
    x = "Attrition",
    y = "Age") +
  theme_classic()
# We can see that the majority of employees who quit (at around 30) are younger than 
# the majority of employees who stay (at around 35).
# Also the proportion of employees in their 20s who quit are higher than those staying.
# On the other hand, employees older than 35 years old follow a similar distribution for both leaving than staying

## Gender 
ggplot(data, aes(Gender, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Gender",
    x = "Gender",
    y = "Number of Employees") +
  theme_classic()
# Gender seems not to be a possible influential feature as both gender have a similar ratio of Attrition

## Marital status
# Did marital status affect employee attrition?
ggplot(data, aes(MaritalStatus, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Marital Status",
    x = "Marital Status",
    y = "Number of Employees") +
  theme_classic()

prop.table(table(data$MaritalStatus,data$Attrition), margin = 1)
# Single has the highest attrition rate(25.5%) among 2 other groups.

prop.table(table(data$MaritalStatus,data$Attrition), margin = 2)
# 48% of those who stay are married whereas
# 50.6% of those who quit the company were single 

## Education
# 1 = 'Below College' ; 2 = 'College' ; 3 = 'Bachelor' ; 4 = 'Master' ; 5 = 'Doctor'
ggplot(data, aes(Education, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Education",
    x = "Education",
    y = "Number of Employees") +
  coord_flip() +
  theme_classic() 
# Most of employees' education are bachelor or master

prop.table(table(data$Education,data$Attrition), margin = 1)
# For employees whose education were below college, they had the highest possibility to quit(18.2%) while 17.3% for bachelor education

prop.table(table(data$Education,data$Attrition), margin = 2)
# Almost 41.7% of those who left the company had bachelor education while 24.5% had master education.

## Education Field
# Which field of education employees is more likely to leave their jobs?
ggplot(data, aes(EducationField, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Education Field",
    x = "Education Field",
    y = "Number of Employees") +
  coord_flip() +
  theme_classic()
# Most of employees come from the fields of life sciences and medical 

prop.table(table(data$EducationField,data$Attrition), margin = 1)
# In all areas of education, human resources(25.9%) and technical(24.2%) fields have the highest attrition rate.

prop.table(table(data$EducationField,data$Attrition), margin = 2)
# 64.2%(life sciences: 37.6%; medical: 26.6%) of those who quit the company come from these two fields of education.

## Number of Companies Worked
ggplot(data, aes(NumCompaniesWorked, fill = Attrition)) +
  geom_histogram(binwidth = 1, position = position_dodge()) +
  xlab('No. of Companies')+ ylab('No. of Employees') +
  scale_x_continuous(limits = c(-1,10), breaks=seq(0,10,1)) + 
  ggtitle('Attrition by Number of Companies Worked') +
  theme_classic()
# For employees who leave, 1 company is the highest while other numbers are similar;
# For employees who stay, 1 company is also the highest but the distribution is right-skewed,
# majority of that group have worked for less than 5 companies
# The distributions are different among 2 groups, there may be a relationship to the attrition.

## Total Working Years
ggplot(data, aes(TotalWorkingYears, fill = Attrition)) +
  geom_histogram(binwidth = 1, position = position_dodge()) +
  xlab('No. of Years')+ ylab('No. of Employees') +
  scale_x_continuous(limits = c(-1,40), breaks=seq(0,40,1)) + 
  ggtitle('Attrition by Total Working Years') +
  theme_classic()
# Two groups (quit and stay) follows a similar distribution : 
# right-skewed but the highest number of years is 10 year for stay group;
# the highest is 1 years for quit group.
# Employees with 1 total working year have around half chance to leave.


### Employees' working condition in the company ###

## Business Travel
# Is frequent business travel related to employee attrition? 
ggplot(data, aes(BusinessTravel, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Frequency of Business Travel",
    x = "Frequency",
    y = "Number of Employees") +
  theme_classic()
# Most of employees rarely have business travel

prop.table(table(data$BusinessTravel,data$Attrition), margin = 1)
# Employees with frequent business travel has a higher proportion of attrition than other 2 groups

prop.table(table(data$BusinessTravel,data$Attrition), margin = 2)
# But employees with frequent business travel is the the major of those who are leabing the company,
# as 65.8% of employees leaving rarely travel instead of travel frequently

## Department
# Which department has the highest attrition rate?
ggplot(data, aes(Department, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Department",
    x = "Department",
    y = "Number of Employees") +
  theme_classic()
# The R&D department has the largest number of employees.

prop.table(table(data$Department,data$Attrition), margin = 1)
# Among three departments, the sales department has the highest attrition rate(20.6%),
# followed by Human Resources with 19.0%.
# Actually, three departments has similar attrition rate at around 13-20%,
# we do not expect this as an influential feature

prop.table(table(data$Department,data$Attrition), margin = 2)
# Half of those(56.1%) who left their jobs are from research & development department.

## Overtime
ggplot(data, aes(OverTime, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(title = "Overtime Situation" , y ="Number of Employees") +
  theme_classic()

prop.table(table(data$OverTime))
# Only 28.3% of the people in the company had worked overtime.

prop.table(table(data$OverTime, data$Attrition), 2)
# About half of the resigned employees had worked overtime,
# but 76.56% of those who stay do not work overtime.
# This feature may distinguish who is leaving the company as the ratio of overtime is 
# different for 2 attrition groups

## Performance Rating
# 1 = 'Low' ; 2 = 'Good' ; 3 = 'Excellent' ; 4 = 'Outstanding'
ggplot(data, aes(factor(PerformanceRating), fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Performance Rating",
    x = "Performance Rating",
    y = "Number of Employees") +
  theme_classic()
# The employees of this company have all received excellent or outstanding performance evaluations.

prop.table(table(data$PerformanceRating,data$Attrition),margin = 1)
# Employees score either 3 or 4 have similar probability of leaving the company at 16% 
# Thus this feature may not be a good indicator of attrition

prop.table(table(data$PerformanceRating,data$Attrition),margin = 2)
# Among the quitted employees, 84% get a score of 3 (excellent rating)

## Job level
ggplot(data, aes(JobLevel , fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Job Level",
    x = "Job Level",
    y = "Number of Employees") +
  theme_classic()

prop.table(table(data$JobLevel,data$Attrition),margin = 1)
# Lower job level employees have higher attrition rate:
# job level 1 has the highest attrition rate among 5 job levels

prop.table(table(data$JobLevel,data$Attrition),margin = 2)
# 60.3% of those who quit the company are at job level 1
# This feature seems to be a possible predictor of attrition

## Job Role
ggplot(data, aes(JobRole, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Job Role",
    x = "Job Role",
    y = "Number of Employees") +
  coord_flip() +
  theme_classic()

table(data$JobRole,data$JobLevel)
prop.table(table(data$JobRole,data$Attrition),margin = 1)
# Sales Representative has the highest attrition rate at 40%, 
# followed by Human Resources and Laboratory Technician at around 23%;
# Research Director has lowest attrition rate at 2.5%

prop.table(table(data$JobRole,data$Attrition),margin = 2)
# Laboratory technical(26%) and sales executive(24%) contribute to half of the number of employees leaving

## Distance From Home
density1 <- data %>%
  filter( Attrition == 'Yes') %>%
  ggplot( aes(x=DistanceFromHome)) +
  geom_density(fill="blue", color="#e9ecef", alpha=0.5) +
  ggtitle("Distance From Home of Employees Stayed") +
  theme_bw()
density2 <- data %>%
  filter( Attrition == 'No') %>%
  ggplot( aes(x=DistanceFromHome)) +
  geom_density(fill="pink", color="#e9ecef", alpha=1) +
  ggtitle("Distance From Home of Employees Quitted") +
  theme_bw()
plot_grid(density1, density2, nrow=2)
# Majority of employees( both stay and quit) live close to the office
# The distribution between 2 group do not differ significantly
# This factor does not seem like an important feature

## Training Times Last Year
ggplot(data, aes(factor(TrainingTimesLastYear) , fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Training Hours",
    x = "Hours of training",
    y = "Number of Employees") +
  theme_classic()

prop.table(table(data$TrainingTimesLastYear,data$Attrition),margin = 1)
# The less the training session, the higher the attrition rate in general,
# except 4 training sessions with 21.13% attrition rate.

prop.table(table(data$TrainingTimesLastYear,data$Attrition),margin = 2)
# The largest percentage, 41.3%, of people who quit received 2 training sessions
# While the second is with 3 sessions at 29.1%.
# There is a large drop off once someone receives more than 3 training sessions on the job.

## Years At Company  
ggplot(data, aes(YearsAtCompany , fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Years at Company",
    x = "Number of Years",
    y = "Number of Employees") +
  theme_classic()

prop.table(table(data$YearsAtCompany,data$Attrition),margin = 1)
prop.table(table(data$YearsAtCompany,data$Attrition),margin = 2)

# Most of the attrition at the company happens in the first 10 years of people working there at 91.51% of
# the people who quit do it before their 11th year. 
# The year most people leave is between their first and second year
# on the job with 24% of the people quitting do it then.

## Years In Current Role
ggplot(data, aes(YearsInCurrentRole, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Years in Current Role",
    x = "Number of Years",
    y = "Number of Employees") +
  theme_classic()

prop.table(table(data$YearsInCurrentRole,data$Attrition),margin = 1)
prop.table(table(data$YearsInCurrentRole,data$Attrition),margin = 2)

# Most people leave their job either within their first year, 30.8% or in their second year, 28.69%
# There is also another spike of people leaving in their 7th year at their current position, at 13.1%

## Years Since Last Promotion
ggplot(data, aes(YearsSinceLastPromotion, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Years Since Last Promotion",
    x = "Number of Years",
    y = "Number of Employees") +
  theme_classic()

prop.table(table(data$YearsSinceLastPromotion,data$Attrition),margin = 1)
prop.table(table(data$YearsSinceLastPromotion,data$Attrition),margin = 2)

# Most people who leave, 46.4%, quit after less than one year after getting a promotion.
# The next largest group leaves after one year of getting a promotion at 20.7%. 
# The last largest group leaves two years after their last promotion at 11.4%. 
# It was odd to see that people would quit a job within one year of getting a promotion.
# It was investigated to see if there was some connection between how long people
# worked at the company, when was their last promotion and if they quit.


ggplot(data[which(data$YearsAtCompany >= 1),], aes(YearsSinceLastPromotion, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Years Since Last Promotion",
    subtitle = "Only showing people who worked there for more than one year",
    x = "Number of Years",
    y = "Number of Employees") +
  theme_classic()

table(data[which(data$YearsAtCompany >= 1),]$YearsSinceLastPromotion, data[which(data$YearsAtCompany >= 1),]$Attrition)
# This shows that no, most people who quit worked for more than one year at the company

ggplot(data[which(data$YearsSinceLastPromotion == 0),], aes(YearsAtCompany, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Years Since Last Promotion",
    subtitle = "People who received a promotion within the last year, and quit",
    x = "Number of Years",
    y = "Number of Employees") +
  theme_classic()

table(data[which(data$YearsSinceLastPromotion == 0),]$YearsAtCompany, data[which(data$YearsSinceLastPromotion == 0),]$Attrition)
prop.table(table(data[which(data$YearsSinceLastPromotion == 0),]$YearsAtCompany, data[which(data$YearsSinceLastPromotion == 0),]$Attrition))

# This shows that most people get a promotion and then leave within one year of getting that promotion are the people who have only worked at the company 
# for one year, but not less than one year. There could be something set up in the company that after one year people automatically
# receive a promotion. Other than that it does not seem like there is a real connection between attrition rate and since
# a person's last promotion.

## Years With Current Manager
ggplot(data, aes(YearsWithCurrManager, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Years with Current Manager",
    x = "Number of Years",
    y = "Number of Employees") +
  theme_classic()

prop.table(table(data$YearsWithCurrManager,data$Attrition),margin = 1)
prop.table(table(data$YearsWithCurrManager,data$Attrition),margin = 2)

# Most people quit when they had the same manager for less than a year, 35.86%, after two years, 21.09% and after 7 years, 13.08%.


### Employees' Satisfaction in the Company ###

## Relationship satisfaction 
# Interpretation : how satisfy is the employee with the colleagues
# 1 = 'Low' ; 2 = 'Medium' ; 3 = 'High' ; 4 = 'Very High'
ggplot(data, aes(RelationshipSatisfaction, fill = Attrition)) +
  geom_histogram(binwidth = 1, position = position_dodge()) +
  labs(y = 'No. of Employees', x = 'Relationship Satisfaction') +
  ggtitle('Attrition by Relationship Satisfaction') +
  theme_classic()
# The proportion of level 3 & 4 for employee quit are larger than that of level 1 & 2 which
# is the same as the employees who stayed. But there are more employee left with low satisfaction than medium satisfaction 


## Environment Satisfaction
# Interpretation : how satisfy is the employee with the working environment
# 1 = 'Low' ; 2 = 'Medium' ; 3 = 'High' ; 4 = 'Very High'
ggplot(data, aes(EnvironmentSatisfaction, fill = Attrition)) +
  geom_histogram(binwidth = 1, position = position_dodge()) +
  labs(y = 'No. of Employees', x = 'Environment Satisfaction') +
  ggtitle('Attrition by Environment Satisfaction') +
  theme_classic()

prop.table(table(data$EnvironmentSatisfaction, data$Attrition), margin =1 )
# Most people have an environmental satisfaction of either 3 or 4 for their working environment.
# Employees with low environment satisfaction ('1') have the highest attrition rate at 25.35%
# whereas employees with other satisfaction levels get 14% attrition rate.


## Job Involvement
# Interpretation : at what extent the employee feel involved in the job
# 1 = 'Low' ; 2 = 'Medium' ; 3 = 'High'; 4 = 'Very High'
ggplot(data, aes(JobInvolvement, fill = Attrition)) +
  geom_histogram(binwidth = 1, position = position_dodge()) +
  ylab('No. of Employees') +
  ggtitle('Attrition by Job Involvement') +
  theme_classic()

prop.table(table(data$JobInvolvement, data$Attrition),1)
# Around one third of the people who report a job involvement of 1 leave.
# The lower the job involvement rating, the higher the attrition rate of that group

prop.table(table(data$JobInvolvement, data$Attrition),2)
# Most people who leave have a job involvement of 3(52.74%), with a steep drop off of attrition with an involvement of 4.


## Job Satisfaction
# Interpretation : how satisfy is the employee with the job
# 1 = 'Low'; 2 = 'Medium'; 3 = 'High'; 4 = 'Very High'
ggplot(data, aes(JobSatisfaction, fill = Attrition)) +
  geom_histogram(binwidth = 1, position = position_dodge()) +
  ylab('No. of Employees') +
  ggtitle('Attrition by Job Satisfaction') +
  theme_classic()

prop.table(table(data$JobSatisfaction, data$Attrition), 1)
# The lower the job satisfaction rating, the higher the attrition rate of that group.
# The attrition rate of 'Very high' rating is a double of that of 'Low' rating.

prop.table(table(data$JobSatisfaction, data$Attrition), 2)
# Most employees staying report a job satisfaction of 3 and 4 while 30% of employees quit
# give job satisfaction 1.

## Work Life Balance
# Interpretation : how do the employee feel about work-life balance in the company
# 1 = 'Bad' ; 2 = 'Good' ; 3 = 'Better' ; 4 = 'Best'
ggplot(data, aes(WorkLifeBalance, fill = Attrition)) +
  geom_histogram(binwidth = 1, position = position_dodge()) +
  ylab('No. of Employees') +
  ggtitle('Attrition by Work Life Balance') +
  theme_classic()

prop.table(table(data$WorkLifeBalance, data$Attrition), 1)
# The lower the work life balance rating, the higher the attrition rate of that group.
# "Bad" Work life balance group has 31.25% attrition rate which is double of other three groups

prop.table(table(data$WorkLifeBalance, data$Attrition), 2)
# Most people staying report a 3 work life balance, so as the employees who leave the company


### Employees' Benefit ###

## Monthly Income
ggplot(data, aes(MonthlyIncome , colour = Attrition)) +
  geom_density(binwidth = 1) + 
  labs(title=" Monthly Income Distribution by Attrition")
# There was no significant difference between the two groups in the distribution trend of monthly wages above 5000,
# but low paid employees (Monthly Income < 5000) accounted for a large proportion of those who left the company.

## Monthly Rate 
# Monthly rate is for internal calculation of the labour cost
ggplot(data, aes(MonthlyRate , colour = Attrition)) +
  geom_density(binwidth = 1) +
  labs(title=" Monthly Rate Distribution by Attrition")
# Most employees have a monthly rate between 5,000 and 20,000.
# The distribution between 2 groups doesn't differ significantly.

## Percent Salary Hike
# Interpretation : The percentage of change in salary between 2 year (2017, 2018)
ggplot(data, aes(PercentSalaryHike , fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Percentage Change in Salary",
    x = "Percentage Increase in Salary",
    y = "Number of Employees") +
  theme_classic()
prop.table(table(data$PercentSalaryHike,data$Attrition),margin = 2)
# The distribution of salary change is right-skewed, majority of employees 
# get 10-15% increase in salary regardless of attrition

prop.table(table(data$PercentSalaryHike,data$Attrition),margin = 1)
# Almost each percentage of salary increase has a attrition rate at around 20% ,
# except 14%,20,21% and 25% of salary increase groups have lower attrition rate

## Stock option level
# Interpretation : How much company stocks you own from this company
ggplot(data, aes(StockOptionLevel, fill = Attrition)) +
  geom_bar(position = position_dodge()) +
  labs(
    title = "Attrition by Stock Option Level",
    x = "Stock Option Level",
    y = "Number of Employees") +
  theme_classic()
# Employees usually get either level 0 or 1 of the company stocks : 
# employees who stay get level 1 are more than those get level 0 , and which is almost 3 times more than level 2;
# employees who quit get level 0 are far more than other stock option levels

### Correlation between numerical variables ###
## Grab only numeric columns
num.cols <- sapply(data, is.numeric)

## Filter to numeric columns for correlation
cor.data <- cor(data[, num.cols])
corrplot(cor.data, method = "color")
# The darker the blue colour is, the higher the correlation
# Years At Company, Years In Current Role, Years Since Last Promotion ,Years With Current Manager are highly correlated
# which is normal as they are all about the experience in the company
# Also, Monthly Income is highly positively correlated to Total Working Years and Age 
# Percent Salary Hike is positively correlated to Performance Rating
# And Total Working years is positively related to age and job level without any surprise 

## Total Working years vs Monthly Income
ggplot(data, aes(x= TotalWorkingYears, y=MonthlyIncome, color = 'pink')) +
  geom_point(alpha = 0.3) +
  labs(x = "Total Working Years",y = "Monthly Income") +
  geom_smooth(method=loess , color="red", se=FALSE) +
  theme_bw()
# There is an increasing trend but the variance is large at the same time
# And there are plenty of outliers of those who get higher monthly income with less working experience and
# those who receive lower monthly salary with more working experience

## Job Level vs Monthly Income by Attrition
ggplot(data, aes(x=factor(JobLevel), y=MonthlyIncome, fill=Attrition)) + 
  geom_boxplot() +
  labs(x = "Job Level",y = "Monthly Income") +
  ggtitle("Relationship between Monthly Income & Job Level by Attrition") +
  theme_bw()
# The higher the job level, the higher the monthly salary
# The distribution of income varies for each level, e.g. the variation is smallest for job level 5
# We can see that the variation of monthly income for each group are similar among employees stayed and who quit
# except for job level 4 where employees quit get lower pay than those who stayed,
# i.e. the upper quartile of those who quit is less than the lower quartile of those who stayed

## Performance Rating and Percentage of Salary Hike by Attrition
ggplot(data, aes(x=factor(PerformanceRating), y=PercentSalaryHike, fill=Attrition)) + 
  geom_boxplot() +
  labs(x = "Performance Rating",y = "Percentage of Salary Hike") +
  ggtitle("Relationship between Performance Rating & Percentage of Salary Hike by Attrition") +
  theme_bw()
# The range of percentage of salary increase are the same for both employees who quit and stayed
# Employees with performance rating 4 get an increase in salary for at least 20% while those with performance rating 3
# only get an increase in salary for less than 20%
# This seems to be the company remuneration policy that the range of salary increase are set for each performance rating


#------------------------------#
####   Feature Engineering  ####
#------------------------------#

### Convert the Categorical variables into factors
## Employees' background
data$Gender <- as.factor(data$Gender)
data$MaritalStatus <- as.factor(data$MaritalStatus)
data$Education <- as.factor(data$Education)
data$EducationField <-as.factor(data$EducationField) 

## Employees' situations in the company
data$Attrition <- factor(data$Attrition,levels = c('Yes','No'), labels = c('Yes','No'))
data$BusinessTravel <- as.factor(data$BusinessTravel)
data$Department <- as.factor(data$Department)
data$OverTime <- as.factor(data$OverTime)
data$PerformanceRating <- as.factor(data$PerformanceRating)
data$StockOptionLevel <- as.factor(data$StockOptionLevel)
data$JobLevel <- as.factor(data$JobLevel)
data$JobRole <- as.factor(data$JobRole)

## Questionnaire responses
data$EnvironmentSatisfaction <- as.factor(data$EnvironmentSatisfaction)
data$JobInvolvement <- as.factor(data$JobInvolvement)
data$RelationshipSatisfaction <- as.factor(data$RelationshipSatisfaction)
data$WorkLifeBalance <- as.factor(data$WorkLifeBalance)
data$JobSatisfaction <- as.factor(data$JobSatisfaction)


#---------------------------#
#####  Model Building   #####
#---------------------------#

## Split data into Training and Testing set ##
set.seed(100)

## Split up the dataset, basically randomly assigns a boolean to a new column "sample"
sample <- sample.split(data$Attrition, SplitRatio = 0.7)

train <- subset(data, sample == TRUE)
test <- subset(data, sample == FALSE)

## Examine if the training and testing set follow a similar distribution of attrition
prop.table(table(data$Attrition))
# 84% of the employees stay in the company and 16% resign in the original dataset
prop.table(table(train$Attrition))
# The training set keep the same proportion of attrition (84% : 16%)
prop.table(table(test$Attrition))
# The testing set keep the same proportion of attrition (84% : 16%)


#---------------------------------#
#### Logistic Regression Model #### 
#---------------------------------#

### Logistic regression model with all the features
logit.model_1<- glm(Attrition ~ ., family = binomial(link = 'logit'), data = train) 
summary(logit.model_1)
vif(logit.model_1)
# P-values and VIF(Variable Inflation Factors) of Department are the biggest, Department is not statistically significant predictor 
# and it has high multicollinearity this independent variable and the others
# We will remove it to see if we have a better model

## Model Evaluation - Predict testing set
predict_logit_1 <- predict(logit.model_1,test, type = 'response')
predict_logit_1 <- as.factor(ifelse(predict_logit_1 > 0.5, "No", "Yes"))

## Create a confusion matrix
confusionMatrix.lm_1 <-confusionMatrix(data = relevel(predict_logit_1, ref = "Yes"), 
                                       reference = relevel(test$Attrition, ref = "Yes"))
confusionMatrix.lm_1
# Accuracy = 85.94%
# Recall (Sensitivity) = 46.48% which means only 46.48% of employees quitted are correctly identified from
# all of the quitted employees
# We can see that from the confusion matrix, only 33 employees are correctly predicted as quit
# and 38 employees quitted are not detected


### Logistic regression model without Department
logit.model_2 <- glm(Attrition ~ . - Department, 
                     family = binomial(link = 'logit'), data = train) 
summary(logit.model_2)
# Education Fields seems to be a statistically insignificant predictor
# We will remove this predictor in the next model

## Model Evaluation - Predict testing set
predict_logit_2 <- predict(logit.model_2,test, type = 'response')
predict_logit_2 <- as.factor(ifelse(predict_logit_2 > 0.5, "No", "Yes"))

## Create a confusion matrix
confusionMatrix.lm_2 <-confusionMatrix(data = relevel(predict_logit_2, ref = "Yes"), 
                                       reference = relevel(test$Attrition, ref = "Yes"))
confusionMatrix.lm_2
# Accuracy = 86.17%
# Recall (Sensitivity) = 47.89% 
# This model does a better job than the previous one in terms of both accuracy and recall


### Logistic regression model without Department and Education Field
logit.model_3 <- glm(Attrition ~ . - Department -EducationField, 
                     family = binomial(link = 'logit'), data = train) 
summary(logit.model_3)

## Model Evaluation - Predict testing set
predict_logit_3 <- predict(logit.model_3,test, type = 'response')
predict_logit_3 <- as.factor(ifelse(predict_logit_3 > 0.5, "No", "Yes"))

## Create a confusion matrix
confusionMatrix.lm_3 <-confusionMatrix(data = relevel(predict_logit_3, ref = "Yes"), 
                                       reference = relevel(test$Attrition, ref = "Yes"))
confusionMatrix.lm_3
# Accuracy = 85.71%
# Recall (Sensitivity) = 46.48% 
# This model does a worse job than the previous one in terms of both accuracy and recall
# We will use the second model instead


#-------------------------#
#### Naive Bayes model ####
#-------------------------#

### Gaussian Naive Bayes model 
model_nb <-naiveBayes(Attrition ~ ., data = train)

## Model Evaluation - Predict testing set
model_nb_predict <- predict(model_nb, newdata=test)

## Confusion matrix 
confusionMatrix.nb <- confusionMatrix(model_nb_predict, test$Attrition)
confusionMatrix.nb
# Accuracy = 78.68%
# Recall(Sensitivity) = 67.61% which is relatively good to other models so far


### Naive Bayes model with Laplace Smoothing
# It is a technique for smoothing categorical data by adding a small-sample correction 
# in every probability estimate, so that no probability will be zero
model_nb_laplace <- naiveBayes(Attrition ~ . , data = train, laplace = 2 )

## Model Evaluation - Predict testing set
model_nb_laplace_predict <- predict(model_nb_laplace , test , type = "class" )

## Confusion matrix 
confusionMatrix.nbl <- confusionMatrix(model_nb_laplace_predict, test$Attrition)
confusionMatrix.nbl
# Accuracy = 79.37% which is slightly better than the previous one
# Recall(Specificity) = 67.61% , no change from the previous model


### Naive Bayes model with Kernel Density Estimation
# Kernel based estimation is a technique for estimation of probability density function of the continuous predictors
# It can improve the Naive Bayes model's prediction accuracy
model_nb_kernel <- naive_bayes(Attrition ~ . , usekernel = T, data = train)

## Model Evaluation - Predict testing set
model_nb_kernel_predict <- predict(model_nb_kernel, newdata = select(test,-Attrition), type = 'class')

## Confusion matrix 
confusionMatrix.nbKDE <- confusionMatrix(model_nb_kernel_predict,test$Attrition)
confusionMatrix.nbKDE
# Accuracy = 81.86% 
# Recall(Sensitivity) = 52.11%  
# This model is the best among three Naive Bayes models in terms of accuracy,
# however we focus on recall for this business problem. Using Kernel based estimation
# for the continuous predictors does not improve the model in this case
# We will choose the Gaussian Naive Bayes model as it does a fairly good job on recall
# among these three Naive Bayes models

# Gaussian Naive Bayes model is doing a good job as it is not sensitive to irrelevant 
# and correlated features, also it is highly scalable with the number of features
# However, it does not perform well if independence assumption is not met

#-------------------------------#
###### K- Nearest Neighbor ######
#-------------------------------#

### Data Preparation for KNN Model ###
## Normalize the data for building KNN model
normalize = function(vec){
  if (is.numeric(vec)) {
    vec = (vec - min(vec)) / (max(vec) - min(vec))
  }
  return (vec)
}

## Normalize all the numeric variables ( scale them between 0 and 1)
data.scaled = mutate_if(data, is.numeric, normalize)

## Expand the factors to a set of dummy variables 
data.normalized = data.frame(model.matrix(~ .-Attrition, data = data.scaled), data.scaled$Attrition)[,-1]
data.normalized <- rename(data.normalized, Attrition = data.scaled.Attrition)

## Train- Test split for the normalized dataset
set.seed(100)
sample <- sample.split(data.normalized$Attrition, SplitRatio = 0.7)

train.normalized <- subset(data.normalized, sample == TRUE)
test.normalized <- subset(data.normalized, sample == FALSE)

## Examine if the training and testing set follow a similar distribution of attrition
prop.table(table(data.normalized$Attrition))
# 84% of the employees stay in the company and 16% resign in the original dataset
prop.table(table(train.normalized$Attrition))
# The training set keep the same proportion of attrition (84% : 16%)
prop.table(table(test.normalized$Attrition))
# # The testing set keep the same proportion of attrition (84% : 16%)

## Try the KNN starting by K = 1
predict_knn1 <- knn(train.normalized[,-61],test.normalized[,-61], train.normalized[,61], k =1)

## Confusion matrix 
confusionMatrix.knn1 <- confusionMatrix(predict_knn1,test.normalized$Attrition)
confusionMatrix.knn1
# Accuracy = 75.51% , lower than the 'No Information Rate' of 83.9% which is 
# the best guess given no information beyond the overall distribution of the classes.
# We can treat 'No information rate' as a baseline of the model performance.
# 83.9% no information rate means it will be 83.9% correct when predicting the majority class (i.e. 'Attrition - No')
# This KNN model does a worse job than the baseline in this case.
# Recall(Sensitivity) = 18.31% which is so far the worst among other models.
# This model cannot predict the 'Yes' category well as only 13 of them are correctly predicted

## Elbow Method for determining K value ##
predicted.purchase = NULL
error.rate = NULL

for(i in 1:30){
  set.seed(100)
  predicted.att = knn(train.normalized[,-61],test.normalized[,-61], train.normalized[,61],k=i)
  error.rate[i] = mean(test.normalized$Attrition != predicted.att)
}
k_values <- 1:30
error.df <- data.frame(k_values,error.rate)

## Visualize the error rate for finding the 'Elbow'
ggplot(error.df, aes(k_values,error.rate)) +
  geom_point() +
  geom_line(lty = 'dotted', color = 'red') +
  theme_bw()
# An "elbow" is a cut-off point where 
# the error rate would not decrease for using a higher K value
# There is no obvious elbow but We can see that 
# increasing beyond K=12 does not help our misclassification at all.
# So we can set that as the K=12 for our model during training.

## KNN model with K = 12
predict_knn2 <- knn(train.normalized[,-61],test.normalized[,-61], train.normalized[,61], k = 12)

## Confusion Matrix
confusionMatrix.knn2 <- confusionMatrix(predict_knn2, test.normalized$Attrition)
confusionMatrix.knn2
# Accuracy = 85% 
# Recall(Sensitivity) = 9.86% which is worse than K=1
# The elbow method finds the best KNN model with the highest accuracy but not the sensitivity.
# In this case, using the Elbow method does not help for finding a better KNN model.
# We can see that this KNN model does not perform well as it wrongly predicts 'yes' as 'no' for most of the cases.
# Because KNN is not good with high dimensional data and does not work well with categorical features.
# Also it does not work well on rare event target variable since our dataset is imbalanced ( 16% quit & 84% stay)


#-----------------------#
###   Decision Tree   ###
#-----------------------#

## Create a decision tree
set.seed(100)
decision_tree <- rpart(Attrition ~ ., method="class", data=train, 
                       control=rpart.control(minsplit=1), parms=list(split="information"))

## Complexity parameter table
printcp(decision_tree)
# There are 22 features are considered in this decision tree model

## Visualize the complexity parameter table
plotcp(decision_tree)
# Y-axis is the Cross validation error, lower x-axis is the cost complexity ( α) value,
# upper x-axis is the number of terminal nodes (i.e., tree size = |T|).
# The cross validation error reaches the lowest with tree size 4 to 6 where
# T = 4 or T =5 are smallest tree within 1 standard error (SE) of the minimum CV error
# we can prune the decision trees with complexity parameter(cp) = 0.027108 
# when T = 3 (nsplit from the cp table )

## Visualize the tree
rpart.plot(decision_tree, type=2, extra=1)
# There are 37 leaves (terminal nodes) 
# The model is too complex to see each nodes and this model may be overfitting
# So we will examine the importance of the features and the tree itself after pruning

## Model Evaluation - Predict testing set
predict_decision_tree <- predict(decision_tree,newdata=test,type="class")

## Create a confusion matrix
confusionMatrix.dt <- confusionMatrix(predict_decision_tree, test$Attrition)
confusionMatrix.dt
# Accuracy = 82.54%
# Recall( Sensitivity) = 40.85% which is relatively high comparing with other models
# However, this tree is too complex and not generalized. If other new test data are used,
# this model may not predict well as it overfits the training set

### Pruning the decision tree ###
## Prune the tree with smallest tree size
# Cp = 0.027108 from the cp table of the previous tree
decision_tree_pruned <- prune(decision_tree, cp = 0.027108)

## Complexity parameter table of the pruned tree
printcp(decision_tree_pruned)
# Only 5 features are used in this pruned tree

## Visualize the pruned decision tree
rpart.plot(decision_tree_pruned)
# We can clearly see the features used now
# The root node is 'Overtime' and the nodes are 'Monthly income', 'Age', 
# 'Education' and 'Stock option level'.
# If the employees do not work overtime, they will fall into the class 'No' even if the employees actually quitted.
# This may ignore other possible features and lead to a lower sensitivity as 
# the majority of class is 'No' (i.e. imbalanced dataset) 

## Model Evaluation - Predict testing set
predicted_att.dt <- predict(decision_tree_pruned, test, type = 'class')

## Create a confusion matrix
confusionMatrix.dtp <- confusionMatrix(predicted_att.dt, test$Attrition)
confusionMatrix.dtp
# Accuracy = 84.35 % which is slightly better than the previous tree
# But in this business case , we focus on recall (sensitivity)
# Recall = 23.94% which is much lower than the original tree
# Decision Tree may not be a good model in this case but we can still identity
# the important features from the tree

### Visualize the feature importance
## Create a data frame for the feature importance
df <- data.frame(imp = decision_tree_pruned$variable.importance)
df2 <- df %>% 
  tibble::rownames_to_column() %>% 
  rename("variable" = rowname) %>% 
  arrange(imp) %>%
  mutate(variable = forcats::fct_inorder(variable))

## Variable importance plot
ggplot(df2) +
  geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
               size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 4, show.legend = F) +
  ylab('importance') +
  ggtitle('Feature Importance') +
  coord_flip() +
  theme_bw()
# This variable importance plot shows the importance of the features which may not
# be used in a split
# Overtime is still the most important feature, followed by Monthly income
# Job level, Job role and Total working year are important but they are not used in the split
# By contrast, education and stock option level are less important but they are used in the split

#-------------------#
### Random Forest ###
#-------------------#

## Create a random forest with default no. of tree = 500
set.seed(100)
rf.model <- randomForest(Attrition ~ . , data = train, proximity = TRUE, importance = TRUE)

## Model Evaluation - Predict testing set
rf.predict <- predict(rf.model , test)

## Confusion Matrix
confusionMatrix.rf <- confusionMatrix(rf.predict, test$Attrition)
confusionMatrix.rf
# Accuracy = 86.85%
# Recall (Sensitivity) = 25.35% which means only 25.35% of employees quitted are correctly identified from
# all of the quitted employees.
# We can see that from the confusion matrix, only 18 employees are correctly predicted as quit
# and 53 employees quitted are not detected.

## Variable importance plot
vip(rf.model, num_features = 25, bar = TRUE)
# The variable importance is based on the average total reduction of 
# the loss function for a given variable across all trees.
# The influential variables are 'Overtime', 'Total working years', 'Age', 'Monthly income',
# 'Job role', Stock option level' and 'Job level' which are identified in the previous decision trees
# But 'Education' is not important in the Random forest model

#------------------------------------#
#### Model Evaluation & Selection ####
#------------------------------------#

## Methodology
# Recall (Sensitivity) : we want to know how well the model detect the employee quitted(TP) from who
# actually quitted (TP + FN)
# It is the proportion of positive result that were correctly classified from all the actual positive
# It is more useful when the class is imbalanced, as true negative is not included in the calculation
# and thus recall is not affected the imbalance
# In this dataset, we have lots of 'No Attrition' relative to 'Yes' and we would like to 
# predict 'Yes' (i.e leave the company) precisely.
# With recall , we can evaluate the models by looking at how well they predict an
# actual 'Yes' from all the actual positive (True positive and false negative)

modelComparison <- data.frame(
  Model = c('Logistic Regression', 'Gaussian Naive Bayes', 'KNN',
            'Decision Tree', 'Pruned Decision Tree', 'Random Forest'),
  Sensitivity = c(confusionMatrix.lm_2$byClass['Sensitivity'],
                  confusionMatrix.nb$byClass['Sensitivity'],
                  confusionMatrix.knn1$byClass['Sensitivity'],
                  confusionMatrix.dt$byClass['Sensitivity'],
                  confusionMatrix.dtp$byClass['Sensitivity'],
                  confusionMatrix.rf$byClass['Sensitivity'])
)
modelComparison

# All the models do not do a good job due to imbalanced class and
# the models cannot detect the true positive well.

### Selected Model : Gaussian Naive Bayes ###
# Only Gaussian Naive Bayes model performs better with recall rate of 67.6%,
# but still it just does a fair job on identifying the employees quitted from all the employees actually quitted.
# With this model, we can only be able identify 67.6% of quitted employees, the human resources department and 
# department managers may need other measures to identify them, e.g. job performance changes or emotion.

# We can manipulate the distribution of the dataset by using different sampling techniques
# to improve the model performance.
# But due to time concern, this would be a further study in the coming future.



#--------------------------#
####    Key Findings    ####
#--------------------------#

#### Employees’ background ####

### Age 
# The majority of employees who quit (at around 30) are younger than 
# the majority of employees who stay (at around 35).
# Also the proportion of employees in their 20s who quit are higher than those staying.
# We can conclude that the employees quitted are pretty young. This could be they are still searching for their career goals or trying to adopt the working environment.

### Total Working Years
# Employees with 1 total working year have around half chance to leave.
# This could be due to employees’ own career planning which is natural attrition.
# Another possible reason is the career path in our company is not clear or attractive enough to the new employees. 

# Human resources department may give the new employees an overview of their development opportunities in the company and understand their career interests in order to assist them in their own career development.


#### Employees' benefits ####

### Salary 
# The salary scale of each job level varies, especially for job level 4 where 
# the employees who quit get lower pay than the lower quartile of that of employees stayed
# In general, there is a higher density of low paid (<$5000) employees quitted than those who stayed.
# Human Resources may do a questionnaire for employees who are leaving to see if monthly income is one of 
# their big concerns of working in the company.

### Salary Hike
# It seems to be the company policy that a range of percentage of salary increase is set with 
# performance rating

### Stock Option Levels
# Overall, most of the employees get either level 0 or 1 of the company stocks : 
# Employees who quit have either level 0 or 1 of stock options while employees with level 2 or 3 are less likely to leave the company
# Stock options has becoming one of the most important feature of the remuneration package over the past decades, 
# employees can be more motivated with stock options as the harder they work and the bigger the company grows, 
# the higher returns and satisfaction they get.
# It would be a good idea to distribute the stock options to those at level 0 (junior employees)
# so that the morale would be higher and the employees would be encouraged to stay and grow with the company.
# At the same time, it is cost effective to the company as the company is not giving money to the employees directly.


#### Employees' situations in the company ####

### Overtime
# Overtime is the top influential factor of attrition. 
# 71% of total employees do not work overtime showing that this company treats its employees with care.
# Among those who need to work overtime, half of them quitted.
# Whereas only 23% quitted among those who get off from work on time.
# The manager may consider adjusting the working time or workload for the employees and encouraging
# work life balance.

### Job Roles
# Half of the employees leaving are either Laboratory Technician or Sales Executive,
# managers of the particular departments may investigate the problems behind:
# would it be the remuneration or working environment?
# The managers can try to understand their subordinates' concerns and try to make a retention plan

### Overtime & Department & JobLevel & JobRole & BusinessTravel
# Half of those who worked overtime left their jobs and 65% of employees working overtime are from R&D department.
# And a large proportion of these people work as research scientists and sales executives
#(most of them have the low Job Level 1 or 2) who have the business travel frequently.
# So it’s possible that employees are more likely to leave the company if they have a high-intensity but low-level job.

### Years at the Company
# Most of the attrition at the company happens in the first 10 years of people working there at 91.51% of
# the people who quit do it before their 11th year. 
# The year most people leave is between their first and second year
# on the job with 24% of the people quitting do it then.
# After the 10th year of working most people do not quit, this is probably because of tenure or seniority.
# It is interesting to see that there is currently a higher than average people working at the company for 5 years,
# this could be because of an expansion in the company because it started doing better.

### Years in Current Role
# Most people leave their job either within their first year, 30.8% or in their second year, 28.69%
# There is also another spike of people leaving in their 7th year at their current position, at 13.1%
# This is most likely that people start the job and then realize they do not like it and leave, or after two years
# they are looking for something new. It would be interesting if the people who leave at 7 years would fill out a 
# servery to explain why they are leaving to see if there is any correlation between them all and as to why.

### Years since last promotion
# It seems that a lot of people quit less than one year after getting a promotion at work.
# When this was looked into it was shown that most people who did so, did it after working 
# for the company for 1 year. This could mean that either there is an automatic promotion 
# after working for the company for 1 year, or that that position(s) is very hard and stressful
# and should be looked into.  

### Years with manager
# This investigation could have led to some more interesting findings as there was people
# who had the same manager for less than a year and quit. Since the data set
# did not include people’s names and who was their manager, human resources department may need to 
# further investigate if it was a specific manager people had that quit.


#### Employees' Satisfaction ####
# In general, employees have high satisfaction rating in different aspect , e.g. job satisfaction,
# relationship satisfaction, environment satisfaction

### Environment Satisfaction
# Environment Satisfaction could be an influential factor of attrition as the lower the
# environment satisfaction rating, the higher the attrition rate.
# Employees with low environment satisfaction ('1') have the highest attrition rate at 25.35%
# whereas employees with other satisfaction levels get 14% attrition rate.

# Human resources can deploy a caring scheme including extra-curricular activities for improving workplace relationship and morale,
# as well as an anonymous feedback system for understanding the difficulties the employees are facing. 
# So that the employees can feel free to express their opinion of the working environment and the company can improve it in return.


#------------------------#
#####    Summary     #####
#------------------------#

### Influential Factors of Attrition
## Our study finds out the employees’ major concerns as following : 
# 1) Employees’ background 
#    a) Age 
#    b) Working experience
# 2) Employees’ working conditions
#    a) Working overtime
#    b) Job role
#    c) Job levels
# 3) Employees’ benefits
#    a) Salary
#    b) Stock options


### Recommendations 
## We suggest the following measures to improve the employees morale and the attrition rate :
# 1) Career counseling and coaching for new employees
# 2) Stock Options for junior employees
# 3) Working time or workload adjustment
# 4) Employees' caring scheme
# 5) Company promotion system
# 6) Working environment improvement


