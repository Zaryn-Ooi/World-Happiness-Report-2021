 
# What makes people in a country happy?

## Task 
Create a model that accurately predicts the happiness of countries around the world.


## Data Source 
[World Happiness Report 2021] (https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021)


## Data Context
The World Happiness Report is a landmark survey of the state of global happiness. The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.


## Data Content
The happiness scores and rankings use data from the Gallup World Poll . The columns following the happiness score estimate the extent to which each of six factors – Economic Production, Social Support, Life Expectancy, Freedom, Absence of Corruption, and Generosity. 

Although there are 20 variables in this dataset, my interest will be on 7 key variables.  

 - Country Name: The names of the countries. 

 - Ladder Score: Happiness score or subjective well-being. This is the national average response to the question of life evaluations.
 
 - Logged GDP per capita: The GDP-per-capita time series from 2019 to 2020 using countryspecific forecasts of real GDP growth in 2020.
 
 - Social Support: Social support refers to assistance or support provided by members of social networks to an individual.
 
 - Healthy Life Expectancy: Healthy life expectancy is the average life in good health.
 
 - Freedom to Make Life Choices: Freedom to make life choices is the national average of binary responses to the GWP question “Are you satisfied or dissatisfied with your freedom to choose what you do with your life?” 
 
 - Generosity: Generosity is the residual of regressing national average of response to the GWP question “Have you donated money to a charity in the past month?” on GDP per capita.?

 - Perceptions of Corruption: The measure is the national average of the survey responses to two questions in the GWP: “Is corruption widespread throughout the government or not” and “Is corruption widespread within businesses or not?”
 

## Install and Load Packages
```{r}
install.packages("pacman")
```

```{r}
pacman::p_load(
  caret,
  corrplot,
  GGally,       
  magrittr,     
  pacman,        
  parallel,      
  randomForest,  
  rattle,        
  rio,           
  tictoc,        
  tidyverse,
  rpart,
  rpart.plot
)
```

```{r}
library(readr)
```

```{r}
install.packages("olsrr")
```

```{r}
library(olsrr)
```


## Import Data
```{r}
data <- read_csv("world-happiness-report-2021.csv")
```


```{r}
# Explore the imported data
head(data)
str(data)
```


## Data Cleaning 

```{r}
# Remove unnecessary columns 
data <- data[ -c(2,4:6,13:20) ]
```


```{r}
# Rename column names
colnames(data)<-
  c("country","ladder","GDP","social","healthy","freedom","generosity","corruption")
head(data)
```


```{r}
# Check NA values
is.null(data)
```

```{r}
data %>% na.omit(data)
```


## Summary statistics of the data
```{r}
data %>% summary()
```
The max ladder score is 7.842, mean is 5.533, median is 5.534, and the min is 2.523. For all our models, the variable "ladder" will be the dependant variable and expressed as the Happiness score for the entire project.  



## Data Visualization

### Happiest Countries
I will create a barchart to visualize the top 5 and bottom 5 rank by cuntries. 

```{r}
# top 10
top10 <- data %>% 
  select(country, ladder) %>% 
  arrange(desc(ladder)) %>%  
  slice(1:10)
print(top10)
```


```{r}
# Barchart to visualize the top 10 happiest countries
top10 %>% 
  ggplot(aes(fct_reorder(country, ladder), ladder)) +
  geom_bar(stat = "identity", fill = "#0099FF", width = 0.7, alpha = 0.7) + labs( x= "Countries",
        y= "Happiness Score",
        title =  "Top 10 Happiest Countries")  +
  theme(axis.title.y = element_blank()) +
  theme_classic() +
  coord_flip()
```


### Least Happiest Countries
```{r}
# bottom 10
bottom10 <- data %>% 
  select(country, ladder) %>% 
  arrange(ladder) %>%  
  slice(1:10)
print(bottom10)
```

```{r}
# Barchart to visualize the bottom 5 happiest country
bottom10 %>% 
  ggplot(aes(fct_reorder(country, -ladder), ladder)) +
  geom_bar(stat = "identity", fill = "#CC0033", width = 0.7, alpha = 0.7) + labs( x= "Countries",
        y= "Happiness Score",
        title =  "Top 10 Least Happiest Countries")  +
  theme(axis.title.y = element_blank()) +
  theme_classic() +
  coord_flip()
```
The happiest country was Finland with a ladder score of 7.84 and the least happy country was Afghanistan with a score of 2.52.



## Show Correlation

```{r}
# Remove "country" variable 
df <- data %>%
  select(-1)
print(df)
```


### Correlation Matrix
```{r}
# Visualize Correlation Matrix 
df %>%
  cor() %>%
  corrplot(
    type   = "upper",     
    diag   = F,           
    order  = "original",  
    tl.col = "black",     
    tl.srt = 45           
  )
```


### Scatterplot Matrix
```{r}
# Scatterplot Matrix
df %>%
  select(      
    ladder,        
    GDP:corruption  
    ) %>%
  ggpairs()
```

Based on the Correlation Matrix and Scatterplot Matrix, it can be seen that most variables are positively correlated, except for "Genorisity" and "Perceptions of Corruption". 



## Model Building 

### Set Seed
```{r}
set.seed(123)
```


### Split data into train data and test data
```{r}
train <- df %>% sample_frac(.70)  
test  <- anti_join(df, train)
```



## Linear Regression

For our first model, I will used our training data to perform a linear regression model with ladder score as the outcome variabe and the rest variables as the predictor variables.

```{r}
# Compute multiple regression with train data 
lm <- lm(ladder ~ ., data=train)
```

```{r}
# Summarize Regression Model
summary(lm)
```

The linear model returned an R-squared value of 73.67% and an adjusted R-squared value of 72.04%, meaning that 73.67% of the variance of ladder can be explained by the predictor variables. This model predicted that "GDP","social support","healthy life expectancy",and "freedom to make life choices" were the most impactful attributes as they have shown to be statistically significant. The p-value is less than .05 which is statistically significant. Therefore, the predictor variables gives us a reliable estimate in determining the ladder score (happiness). 



### Diagnostic Plot

Although this linear regression model appears to be fairly accurate, it needs to be verified that the data it is being applied to is normally distributed. If the data are not normally distributed, the model results cannot be accurately applied when using a linear regression model. 

```{r}
# Diagnostic Plot
lm %>% plot()
```

#### Diagnostic Plot 2 - Normal Q-Q 

I will focus on the Normal Q-Q plot. This plot is used to examine whether the residuals are normally distributed. It’s good if residuals points follow the straight dashed line. In our example, the points fall roughly along the straight diagonal line. The observations #44 and #6 and #18 deviate a bit from the line at the tail ends, but not enough to declare that the residuals are non-normally distributed. Therefore, Normal Q-Q plot confirms that the data is normally distributed.


```{r}
# Apply Linear Regression on Test Data
lm_p <- predict(   
lm, newdata = test       
)   
```


```{r}
# Get predicted happiness values of countries
lm_p 
```


### Compute accuracy of Linear Regression Model 

I will calculate the Root Mean Squared Error(RMSE), which measures the model prediction error to assess the performance of the models. It corresponds to the average difference between the observed known values of the outcome and the predicted value by the model. The lower the RMSE, the better the model.

```{r}
# Compute RMSE for linear regression model
linear.rmse <- RMSE(lm_p, test$ladder)
```

```{r}
linear.rmse
```
The Root Mean-Squared Error (RMSE) for this linear regression model was 44.95%.


```{r}
# Create a table to save our results for each model
accuracy_results <- tibble(method = "Linear_Regression", RMSE = linear.rmse)
```

```{r}
# View the accuracy table
accuracy_results %>% knitr::kable()
```



## Selecting predictors for multiple regression

### All Possible Regression
```{r}
# Criteria for all possible model combination
all.mod <- ols_step_all_possible(lm)
all.mod
```


```{r}
# Visualize the model
plot(all.mod)
```

### Best Subset Regression

Best Subset Regression select the subset of predictors that do the best at meeting some well-defined objective criterion, such as having the largest R2 value or the smallest MSE, Mallow’s Cp or AIC.

```{r}
# Criteria of the best subset of the model
best.mod <- ols_step_best_subset(lm)
best.mod
```

```{r}
# Visualize the model
plot(best.mod)
```
In this case, model with 3 predictors is the best fitting model due to its largest R2 value and the smallest MSE, Mallow’s Cp and AIC value. 


### Stepwise Procedures

#### Stepwise Forward Regression

Forward Regression starts with no predictors in the model, iteratively adds the most contributive predictors, and stops when the improvement is no longer statistically significant.

```{r} 
# Forward selection based on p-values
ols_step_forward_p(lm, details = FALSE)
```
In this case, the most contributive predictors are GDP, Corruption, Social, Freedom, and Healthy. 


#### Stepwise Backward Regression

Backward selection starts with all predictors in the model, iteratively removes the least contributive predictors, and stops when all of the  predictors are statistically significant.

```{r}
# Backward selection based on p-values
ols_step_backward_p(lm, details = FALSE)
```

In this case, Stepwise backward regression has eliminated the least significant variables predictor, "Generosity" . 


#### Stepwise Regression

Stepwise Regression is a combination of forward and backward selections. 

```{r}
# Variable Selection
step <- ols_step_both_p(lm, details = FALSE)
```

Based on the stepwise regression result, the three predictor model with the three predictors : "GDP", "social", and "corruption" are the best fitting model. 



### Gradient Boosting Model(GBM)

For the second model, I will try a Gradient Boosting Model. This approaches creates an ensemble where new models are added sequentially rather than simply averaging the predicted values of the models. 

```{r}
# Load necessary packages 
library(gbm)
library(MASS)
```


```{r}
Boston.boost <- 
  gbm(ladder ~ . ,
      data = train, 
      distribution = "gaussian",
      n.trees = 10000,
      shrinkage = 0.01, 
      interaction.depth = 4)
Boston.boost
```

```{r}
#Summary of Variable Importance and a plot of Variable Importance
summary(Boston.boost) 
```

The summary of the Model gives a feature importance plot. In the above list is on the top is the most important variable and at last is the least important variable. In this case, "Healthy", "GDP", and "Social" appear to be the top 3 important predictors.    


```{r}
# Predict on Test Set
Boston.boost_p <- predict(Boston.boost, test)
```


#### Compute Accuracy

```{r}
# Compute RMSE for gbm model
gbm.rmse <- RMSE(Boston.boost_p, test$ladder)
```

```{r}
gbm.rmse
```
The Root Mean-Squared Error (RMSE) for this GBM model was 63.83%. 


```{r}
# Save the gbm model RMSE result into our accuracy table
accuracy_results <- bind_rows(accuracy_results, tibble(method = "GBM", RMSE = gbm.rmse))
```

```{r}
# View the accuracy table
accuracy_results %>% knitr::kable()
```



## Decision Tree for Regression

For the third model, I will train a regression decision tree to predict the happiness of country. 

```{r}
library(rpart)
```

```{r}
decision_tree <- rpart(ladder~., 
             method = "anova", 
             data = train)
```

```{r}
decision_tree
```


```{r}
rpart.plot(decision_tree,
          main = "Predict Happiness using Decision Tree")
```

This decision tree is a way of predicting adder score. It has provided 5 understandable decision criteria. The first decision is whether GDP of the country is above or below 9.8. If <9.8, that's yes, then we look at whether the country have a score on healthy that has less than 62. If <62, we look at whether the country's GDP is less than 8.1. If <8.1, then the ladder score of the country is 4.3.  

If the first decision in which whether the country's GDP is greater than 9.8, that's no, then we will look at whether the country's GDP is less than or greater than 11. If below 11, then the country will have a ladder score of 6. On the other hand, if the contry's GDP is greater than 11, then the ladder score will be 7. These percentages at the bottom tell us what percentage of total cases fall into that particular cell. 



```{r}
# Predict Test Data
ladder_p <- decision_tree %>% 
predict(newdata = test)
```


#### Compute Accuracy
```{r}
# Compute RMSE for decision tree
decision.rmse <- RMSE(ladder_p, test$ladder)
```

```{r}
decision.rmse
```
The Root Mean-Squared Error (RMSE) for this decision tree model was 69.97%.

6
```{r}
# Save the decision tree RMSE result into our accuracy table
accuracy_results <- bind_rows(accuracy_results, tibble(method = "Decision Tree", RMSE = decision.rmse))
```

```{r}
# View the accuracy table
accuracy_results %>% knitr::kable()
```



### Random Forest

For the last model, I will train a Random Forest model. Random Forest takes the average of multiple decision trees in order to improve predictions.  

```{r}
# Define Parameters
control <- trainControl(
method = "repeatedcv", 
number = 10,           
repeats = 3,           
search = "random",     
allowParallel = TRUE   
) 
```


```{r}
# Train random forest model 
randomforest <- train(          
ladder~.,              
data = train,    
method = "rf",         
trControl = control,   
tuneLength = 15,       
ntree = 800,
importance = TRUE
)
randomforest
```


```{r}
# Plot accuracy by number of predictors 
randomforest %>% plot()
```

This tells us that in the 800 randomly generated decision trees, how many predictors did they each have? Most decision trees only had 1 predictor, and very few had 2 or 3 predictors. Between 4 and 6 predictors, the number of decision trees slowly increases but slightly drop on having 6 predictors. 



```{r}
randomforest$finalModel
```



```{r}
plot(randomforest$finalModel)
```


```{r}
which.min(randomforest$finalModel$mse)
```
I need 785 trees to have the lowest error. I will now rerun the model and add an argument called “ntree” to indicating the number of trees I want to generate.

```{r}
randomforest2 <- train(          
ladder~.,              
data = train,    
method = "rf",         
trControl = control,   
tuneLength = 15,       
ntree = 785,
importance = TRUE
)
randomforest2
```

```{r}
randomforest2$finalModel
```

We can now see which of the predictors in our model are the most useful.

```{r}
varImpPlot(randomforest2$finalModel)
```

The higher the INCMSE and IncNodePurity, the more important the predictor variable. "GDP" is most important followed by "healthy" and then "social". 


```{r}
# Predict Test data
randomforest_p <-predict(randomforest2, newdata = test)
```


#### Compute Accuracy
```{r}
# Compute RMSE for random forest
randomforest.rmse <- RMSE(randomforest_p, test$ladder)
```

```{r}
randomforest.rmse
```
The Root Mean-Squared Error (RMSE) for this decision tree model was 61.67%.


```{r}
# Save the decision tree RMSE result into our accuracy table
accuracy_results <- bind_rows(accuracy_results, tibble(method = "Random Forest", RMSE = randomforest.rmse))
```

```{r}
# View the accuracy table
accuracy_results %>% knitr::kable()
```



## Conclusion

The aim of this project was to create a model that accurately predicts the happiness of countries around the world. I have analyzed 4 different models, linear regression, gbm, decision tree, and random forest. From the results, I can conclude that multiple linear regression model estimated best accuracy as it has the lowest RMSE value. The RMSE of each model is shown in the table below. 

```{r}
# Determining the Final Model 
accuracy_results %>% knitr::kable()
```


According to the linear regression model, the results conclude that GDP  is the single most influential predictor in determining a country’s happiness. Other significant attributes are a country’s social support and corruption scores. If a country with lower happiness score opts to use the results of this study to drive their upcoming initiative, they should focus on ways to stimulate the economies.


Below are the scatterplot showing how each of the top predictors relate to happiness. 

```{r}
# GDP vs Happiness
ggplot(data = df) +
  geom_point(mapping = aes(x= GDP , y=ladder, color= ladder)) +
  theme_classic() +
  labs( x= "GDP per capita",
        y= "ladder score",
    title = "The higher the GDP per capita, the happier the country")
```

```{r}
# Social Support vs Happiness
ggplot(data = df) +
  geom_point(mapping = aes(x= social , y=ladder, color= ladder)) +
  theme_classic() +
  labs( x= "social support",
        y= "ladder score",
    title = "The more social support, the happier the country")
```

```{r}
# Corruption score vs Happiness
ggplot(data = df) +
  geom_point(mapping = aes(x= corruption , y=ladder, color= ladder)) +
  theme_classic() +
  labs( x= "corruption score",
        y= "ladder score",
    title = "The lower the corruption score, the happier the country")
```








