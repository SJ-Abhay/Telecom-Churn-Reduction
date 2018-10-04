# Remove Lists
rm(list = ls())
#
# Set Working Directory
setwd("F:/Churn Reduction")
#
# Get Working Directory
getwd()
#
# Load Data
train_data <- read.csv("train_data.csv", header = T)
test_data <- read.csv("test_data.csv", header = T )
#
# Load libraries
library(tidyr)
library(Hmisc)
library(knitr)
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(gridExtra)
library(ROCR)
library(corrplot)
library(usdm)
library(ROSE)
library(rpart)
library(C50)
library(ROSE)
library(corrgram)
library(gmodels)
#
# Explore Data
str(train_data)
str(test_data)
#
# Data Combine to get a complete row data
customer_df = rbind(train_data,test_data)
str(customer_df)
#
# Missing Value Analysis
sum(is.na(train_data))
sum(is.na(test_data))
sapply(customer_df,function(x)sum(is.na(x)))
#
# Missing Value Visualization
missing_values <- customer_df %>% summarize_all(funs(sum(is.na(.))/n()))
missing_values <- gather(missing_values, key="feature", value="missing_pct")
missing_values %>% 
    ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +
    geom_bar(stat="identity",fill="red")+
    coord_flip()+theme_bw()
#
# Boxplots to check for outliers in the data
ggplot(stack(customer_df), aes(x = ind, y = values)) +
  geom_boxplot() + coord_flip() 
#
#Variable Transformations
customer_df$Churn <- as.integer(customer_df$Churn)
customer_df$voice.mail.plan <- as.integer(customer_df$voice.mail.plan)
customer_df$international.plan <- as.integer(customer_df$international.plan)
customer_df$area.code <- as.factor(customer_df$area.code)
#
# Give binary structure
customer_df$Churn[customer_df$Churn == '1'] <- 0
customer_df$Churn[customer_df$Churn == '2'] <- 1
#
customer_df$voice.mail.plan[customer_df$voice.mail.plan == '1'] <- 0
customer_df$voice.mail.plan[customer_df$voice.mail.plan == '2'] <- 1
#
customer_df$international.plan[customer_df$international.plan == '1'] <- 0
customer_df$international.plan[customer_df$international.plan == '2'] <- 1
#
#################### Data Visualization #############################
# Initialize
dev.off()

# Realtion between area code and churning customers
ggplot(customer_df, aes(x = customer_df$area.code, y = customer_df$account.length )) + geom_col(show.legend = TRUE )

# Correlation of other varibales with the Target Variable
ggplot(customer_df, aes(x=international.plan, y=Churn)) +
  geom_point(shape=1) +    
  geom_smooth(method=lm)

ggplot(customer_df, aes(x=total.day.minutes, y=Churn)) +
  geom_smooth(method=lm)

ggplot(customer_df, aes(x=total.day.charge, y=Churn)) +
  geom_smooth(method=lm)

ggplot(customer_df, aes(y=number.customer.service.calls, x=Churn)) +
  geom_smooth(method=lm)



#
####################### Variable Selection #############################################

# Drop no use Variable 
customer_df$area.code <- NULL
customer_df$state <- NULL
customer_df$phone.number <- NULL

# Draw correlation plot
corrgram(customer_df, order = F, lower.panel = panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

# VIF to find out the variables have collinearity problems
vifcor(customer_df[-18], th = 0.9)

# Drop Multicollenear variables
final_df = customer_df[, -c(7,10:11,16,3)]

############# Chi-squared Test of Independence #########################################

# Convert target variables into factor 
final_df$Churn <- as.factor(final_df$Churn)

# Chi-square test
chisq.test(final_df$international.plan,final_df$Churn)
chisq.test(final_df$total.intl.calls,final_df$Churn)
chisq.test(final_df$number.customer.service.calls,final_df$Churn)

########### Target Class Distribution #################################################
barplot(prop.table(table(customer_df$Churn)),
        col = rainbow(2),
        ylim = c(0,1),
        main = 'Target Class Distribution')

prop.table(table(customer_df$Churn))

#There is a perfect target class imbalance problem where 14% customer only churn

################### Model Development ##################################################
# Data Split into Train and test
set.seed(1234)
cdf.indx <- sample(2,nrow(final_df),replace = T, prob = c(0.7,0.3))
cdf_train <- final_df[cdf.indx == 1,]
cdf_test <- final_df[cdf.indx == 2,]


#Creating over,under,both and synthetic samples to overcome target imbalance
cdf_over = ovun.sample(Churn ~., data = final_df, method = ' over',N = 5984)$data
cdf_under = ovun.sample(Churn ~., data = final_df, method = 'under',N = 1004)$data
cdf_both = ovun.sample(Churn ~., data = final_df, method = 'both',
                       p = 0.5,
                       seed = 221,
                       N = 3494)$data

cdf_ROSE = ROSE(Churn ~., data = final_df,
                N = 5000,
                seed = 221)$data



######################### Decision Tree ################################################

dt_model = C5.0(Churn ~ ., data = cdf_train,trials = 100, rules = FALSE)
summary(dt_model)

fit <- rpart(Churn ~ .,method="class", data=cdf_train)
printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, 
     main="Classification Tree for Churn")
text(fit, use.n=TRUE, all=TRUE, cex=.8  )


#Predictions with the Training Data
DT_pred = predict(dt_model, cdf_test[,-18], type = "class")

#ROC Curve
DT_roc = predict(dt_model,cdf_test,type = 'prob')[,2]
DT_roc =  prediction(DT_roc,cdf_test$Churn)
eval = performance(DT_roc,'acc')
plot(eval)


#Evaluating Model Performance using Confusion Matrix
cnf = table(cdf_test$Churn,DT_pred)
confusionMatrix(cnf) 
CrossTable(cdf_test$Churn,DT_pred,prop.c = F,prop.chisq = F,
           prop.r = F, dnn = c('actual default','predicted default') )

# Accuracy 96%
# Precision =  1250/(1250+36) = 96%
# Recall = 1250/(1250 + 12) = 99%
######################### Random Forest ################################################
RF_model = randomForest(Churn ~ ., cdf_train, importance = TRUE, ntree = 500)
importance(RF_model)

#Variable Importance 
plot.new()

varImpPlot(RF_model,type = 1)
abline(v = 45, col= 'blue')
#This plot resembles the important parameters in RF prediction

#Predict test data using random forest model
RF_Predictions = predict(RF_model, cdf_test[,-13])

#ROC Curve
RF_roc = predict(RF_model,cdf_test,type = 'prob')[,2]
RF_roc =  prediction(RF_roc,cdf_test$Churn)
eval_ = performance(RF_roc,'acc')
plot(eval_)

##Evaluate the performance of classification model
ConfMatrix_RF = table(cdf_test$Churn, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

# Accuracy = 96%
# Precision = 1253/(1253+50) = 96%
# Recall =  1253 / (1253+9) = 99%
############################ Logistic Regression Model ###############################
scaled_train = cdf_train
scaled_test = cdf_test
scaled_train[,1:12] = scale(scaled_train[,1:12])
scaled_test[,1:12] = scale(scaled_test[,1:12])

# Binary Classification problem
logit_model = glm(Churn ~ ., data = scaled_train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = scaled_test[,-13], type = "response")

#Check prediction by value
logit_Predictions = ifelse(logit_Predictions > 0.5,1,0)

#Evaluate the performance of classification model
ConfMatrix_LG = table(cdf_test$Churn, logit_Predictions)
confusionMatrix(ConfMatrix_LG)

# Accuracy = 86%
# Precision = 1230/(1230+167) = 88%
# Recall = 1230 / (1230+32) = 97%
###########################  KNN Implementation #######################################
library(class)

#Predict test data
scaled_train[,1:12] = scale(scaled_train[,1:12])
scaled_test[,1:12] = scale(scaled_test[,1:12])
KNN_Predictions = knn(scaled_train[,-13], scaled_test[,-13], 
                      cl = scaled_train[,13], k = 5)


#Confusion matrix
Conf_matrix = table(scaled_test[,13], KNN_Predictions)
confusionMatrix(Conf_matrix)

# Accuracy = 89%
# Precision = 1253/(1253+136) = 90%
# Recall = 1253/(1253+19) = 98%
######################################################################################
# We will choose prediction Value of Decision Tree and random forest as they have highest accuracy

write.csv(DT_pred, file = 'churnpredictRF.csv')
write.csv(RF_Predictions, file = 'churnpredictDT.csv'