 {r}
# Loading necessary libraries
library(ranger)
library(caret)
library(data.table)
library(caTools)
library(pROC)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(gbm)
library(ROSE) # For SMOTE oversampling


{r}
# Loading data
creditcard_data <- read.csv("creditcard.csv")

# Exploratory Data Analysis (EDA)
# Visualizing data distributions
par(mfrow=c(1,2))
hist(creditcard_data$Amount, main="Amount Distribution")
barplot(table(creditcard_data$Class), main="Class Distribution")


{r}
# Applying SMOTE oversampling
creditcard_data_balanced <- ROSE(Class ~ ., data = creditcard_data)$data



{r}
# Plotting correlation matrix heatmap
library(corrplot)
corr_matrix <- cor(creditcard_data_balanced[, -ncol(creditcard_data_balanced)])
corrplot(corr_matrix, method = "color", type = "upper", order = "hclust", 
         addrect = 8, tl.cex = 0.7, diag = FALSE)


{r}
# Model Implementation Before SMOTE
# Splitting data
set.seed(123)
data_sample <- sample.split(creditcard_data$Class, SplitRatio=0.80)
train_data <- subset(creditcard_data, data_sample==TRUE)
test_data <- subset(creditcard_data, data_sample==FALSE)


{r}
# Plotting histograms for continuous features
par(mfrow=c(3,3))
for (i in 1:ncol(train_data)) {
  if (!is.factor(train_data[,i])) {
    hist(train_data[,i], main=paste("Histogram of", colnames(train_data)[i]))
  }
}



{r}
# Fitting Logistic Regression Model
Logistic_Model <- glm(Class ~ ., data = train_data, family = binomial())
summary(Logistic_Model)

{r}
plot(Logistic_Model)


{r}
# Fitting Decision Tree Model
decisionTree_model <- rpart(Class ~ . , data = train_data, method = 'class')
rpart.plot(decisionTree_model)


{r}
# Plotting correlation matrix heatmap
#corr_matrix <- cor(creditcard_data_balanced[, -ncol(creditcard_data_balanced)])
#corrplot(corr_matrix, method = "color", type = "upper", order = "hclust", 
#         addrect = 8, tl.cex = 0.7, diag = FALSE)


{r}
# Fitting Gradient Boosting Model
model_gbm <- gbm(Class ~ ., distribution = "bernoulli", data = train_data,
                 n.trees = 500, interaction.depth = 3, n.minobsinnode = 100,
                 shrinkage = 0.01, bag.fraction = 0.5,
                 train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data)))


{r}
# Model Implementation After SMOTE
# Splitting balanced data
set.seed(123)
data_sample_balanced <- sample.split(creditcard_data_balanced$Class, SplitRatio=0.80)
train_data_balanced <- subset(creditcard_data_balanced, data_sample_balanced==TRUE)
test_data_balanced <- subset(creditcard_data_balanced, data_sample_balanced==FALSE)


{r}
# Plotting histograms for continuous features in balanced data
par(mfrow=c(3,3))
for (i in 1:ncol(train_data_balanced)) {
  if (!is.factor(train_data_balanced[,i])) {
    hist(train_data_balanced[,i], main=paste("Histogram of", colnames(train_data_balanced)[i]))
  }
}


{r}
# Plotting bar plots for categorical features in balanced data
par(mfrow=c(1,1))
for (i in 1:ncol(train_data_balanced)) {
  if (is.factor(train_data_balanced[,i])) {
    barplot(table(train_data_balanced[,i]), main=paste("Barplot of", colnames(train_data_balanced)[i]))
  }
}


{r}
# Fitting Logistic Regression Model after SMOTE
Logistic_Model_balanced <- glm(Class ~ ., data = train_data_balanced, family = binomial())
summary(Logistic_Model_balanced)
plot(Logistic_Model_balanced)


{r}
# Fitting Decision Tree Model after SMOTE
decisionTree_model_balanced <- rpart(Class ~ . , data = train_data_balanced, method = 'class')
rpart.plot(decisionTree_model_balanced)


{r}
# Fitting Artificial Neural Network after SMOTE with smaller dataset
#train_data_subset <- train_data_balanced[sample(nrow(train_data_balanced), 10000, replace = FALSE), ]
#ANN_model_optimized_subset <- neuralnet(Class ~ ., train_data_subset, hidden = c(10), linear.output=FALSE,
#                                        learningrate = 0.01, algorithm = "rprop+", stepmax = 1e6)
#plot(ANN_model_optimized_subset)


{r}
# Fitting Gradient Boosting Model after SMOTE
model_gbm_balanced <- gbm(Class ~ ., distribution = "bernoulli", data = train_data_balanced,
                          n.trees = 500, interaction.depth = 3, n.minobsinnode = 100,
                          shrinkage = 0.01, bag.fraction = 0.5,
                          train.fraction = nrow(train_data_balanced) / 
                            (nrow(train_data_balanced) + nrow(test_data_balanced)))


{r}
# Evaluate models and plot confusion matrices
evaluate_model <- function(model, test_data) {
  predictions <- predict(model, test_data, type="response")
  pred_class <- ifelse(predictions > 0.5, 1, 0)
  confusionMatrix(data = factor(pred_class), reference = factor(test_data$Class))
}


{r}
# Define a function to evaluate model and calculate confusion matrix
evaluate_model <- function(model, test_data) {
  if (inherits(model, "rpart")) { # Decision tree model
    predictions <- predict(model, test_data, type = "class")
  } else { # Assume logistic regression, neural network, or other binary classification models
    predictions <- predict(model, test_data, type = "response")
    predictions <- ifelse(predictions > 0.5, 1, 0) # Convert probabilities to class labels
  }
  confusionMatrix(data = factor(predictions), reference = factor(test_data$Class))
}




{r}
# Confusion matrices before SMOTE
confusion_LR <- evaluate_model(Logistic_Model, test_data)
confusion_DT <- evaluate_model(decisionTree_model, test_data)
#confusion_ANN <- evaluate_model(ANN_model, test_data)
confusion_GBM <- evaluate_model(model_gbm, test_data)


{r}
# Confusion matrices after SMOTE
confusion_LR_balanced <- evaluate_model(Logistic_Model_balanced, test_data_balanced)
confusion_DT_balanced <- evaluate_model(decisionTree_model_balanced, test_data_balanced)
#confusion_ANN_optimized_subset <- evaluate_model(ANN_model_optimized_subset, test_data_balanced)
confusion_GBM_balanced <- evaluate_model(model_gbm_balanced, test_data_balanced)


{r}
# Plotting confusion matrices
par(mfrow=c(2,4))
plot(confusion_LR$table, main="LR Before SMOTE")
plot(confusion_DT$table, main="DT Before SMOTE")
#plot(confusion_ANN$table, main="ANN Before SMOTE")
plot(confusion_GBM$table, main="GBM Before SMOTE")

plot(confusion_LR_balanced$table, main="LR After SMOTE")
plot(confusion_DT_balanced$table, main="DT After SMOTE")
#plot(confusion_ANN_optimized_subset$table, main="ANN After SMOTE (Optimized)")
plot(confusion_GBM_balanced$table, main="GBM After SMOTE")
