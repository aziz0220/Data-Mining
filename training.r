# Load necessary libraries
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(ggplot2)
library(pheatmap)
library(viridis)
library(rpart.plot)
library(nnet)

# Download and read the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
download.file(url, destfile = "student.zip")
unzip("student.zip")
data <- read.csv("student-mat.csv", sep = ";")

# Redefine "status" with adjusted thresholds
data$status <- NA
data$status[data$absences >= 10 & data$G3 <= 10] <- "Dropout"
data$status[data$absences < 10 & data$absences >= 5 & data$G3 > 10 & data$G3 <= 15] <- "Enrolled"
data$status[data$absences < 5 & data$G3 > 15] <- "Graduate"
data <- na.omit(data)

# Ensure "status" is a factor with specified levels
data$status <- factor(data$status, levels = c("Dropout", "Enrolled", "Graduate"))

# Identify categorical and numerical variables
categorical_vars <- c("school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                      "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
                      "nursery", "higher", "internet", "romantic")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Normalize numerical variables excluding "status", "G3", and "absences"
numerical_vars <- names(data)[sapply(data, is.numeric)]
numerical_vars <- setdiff(numerical_vars, c("status", "G3", "absences"))
normalize <- function(x) {(x - min(x)) / (max(x) - min(x))}
data[numerical_vars] <- lapply(data[numerical_vars], normalize)

# Stratified Sampling
set.seed(123)
train_index <- createDataPartition(data$status, p = 0.7, list = FALSE, times = 1)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Decision Tree Model
dt_model <- rpart(status ~ ., data = train_data, method = "class")
rpart.plot(dt_model, extra = 1, main = "Decision Tree for Student Classification")
dt_pred <- predict(dt_model, newdata = test_data, type = "class")

# Random Forest Model with Tuning and Class Weights using caret
set.seed(123)
ctrl <- trainControl(method = "cv", number = 5)
grid <- expand.grid(mtry = c(1:10))  # Adjust the range as needed
rf_model_tuned <- train(status ~ ., data = train_data, method = "rf", tuneGrid = grid, 
                        trControl = ctrl, classwt = c(Dropout = 2, Enrolled = 2, Graduate = 1), importance = TRUE)
rf_pred <- predict(rf_model_tuned, newdata = test_data)

# Multinomial Logistic Regression
mlr_model <- multinom(status ~ ., data = train_data)
mlr_pred <- predict(mlr_model, newdata = test_data)

# Confusion Matrices
dt_cm <- confusionMatrix(dt_pred, test_data$status)
print("Decision Tree Confusion Matrix:")
print(dt_cm$table)

rf_cm <- confusionMatrix(rf_pred, test_data$status)
print("Random Forest Confusion Matrix:")
print(rf_cm$table)

mlr_cm <- confusionMatrix(mlr_pred, test_data$status)
print("Multinomial Logistic Regression Confusion Matrix:")
print(mlr_cm$table)

# Feature Importance for Random Forest
rf_importance <- importance(rf_model_tuned$finalModel)
rf_importance_df <- data.frame(Feature = row.names(rf_importance),
                               Importance = rf_importance[, "MeanDecreaseGini"])
rf_importance_df <- rf_importance_df[order(-rf_importance_df$Importance), ]
print("Top 10 Important Features for Random Forest:")
print(head(rf_importance_df, 10))
