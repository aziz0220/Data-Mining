# Load necessary libraries
library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(ROCR)
library(ggplot2)
library(pheatmap)
library(viridis)
library(rpart.plot)

# Download and read the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
download.file(url, destfile = "student.zip")
unzip("student.zip")
data <- read.csv("student-mat.csv", sep = ";")

# Create the new target variable "status" before normalization
data$status <- NA
data$status[data$absences >= 30 & data$G3 <= 10] <- "Dropout"
data$status[data$absences < 30 & data$absences >= 15 & data$G3 > 10 & data$G3 <= 15] <- "Enrolled"
data$status[data$absences < 15 & data$G3 > 15] <- "Graduate"
data <- na.omit(data)

# Ensure "status" is a factor with specified levels
data$status <- factor(data$status, levels = c("Dropout", "Enrolled", "Graduate"))

# Identify categorical and numerical variables
categorical_vars <- c("school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                      "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
                      "nursery", "higher", "internet", "romantic")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Normalize numerical variables excluding "status" and "G3"
numerical_vars <- names(data)[sapply(data, is.numeric)]
numerical_vars <- setdiff(numerical_vars, c("status", "G3"))
normalize <- function(x) {(x - min(x)) / (max(x) - min(x))}
data[numerical_vars] <- lapply(data[numerical_vars], normalize)

# 1. Correlation Heatmap
numerical_data <- data[, numerical_vars]
cor_matrix <- cor(numerical_data)
pheatmap(cor_matrix, cluster_rows = TRUE, cluster_cols = TRUE,
         main = "Correlation Heatmap of Numerical Features", color = viridis::viridis(100))

# Define predictor variables
predictors <- setdiff(names(data), "status")

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data$status, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# 2. Decision Tree Plot
# Build Decision Tree model
dt_model <- rpart(status ~ ., data = train_data, method = "class")
rpart.plot(dt_model, extra = 1, main = "Decision Tree for Student Classification")

# Predict on test set
dt_pred <- predict(dt_model, newdata = test_data, type = "class")

# 3. Random Forest Model
# Build Random Forest model
rf_model <- randomForest(status ~ ., data = train_data, importance = TRUE)

# Predict on test set
rf_pred <- predict(rf_model, newdata = test_data)

# 4. Confusion Matrices
# Evaluate Decision Tree
dt_cm <- confusionMatrix(dt_pred, test_data$status)
print("Decision Tree Confusion Matrix:")
print(dt_cm$table)
pheatmap(as.matrix(dt_cm$table), cluster_rows = FALSE, cluster_cols = FALSE,
         main = "Decision Tree Confusion Matrix Heatmap", color = viridis::viridis(10))

# Evaluate Random Forest
rf_cm <- confusionMatrix(rf_pred, test_data$status)
print("Random Forest Confusion Matrix:")
print(rf_cm$table)
pheatmap(as.matrix(rf_cm$table), cluster_rows = FALSE, cluster_cols = FALSE,
         main = "Random Forest Confusion Matrix Heatmap", color = viridis::viridis(10))

# 5. Metric Comparison for Each Class
# Decision Tree metrics by class
print("Decision Tree Metrics by Class:")
print(dt_cm$byClass)

# Random Forest metrics by class
print("Random Forest Metrics by Class:")
print(rf_cm$byClass)

# 6. Feature Importance for Random Forest
rf_importance <- importance(rf_model)
rf_importance_df <- data.frame(Feature = row.names(rf_importance),
                               Importance = rf_importance[, "MeanDecreaseGini"])
rf_importance_df <- rf_importance_df[order(-rf_importance_df$Importance), ]
print("Top 10 Important Features for Random Forest:")
print(head(rf_importance_df, 10))
