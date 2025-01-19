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

# Create binary target variable
data$G3_binary <- ifelse(data$G3 >= 10, 1, 0)
data$G3_binary <- factor(data$G3_binary, levels = c(0, 1))

# Identify categorical and numerical variables
categorical_vars <- c("school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                      "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
                      "nursery", "higher", "internet", "romantic")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Normalize numerical variables excluding G3 and G3_binary
numerical_vars <- names(data)[sapply(data, is.numeric)]
numerical_vars <- setdiff(numerical_vars, c("G3", "G3_binary"))
normalize <- function(x) {(x - min(x)) / (max(x) - min(x))}
data[numerical_vars] <- lapply(data[numerical_vars], normalize)

# **1. Correlation Heatmap**
numerical_data <- data[, numerical_vars]
cor_matrix <- cor(numerical_data)
pheatmap(cor_matrix, cluster_rows = TRUE, cluster_cols = TRUE,
         main = "Correlation Heatmap of Numerical Features", color = viridis::viridis(100))

# Define predictor variables
predictors <- setdiff(names(data), c("G3", "G3_binary"))

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data$G3_binary, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# **2. Decision Tree Plot**
# Build Decision Tree model
dt_model <- rpart(G3_binary ~ ., data = train_data[, c(predictors, "G3_binary")], method = "class")
rpart.plot(dt_model, extra = 1, main = "Decision Tree for Student Performance")

# Predict on test set
dt_pred <- predict(dt_model, test_data[, predictors], type = "class")
dt_prob <- predict(dt_model, test_data[, predictors], type = "prob")

# **3. Random Forest Model**
# Build Random Forest model
rf_model <- randomForest(G3_binary ~ ., data = train_data[, c(predictors, "G3_binary")], importance = TRUE)

# Predict on test set
rf_pred <- predict(rf_model, test_data[, predictors])
rf_prob <- predict(rf_model, test_data[, predictors], type = "prob")

# **4. Confusion Matrices**
# Evaluate Decision Tree
dt_cm <- confusionMatrix(dt_pred, test_data$G3_binary)
print("Decision Tree Confusion Matrix:")
print(dt_cm$table)
pheatmap(as.matrix(dt_cm$table), cluster_rows = FALSE, cluster_cols = FALSE,
         main = "Decision Tree Confusion Matrix Heatmap", color = viridis::viridis(10))

# Evaluate Random Forest
rf_cm <- confusionMatrix(rf_pred, test_data$G3_binary)
print("Random Forest Confusion Matrix:")
print(rf_cm$table)
pheatmap(as.matrix(rf_cm$table), cluster_rows = FALSE, cluster_cols = FALSE,
         main = "Random Forest Confusion Matrix Heatmap", color = viridis::viridis(10))

# **5. ROC Curves Comparison**
# ROC Curves
dt_pred_obj <- prediction(dt_prob[,2], as.numeric(test_data$G3_binary))
dt_perf <- performance(dt_pred_obj, "tpr", "fpr")
dt_auc <- performance(dt_pred_obj, "auc")@y.values[[1]]

rf_pred_obj <- prediction(rf_prob[,2], as.numeric(test_data$G3_binary))
rf_perf <- performance(rf_pred_obj, "tpr", "fpr")
rf_auc <- performance(rf_pred_obj, "auc")@y.values[[1]]

# Plot ROC curves
plot(dt_perf, col = "blue", main = "ROC Curves Comparison", xlab = "False Positive Rate", ylab = "True Positive Rate")
plot(rf_perf, col = "red", add = TRUE)
legend("bottomright",
       legend = c(paste("Decision Tree (AUC =", round(dt_auc, 3), ")"),
                  paste("Random Forest (AUC =", round(rf_auc, 3), ")")),
       col = c("blue", "red"),
       lty = 1)

# **6. Metric Comparison for Each Class**
# Decision Tree metrics by class
print("Decision Tree Metrics by Class:")
print(dt_cm$byClass)

# Random Forest metrics by class
print("Random Forest Metrics by Class:")
print(rf_cm$byClass)

# **7. Feature Importance for Random Forest**
rf_importance <- importance(rf_model)
rf_importance_df <- data.frame(Feature = row.names(rf_importance),
                               Importance = rf_importance[, "MeanDecreaseGini"])
rf_importance_df <- rf_importance_df[order(-rf_importance_df$Importance), ]
print("Top 10 Important Features for Random Forest:")
print(head(rf_importance_df, 10))
