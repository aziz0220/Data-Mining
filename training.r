# Load necessary libraries
# library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)
library(pheatmap)

# Load the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
temp <- tempfile()
download.file(url, temp)
unzip(temp, exdir = "student_data")
data <- read.csv("student_data/student-por.csv", sep = ";")

# Clean up temporary files
unlink(temp)

# Explore the dataset
head(data)
str(data)

# Check for missing values
sum(is.na(data))  # No missing values

# Convert the target variable to a factor
data$Target <- factor(data$Target, levels = c("Dropout", "Enrolled", "Graduate"))

# Normalize numerical features (if needed)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

numerical_cols <- sapply(data, is.numeric)
data[numerical_cols] <- lapply(data[numerical_cols], normalize)

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data$Target, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Decision Tree Model
dt_model <- rpart(Target ~ ., 
                  data = train_data,
                  method = "class",
                  control = rpart.control(maxdepth = 5))

# Plot the decision tree
windows()
rpart.plot(dt_model, extra = 1, main = "Decision Tree for Student Dropout Prediction")

# Make predictions using the Decision Tree model
dt_pred <- predict(dt_model, test_data, type = "class")
dt_prob <- predict(dt_model, test_data, type = "prob")

# Evaluate the Decision Tree model
dt_cm <- confusionMatrix(dt_pred, test_data$Target)
print("Decision Tree Confusion Matrix:")
print(dt_cm)

# Random Forest Model
rf_model <- randomForest(Target ~ ., 
                         data = train_data,
                         ntree = 100,
                         importance = TRUE)

# Make predictions using the Random Forest model
rf_pred <- predict(rf_model, test_data)
rf_prob <- predict(rf_model, test_data, type = "prob")

# Evaluate the Random Forest model
rf_cm <- confusionMatrix(rf_pred, test_data$Target)
print("Random Forest Confusion Matrix:")
print(rf_cm)

# Feature Importance for Random Forest
importance(rf_model)
varImpPlot(rf_model, main = "Feature Importance (Random Forest)")

# ROC Curve and AUC for Decision Tree
dt_roc <- multiclass.roc(test_data$Target, dt_prob)
print("Decision Tree AUC:")
print(dt_roc$auc)

# ROC Curve and AUC for Random Forest
rf_roc <- multiclass.roc(test_data$Target, rf_prob)
print("Random Forest AUC:")
print(rf_roc$auc)

# Confusion Matrix Heatmaps
# Decision Tree
windows()
pheatmap(dt_cm$table, cluster_rows = FALSE, cluster_cols = FALSE, 
         main = "Decision Tree Confusion Matrix Heatmap", color = viridis::viridis(10))

# Random Forest
windows()
pheatmap(rf_cm$table, cluster_rows = FALSE, cluster_cols = FALSE, 
         main = "Random Forest Confusion Matrix Heatmap", color = viridis::viridis(10))

# Compare results
results_comparison <- data.frame(
  Model = c("Decision Tree", "Random Forest"),
  Accuracy = c(dt_cm$overall["Accuracy"], rf_cm$overall["Accuracy"]),
  Precision = c(dt_cm$byClass["Precision"], rf_cm$byClass["Precision"]),
  Recall = c(dt_cm$byClass["Recall"], rf_cm$byClass["Recall"]),
  Specificity = c(dt_cm$byClass["Specificity"], rf_cm$byClass["Specificity"]),
  F1_Score = c(2 * (dt_cm$byClass["Precision"] * dt_cm$byClass["Recall"]) / (dt_cm$byClass["Precision"] + dt_cm$byClass["Recall"]),
               2 * (rf_cm$byClass["Precision"] * rf_cm$byClass["Recall"]) / (rf_cm$byClass["Precision"] + rf_cm$byClass["Recall"])),
  Kappa = c(dt_cm$overall["Kappa"], rf_cm$overall["Kappa"]),
  AUC = c(dt_roc$auc, rf_roc$auc)
)

print("\nModel Comparison Results:")
print(results_comparison)