library(lightgbm)
library(data.table)
library(Matrix)
library(ModelMetrics)

# Load 2021 data
data <- fread("2021 Data.csv")

# Specify numerical and categorical columns
numerical_features <- c("AGEYEARS", "EDMINS", "SBP", "PULSERATE", "TEMPERATURE", 
                        "RESPIRATORYRATE", "PULSEOXIMETRY", "HEIGHT", "WEIGHT", "TOTALGCS", "BEDSIZE", 
                        "RibFx", "BladderInj", "SmallBowelInj", "PelvicFx", "LowerExtremityFx", "HTX",
                        "PTX", "SCI", "Cspine", "TBI", "EsophagusInj", "HeartInj", "SkullFx", "Spleen",
                        "Liver", "Gallbladder", "BileDuct", "Pancreas", "Stomach", "Colon", "Rectum",
                        "Kidney", "Ureter", "Urethra", "FacialFx", "UpperExtremityFx", "ThoracicVessels",
                        "FlailChest", "CervicalCord", "ThoracicCord", "ALCOHOLSCREENRESULT", "ISS")

# Adjust the categorical features to exclude 'inc_key'
categorical_features <- setdiff(names(data), c(numerical_features, "HC_UNPLANNEDICU", "inc_key"))

# Convert categorical columns to factors and then to integer
data[, (categorical_features) := lapply(.SD, function(x) as.integer(as.factor(x))), .SDcols = categorical_features]

# Convert the target variable to a binary format for binary classification
data$HC_UNPLANNEDICU <- as.numeric(as.factor(data$HC_UNPLANNEDICU)) - 1

# Splitting the data into training and testing sets
set.seed(123)
train_indices <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Preparing LightGBM datasets
dtrain <- lgb.Dataset(data = as.matrix(train_data[, .SD, .SDcols = !c("HC_UNPLANNEDICU", "inc_key")]), 
                      label = train_data$HC_UNPLANNEDICU)
dtest <- lgb.Dataset(data = as.matrix(test_data[, .SD, .SDcols = !c("HC_UNPLANNEDICU", "inc_key")]), 
                     label = test_data$HC_UNPLANNEDICU)

# Calculate scale_pos_weight
positive_cases <- sum(train_data$HC_UNPLANNEDICU == 1)
negative_cases <- sum(train_data$HC_UNPLANNEDICU == 0)
scale_pos_weight_val <- negative_cases / positive_cases


# Set parameters for LightGBM
params <- list(
  objective = "binary",
  metric = c("binary_logloss", "auc"),
  boost_from_average = FALSE,
  scale_pos_weight = scale_pos_weight_val,
  learning_rate = 0.05,
  max_depth = 5,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  lambda_l1 = 0.1,
  lambda_l2 = 0.1
)

# Train the model
model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  valids = list(test = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

################################################################################
# This code determines optimal threshold based on threshold that maximizes F1 

# Initialize max F1 and corresponding threshold
max_f1 <- 0
optimal_threshold_f1 <- 0

# Iterate over a sequence of possible threshold values
for(threshold in seq(0, 1, by = 0.01)) {
  # Convert probabilities to binary predictions based on the threshold
  predicted_classes <- ifelse(predictions > threshold, 1, 0)
  
  # Calculate precision and recall
  precision <- sum(predicted_classes == 1 & test_data$HC_UNPLANNEDICU == 1) / sum(predicted_classes == 1)
  recall <- sum(predicted_classes == 1 & test_data$HC_UNPLANNEDICU == 1) / sum(test_data$HC_UNPLANNEDICU == 1)
  
  # Handle edge cases for precision and recall
  precision <- ifelse(is.nan(precision), 0, precision)
  recall <- ifelse(is.nan(recall), 0, recall)
  
  # Calculate F1 score
  f1 <- 2 * (precision * recall) / (precision + recall)
  f1 <- ifelse(is.nan(f1), 0, f1)
  
  # Check if this F1 is the best so far
  if(f1 > max_f1) {
    max_f1 <- f1
    optimal_threshold_f1 <- threshold
  }
}

# Print the optimal threshold based on F1 score
print(paste("Optimal Threshold for F1 score:", optimal_threshold_f1))

################################################################################

# Testing on 2020 data
# Load 2020 data
data_2020 <- fread("2020 Data.csv")

# Specify numerical and categorical columns
numerical_features <- c("AGEYEARS", "EDMINS", "SBP", "PULSERATE", "TEMPERATURE", 
                        "RESPIRATORYRATE", "PULSEOXIMETRY", "HEIGHT", "WEIGHT", "TOTALGCS", "BEDSIZE", 
                        "RibFx", "BladderInj", "SmallBowelInj", "PelvicFx", "LowerExtremityFx", "HTX",
                        "PTX", "SCI", "Cspine", "TBI", "EsophagusInj", "HeartInj", "SkullFx", "Spleen",
                        "Liver", "Gallbladder", "BileDuct", "Pancreas", "Stomach", "Colon", "Rectum",
                        "Kidney", "Ureter", "Urethra", "FacialFx", "UpperExtremityFx", "ThoracicVessels",
                        "FlailChest", "CervicalCord", "ThoracicCord", "ALCOHOLSCREENRESULT", "ISS")

# Identify categorical features
categorical_features <- setdiff(names(data_2020), c(numerical_features, "HC_UNPLANNEDICU", "inc_key"))

# Convert categorical columns to factors and then to integer
data_2020[, (categorical_features) := lapply(.SD, function(x) as.integer(as.factor(x))), .SDcols = categorical_features]

data_2020_prediction <- copy(data_2020)

# Prepare the data for prediction
test_data_2020_features <- data_2020_prediction[, !c("HC_UNPLANNEDICU", "inc_key"), with = FALSE]
test_label_2020 <- data_2020_prediction$HC_UNPLANNEDICU

# Predict on the 2020 test set
preds_test_2020 <- predict(model, as.matrix(test_data_2020_features))

# Calculate metrics for the 2020 test set
conf_matrix_2020 <- table(Predicted = preds_test_2020 > 0.76, True = test_label_2020)
accuracy_2020 <- sum(diag(conf_matrix_2020)) / sum(conf_matrix_2020)
precision_2020 <- conf_matrix_2020[2, 2] / sum(conf_matrix_2020[2, ])
recall_2020 <- conf_matrix_2020[2, 2] / sum(conf_matrix_2020[, 2])
specificity_2020 <- conf_matrix_2020[1, 1] / (conf_matrix_2020[1, 1] + conf_matrix_2020[1, 2])
f1_score_2020 <- 2 * (precision_2020 * recall_2020) / (precision_2020 + recall_2020)
auc_value_2020 <- auc(test_label_2020, preds_test_2020)

# Calculate the Brier score manually
brier_score_2020 <- mean((preds_test_2020 - data_2020$HC_UNPLANNEDICU)^2)

# Output metrics for the 2020 test set
cat("Metrics for 2020 Test Set:\n")
cat("Accuracy:", accuracy_2020, "\n")
cat("Precision:", precision_2020, "\n")
cat("Recall:", recall_2020, "\n")
cat("Specificity:", specificity_2020, "\n")
cat("F1 Score:", f1_score_2020, "\n")
cat("AUC:", auc_value_2020, "\n")
cat("Brier Score:", brier_score_2020, "\n") 

# Initialize max F1 and corresponding threshold for 2020 data
max_f1_2020 <- 0
optimal_threshold_f1_2020 <- 0

# Iterate over a sequence of possible threshold values for 2020 data
for(threshold in seq(0, 1, by = 0.01)) {
  # Convert probabilities to binary predictions based on the threshold for 2020 data
  predicted_classes_2020 <- ifelse(preds_test_2020 > threshold, 1, 0)
  
  # Calculate precision and recall for 2020 data
  precision_2020 <- sum(predicted_classes_2020 == 1 & test_label_2020 == 1) / sum(predicted_classes_2020 == 1)
  recall_2020 <- sum(predicted_classes_2020 == 1 & test_label_2020 == 1) / sum(test_label_2020 == 1)
  
  # Handle edge cases for precision and recall for 2020 data
  precision_2020 <- ifelse(is.nan(precision_2020), 0, precision_2020)
  recall_2020 <- ifelse(is.nan(recall_2020), 0, recall_2020)
  
  # Calculate F1 score for 2020 data
  f1_2020 <- 2 * (precision_2020 * recall_2020) / (precision_2020 + recall_2020)
  f1_2020 <- ifelse(is.nan(f1_2020), 0, f1_2020)
  
  # Check if this F1 is the best so far for 2020 data
  if(f1_2020 > max_f1_2020) {
    max_f1_2020 <- f1_2020
    optimal_threshold_f1_2020 <- threshold
  }
}

# Print the optimal threshold based on F1 score for 2020 data
print(paste("Optimal Threshold for F1 score (2020 Data):", optimal_threshold_f1_2020))
