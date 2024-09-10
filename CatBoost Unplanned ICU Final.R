library(catboost)
library(data.table)
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

# Convert the target variable to a binary  format for binary classification
data$HC_UNPLANNEDICU <- as.numeric(as.factor(data$HC_UNPLANNEDICU)) - 1

# Splitting the data into training and testing sets
set.seed(123) # for reproducibility
train_indices <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Prepare data for CatBoost
train_features <- train_data[, .SD, .SDcols = !c("HC_UNPLANNEDICU", "inc_key")]
train_labels <- train_data$HC_UNPLANNEDICU
test_features <- test_data[, .SD, .SDcols = !c("HC_UNPLANNEDICU", "inc_key")]
test_labels <- test_data$HC_UNPLANNEDICU

# Calculate scale_pos_weight
positive_cases <- sum(train_labels == 1)
negative_cases <- sum(train_labels == 0)
scale_pos_weight_val <- negative_cases / positive_cases

# Set parameters for CatBoost (without categorical_features)
params <- list(
  loss_function = "Logloss",
  eval_metric = "Logloss",
  iterations = 500,
  learning_rate = 0.05,
  depth = 5,
  l2_leaf_reg = 3,
  auto_class_weights = "Balanced",
  border_count = 32,
  random_seed = 123
)

# Prepare the pools for training and testing
train_pool <- catboost.load_pool(data = train_features, label = train_labels)
test_pool <- catboost.load_pool(data = test_features, label = test_labels)

# Train the CatBoost model
model <- catboost.train(
  learn_pool = train_pool,
  params = params
)

################################################################################

# Testing on 2020 data
# Load 2020 data
data_2020 <- fread("2020 Data.csv")

# Specify numerical and categorical columns
numerical_features_2020 <- c("AGEYEARS", "EDMINS", "SBP", "PULSERATE", "TEMPERATURE", 
                             "RESPIRATORYRATE", "PULSEOXIMETRY", "HEIGHT", "WEIGHT", "TOTALGCS", "BEDSIZE", 
                             "RibFx", "BladderInj", "SmallBowelInj", "PelvicFx", "LowerExtremityFx", "HTX",
                             "PTX", "SCI", "Cspine", "TBI", "EsophagusInj", "HeartInj", "SkullFx", "Spleen",
                             "Liver", "Gallbladder", "BileDuct", "Pancreas", "Stomach", "Colon", "Rectum",
                             "Kidney", "Ureter", "Urethra", "FacialFx", "UpperExtremityFx", "ThoracicVessels",
                             "FlailChest", "CervicalCord", "ThoracicCord", "ALCOHOLSCREENRESULT", "ISS")

# Identify categorical features
categorical_features_2020 <- setdiff(names(data_2020), c(numerical_features_2020, "HC_UNPLANNEDICU", "inc_key"))

# Convert categorical columns to factors and then to integer
data_2020[, (categorical_features_2020) := lapply(.SD, function(x) as.integer(as.factor(x))), .SDcols = categorical_features_2020]

# Prepare the data for prediction
test_data_2020_features <- data_2020[, !c("HC_UNPLANNEDICU", "inc_key"), with = FALSE]
test_label_2020 <- data_2020$HC_UNPLANNEDICU

# Create a CatBoost pool for the 2020 data
test_pool_2020 <- catboost.load_pool(data = test_data_2020_features, label = test_label_2020, cat_features = categorical_features_2020)

# Get predictions on the 2020 test set
test_predictions_2020 <- catboost.predict(model, test_pool_2020)

# Convert test_predictions_2020 to a data frame
test_predictions_2020_df <- data.frame(Probability = test_predictions_2020, Actual = test_label_2020)

# Convert raw predictions to probabilities
test_predictions_2020 <- 1 / (1 + exp(-test_predictions_2020))

# Calculate metrics for the 2020 test set
conf_matrix_2020 <- table(Predicted = ifelse(test_predictions_2020 > 0.77, 1, 0), True = test_label_2020)
tp_2020 <- conf_matrix_2020[2, 2]
fp_2020 <- conf_matrix_2020[1, 2]
tn_2020 <- conf_matrix_2020[1, 1]
fn_2020 <- conf_matrix_2020[2, 1]
accuracy_2020 <- (tp_2020 + tn_2020) / (tp_2020 + tn_2020 + fp_2020 + fn_2020)
precision_2020 <- tp_2020 / (tp_2020 + fp_2020)
recall_2020 <- tp_2020 / (tp_2020 + fn_2020)
specificity_2020 <- tn_2020 / (tn_2020 + fp_2020)
# Calculate AUC using ModelMetrics for the 2020 test set

# Calculate F1 Score
f1_score_2020 <- 2 * (precision_2020 * recall_2020) / (precision_2020 + recall_2020)

auc_value_2020 <- auc(ifelse(test_label_2020 == 1, 1, 0), test_predictions_2020)

# Calculate the Brier score manually
brier_score_2020 <- mean((test_predictions_2020 - data_2020$HC_UNPLANNEDICU)^2)

# Print the calculated metrics for the 2020 test set
cat("Accuracy (2020):", accuracy_2020, "\n")
cat("Precision (2020):", precision_2020, "\n")
cat("Recall (2020):", recall_2020, "\n")
cat("AUC (2020):", auc_value_2020, "\n")
cat("Brier Score (2020):", brier_score_2020, "\n") 
cat("Specificity (2020):", specificity_2020, "\n")

# Print the F1 Score
cat("F1 Score:", f1_score_2020, "\n")

################################################################################
# Initialize max F1 and corresponding threshold for 2020 data
max_f1_2020 <- 0
optimal_threshold_f1_2020 <- 0

# Iterate over a sequence of possible threshold values for 2020 data
for(threshold in seq(0, 1, by = 0.01)) {
  # Convert probabilities to binary predictions based on the threshold for 2020 data
  predicted_classes_2020 <- ifelse(test_predictions_2020 > threshold, 1, 0)
  
  # Calculate precision and recall for 2020 data
  precision_2020 <- sum(predicted_classes_2020 == 1 & test_label_2020 == 1) / sum(predicted_classes_2020 == 1)
  recall_2020 <- sum(predicted_classes_2020 == 1 & test_label_2020 == 1) / sum(test_label_2020 == 1)
  
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

# Output the optimal F1 score and corresponding threshold
cat("Max F1 Score (2020):", max_f1_2020, "\n")
cat("Optimal Threshold for F1 (2020):", optimal_threshold_f1_2020, "\n")

#########################################################################                                                  
                                                  
# Prepare the data for SHAP analysis
test_data_2020_features <- data_2020[, !c("HC_UNPLANNEDICU", "inc_key"), with = FALSE]
test_label_2020 <- data_2020$HC_UNPLANNEDICU

# Create a CatBoost pool for the 2020 data
test_pool_2020 <- catboost.load_pool(data = test_data_2020_features, 
                                     label = test_label_2020, 
                                     cat_features = categorical_features_2020)

# Calculate SHAP values
shap_values <- catboost.get_feature_importance(model, 
                                               pool = test_pool_2020, 
                                               type = 'ShapValues')

shap_values <- shap_values[, -ncol(shap_values)]

mean_shap_values <- colMeans(abs(shap_values))

# Create a data frame of feature importances
importance_df <- data.frame(
  feature = colnames(test_data_2020_features),
  importance = mean_shap_values
)

# Sort by importance
importance_df <- importance_df[order(-importance_df$importance), ]

# Print the top 10 most important features
print(head(importance_df, 10))
                                            
