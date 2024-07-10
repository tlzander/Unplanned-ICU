#XGBoost Unplanned ICU

library(xgboost)
library(data.table)
library(Matrix)
library(ModelMetrics)

#Load 2021 Data
data <- fread("2021 Data.csv")

# Specify numerical and categorical columns
numerical_features <- c("AGEYEARS", "EDMINS", "SBP", "PULSERATE", "TEMPERATURE", 
                        "RESPIRATORYRATE", "PULSEOXIMETRY", "HEIGHT", "WEIGHT", "TOTALGCS", "BEDSIZE", 
                        "RibFx", "BladderInj", "SmallBowelInj", "PelvicFx", "LowerExtremityFx",
                        "HTX", "PTX", "SCI", "Cspine", "TBI", "EsophagusInj", "HeartInj", "SkullFx",
                        "Spleen", "Liver", "Gallbladder", "BileDuct", "Pancreas", "Stomach", "Colon",
                        "Rectum", "Kidney", "Ureter", "Urethra", "FacialFx", "UpperExtremityFx",
                        "ThoracicVessels", "FlailChest", "CervicalCord", "ThoracicCord", 
                        "ALCOHOLSCREENRESULT", "ISS")

categorical_features <- setdiff(names(data), c(numerical_features, "HC_UNPLANNEDICU", "inc_key"))

# Convert categorical columns to factors and then to integer
data[, (categorical_features) := lapply(.SD, function(x) as.integer(as.factor(x))), .SDcols = categorical_features]

# Convert the target variable to a binary  format for binary classification
data$HC_UNPLANNEDICU <- as.numeric(as.factor(data$HC_UNPLANNEDICU)) - 1

# Splitting the data into training and testing sets
set.seed(123)
train_indices <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Prepare data for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, .SD, .SDcols = !c("HC_UNPLANNEDICU", "inc_key")]), 
                      label = train_data$HC_UNPLANNEDICU)
dtest <- xgb.DMatrix(data = as.matrix(test_data[, .SD, .SDcols = !c("HC_UNPLANNEDICU", "inc_key")]), 
                     label = test_data$HC_UNPLANNEDICU)

# Calculate scale_pos_weight
positive_cases <- sum(train_data$HC_UNPLANNEDICU == 1)
negative_cases <- sum(train_data$HC_UNPLANNEDICU == 0)
scale_pos_weight_val <- negative_cases / positive_cases

# Set parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  scale_pos_weight = scale_pos_weight_val,
  eta = 0.05,
  max_depth = 5,
  colsample_bytree = 0.8,
  subsample = 0.8,
  lambda = 0.1,
  alpha = 0.1
)

# Train the model
model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(test = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

################################################################################
# Testing on 2020 data
# Load 2020 data
data_2020 <- fread("2020 Data")

# Specify numerical and categorical columns (unchanged from your previous code)
numerical_features <- c("AGEYEARS", "EDMINS", "SBP", "PULSERATE", "TEMPERATURE", 
                        "RESPIRATORYRATE", "PULSEOXIMETRY", "HEIGHT", "WEIGHT", "TOTALGCS", "BEDSIZE", 
                        "RibFx", "BladderInj", "SmallBowelInj", "PelvicFx", "LowerExtremityFx", "HTX",
                        "PTX", "SCI", "Cspine", "TBI", "EsophagusInj", "HeartInj", "SkullFx", "Spleen",
                        "Liver", "Gallbladder", "BileDuct", "Pancreas", "Stomach", "Colon", "Rectum",
                        "Kidney", "Ureter", "Urethra", "FacialFx", "UpperExtremityFx", "ThoracicVessels",
                        "FlailChest", "CervicalCord", "ThoracicCord", "ALCOHOLSCREENRESULT", "ISS")

# Identify categorical features
categorical_features <- setdiff(names(data_2020), c(numerical_features, "HC_UNPLANNEDICU", "inc_key"))

# Get the feature names from the training set
train_features <- setdiff(colnames(train_data), c("HC_UNPLANNEDICU", "inc_key"))

# Ensure the 2020 data has the same features in the same order
data_2020 <- data_2020[, c(train_features, "HC_UNPLANNEDICU"), with = FALSE]

# Convert categorical columns to factors and then to integer
data_2020[, (categorical_features) := lapply(.SD, function(x) as.integer(as.factor(x))), .SDcols = categorical_features]

# Ensure the target variable is prepared in the same way as for training data
data_2020$HC_UNPLANNEDICU <- as.numeric(as.factor(data_2020$HC_UNPLANNEDICU)) - 1

# Prepare the data for prediction
dtest_2020 <- xgb.DMatrix(data = as.matrix(data_2020[, train_features, with = FALSE]), 
                          label = data_2020$HC_UNPLANNEDICU)

#Predicting
preds_test_2020 <- predict(model, dtest_2020)


# Convert predictions to binary
predicted_classes_2020 <- ifelse(preds_test_2020 > 0.76, 1, 0)

# Calculate metrics for the 2020 test set
conf_matrix_2020 <- table(Predicted = predicted_classes_2020, True = data_2020$HC_UNPLANNEDICU)
accuracy_2020 <- sum(diag(conf_matrix_2020)) / sum(conf_matrix_2020)
precision_2020 <- conf_matrix_2020[2, 2] / sum(conf_matrix_2020[2, ])
recall_2020 <- conf_matrix_2020[2, 2] / sum(conf_matrix_2020[, 2])
specificity_2020 <- conf_matrix_2020[1, 1] / (conf_matrix_2020[1, 1] + conf_matrix_2020[1, 2])
f1_score_2020 <- 2 * (precision_2020 * recall_2020) / (precision_2020 + recall_2020)
auc_value_2020 <- auc(data_2020$HC_UNPLANNEDICU, preds_test_2020)

# Calculate the Brier score
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

# Determine optimal threshold based on F1 score for 2020 data
max_f1_2020 <- 0
optimal_threshold_f1_2020 <- 0

for(threshold in seq(0, 1, by = 0.01)) {
  predicted_classes_2020 <- ifelse(preds_test_2020 > threshold, 1, 0)
  precision_2020 <- sum(predicted_classes_2020 == 1 & data_2020$HC_UNPLANNEDICU == 1) / sum(predicted_classes_2020 == 1)
  recall_2020 <- sum(predicted_classes_2020 == 1 & data_2020$HC_UNPLANNEDICU == 1) / sum(data_2020$HC_UNPLANNEDICU == 1)
  
  # Avoid division by zero
  precision_2020 <- ifelse(is.nan(precision_2020), 0, precision_2020)
  recall_2020 <- ifelse(is.nan(recall_2020), 0, recall_2020)
  
  f1_2020 <- 2 * (precision_2020 * recall_2020) / (precision_2020 + recall_2020)
  f1_2020 <- ifelse(is.nan(f1_2020), 0, f1_2020)
  
  if(f1_2020 > max_f1_2020) {
    max_f1_2020 <- f1_2020
    optimal_threshold_f1_2020 <- threshold
  }
}

# Print the optimal threshold based on F1 score for 2020 data
print(paste("Optimal Threshold for F1 score (2020 Data):", optimal_threshold_f1_2020))

################################################################################
#Calculate mean absolute SHAP values

library(SHAPforxgboost)

# Define test_data_2020_features
test_data_2020_features <- data_2020[, train_features, with = FALSE]

# Use all rows from the test dataset
X <- data.matrix(test_data_2020_features[sample(nrow(test_data_2020_features), 538583), ])

# Prepare SHAP values
shap <- SHAPforxgboost::shap.prep(model, X_train = X)

shap.importance(shap)

# Compute SHAP importance
importance <- shap.importance(shap)

# Convert to dataframe
importance_df <- as.data.frame(importance)

# View the dataframe
print(importance_df)

################################################################################
# SHAP DEPENDENCY PLOTS
# Get SHAP feature importance
shap_importance_df <- shap.importance(shap)

# Select the top 10 features
top_10_shap <- head(shap_importance_df, 10)

# Loop and generate the dependence plots for top 10 important features based on SHAP values
suppressWarnings({
  for (v in top_10_shap$variable) {
    p <- SHAPforxgboost::shap.plot.dependence(shap, v, color_feature = "NULL",
                                              alpha = 0.5, jitter_width = 0.1) +
      ggplot2::ggtitle(v)
    print(p)
  }
})

# Start the PDF graphics device
pdf("Shapley_plots_UnplannedICU.pdf")

# Loop through and generate each plot only for top 10 important features
for (v in top_10_shap$variable) {
  p <- SHAPforxgboost::shap.plot.dependence(shap, v, color_feature = "NULL",
                                            alpha = 0.5, jitter_width = 0.1) +
    ggplot2::ggtitle(v)
  print(p)
}

# Close the PDF graphics device
dev.off()

################################################################################
#Assess precision & recall by model threshold

# Separate patients based on actual results
unplanned_icu_patients_2020 <- data_2020[HC_UNPLANNEDICU == 1, ]
non_icu_patients_2020 <- data_2020[HC_UNPLANNEDICU == 0, ]

# Count the number of unplanned ICU patients
num_unplanned_icu_patients_2020 <- nrow(unplanned_icu_patients_2020)
cat("Number of unplanned ICU patients in 2020:", num_unplanned_icu_patients_2020, "\n")

# Count the number of non-ICU patients
num_non_icu_patients_2020 <- nrow(non_icu_patients_2020)
cat("Number of non-ICU patients in 2020:", num_non_icu_patients_2020, "\n")


# Define bins for predicted probabilities
bins <- seq(0, 1, by = 0.05)
risk_df <- data.frame(Threshold = head(bins, -1), 
                      UnplannedICUAdmits = integer(length(bins) - 1),
                      ActualNegatives = integer(length(bins) - 1),
                      TotalPatients = integer(length(bins) - 1),
                      Risk = numeric(length(bins) - 1))

# Add predicted probabilities to the data_2020 dataset
data_2020$PredictedProbability <- preds_test_2020

# Calculate counts and risk for each bin
for (i in 1:(length(bins) - 1)) {
  lower_bound <- bins[i]
  upper_bound <- bins[i + 1]
  
  # Select patients within the current bin
  patients_in_bin <- data_2020[data_2020$PredictedProbability > lower_bound & data_2020$PredictedProbability <= upper_bound, ]
  
  # Count unplanned ICU admissions (HC_UNPLANNEDICU == 1)
  num_unplanned_icu_admits <- sum(patients_in_bin$HC_UNPLANNEDICU == 1)
  risk_df$UnplannedICUAdmits[i] <- num_unplanned_icu_admits
  
  # Count actual negatives (HC_UNPLANNEDICU == 0)
  num_actual_negatives <- sum(patients_in_bin$HC_UNPLANNEDICU == 0)
  risk_df$ActualNegatives[i] <- num_actual_negatives
  
  # Calculate total patients in the bin
  total_patients <- num_unplanned_icu_admits + num_actual_negatives
  risk_df$TotalPatients[i] <- total_patients
  
  # Calculate risk
  risk_df$Risk[i] <- ifelse(total_patients > 0, 
                            num_unplanned_icu_admits / total_patients, 
                            0)
}

# Print the risk dataframe
print(risk_df)

# Define bins for predicted probabilities
bins <- seq(0, 1, by = 0.05)
risk_df <- data.frame(Threshold = head(bins, -1), 
                      UnplannedReadmissions = integer(length(bins) - 1),
                      TotalPatients = integer(length(bins) - 1),
                      CumulativeRisk = numeric(length(bins) - 1),
                      PercentTotalUnplannedReadmissions = numeric(length(bins) - 1))

# Add predicted probabilities to the test_data dataset
#data_2020$PredictedProbability <- predictions

# Calculate the total number of unplanned readmissions
total_unplanned_readmissions <- sum(data_2020$HC_UNPLANNEDICU == 1)

# Calculate counts, cumulative risk and percent of total unplanned readmissions for each bin
for (i in 1:(length(bins) - 1)) {
  lower_bound <- bins[i]
  
  # Select patients with predicted probability greater than or equal to the current threshold
  patients_above_threshold <- data_2020[data_2020$PredictedProbability >= lower_bound, ]
  
  # Count unplanned readmissions (UNPLANNEDREADMISSION1 == 1)
  num_unplanned_readmissions <- sum(patients_above_threshold$HC_UNPLANNEDICU == 1)
  risk_df$UnplannedReadmissions[i] <- num_unplanned_readmissions
  
  # Calculate total patients above the threshold
  total_patients <- nrow(patients_above_threshold)
  risk_df$TotalPatients[i] <- total_patients
  
  # Calculate cumulative risk
  risk_df$CumulativeRisk[i] <- ifelse(total_patients > 0, 
                                      num_unplanned_readmissions / total_patients * 100, 
                                      0)
  
  # Calculate percent of total unplanned readmissions
  risk_df$PercentTotalUnplannedReadmissions[i] <- num_unplanned_readmissions / total_unplanned_readmissions * 100
}

# Print the risk dataframe
print(risk_df)
