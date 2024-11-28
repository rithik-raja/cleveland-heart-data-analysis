####################
# PRE-PROCESSING
####################

library(DMwR2)
library(caret)
library(e1071)
library(randomForest)
library(ggplot2)
rm(list=ls())
set.seed(300)

column_names <- c(
  "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
  "exang", "oldpeak", "slope", "ca", "thal", "num"
)

# load data and fill NA values based on chosen method:
# 0 - use KNN imputation (fill based on the k nearest neighbors)
# 1 - simply drop rows with missing values
get_heart_data <- function(clean_method = 0) {
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
  data <- read.csv(url, header = FALSE, na.strings = "?")
  colnames(data) <- column_names
  data$num <- ifelse(data$num > 0, 1, 0)
  
  data$sex <- as.factor(data$sex)
  data$cp <- as.factor(data$cp)
  data$fbs <- as.factor(data$fbs)
  data$restecg <- as.factor(data$restecg)
  data$exang <- as.factor(data$exang)
  data$slope <- as.factor(data$slope)
  #data$ca <- as.factor(data$ca)
  data$thal <- as.factor(data$thal)
  
  if (clean_method == 0) {
    data <- knnImputation(data)
  } else {
    data <- na.omit(data)
  }
  
  return(data)
}

####################
# CV FUNCTION
####################

# run a repeated cross-validation on the chosen model
run_cross_validation <- function(data, model_function, k = 5, repeats = 10) {
  
  accuracies <- numeric(k * repeats)
  count <- 0
  for (r in 1:repeats) {
    data <- data[sample(nrow(data)), ] # shuffle data to avoid duplicate accuracies
    folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)
    
    for (i in 1:k) {
      count <- count + 1
      test_indices <- which(folds == i, arr.ind = TRUE)
      test_data <- data[test_indices, ]
      train_data <- data[-test_indices, ]
      res <- model_function(train_data, test_data)
      accuracies[count] <- res$accuracy
    }
  }
  
  return(accuracies)
}

####################
# ALL MODELS
####################

run_logistic_regression <- function(train_data, test_data) {
  model <- glm(num ~ ., data = train_data, family = binomial)
  test_predictions <- predict(model, newdata = test_data, type = "response")
  test_predictions <- ifelse(test_predictions > 0.5, 1, 0)
  return(list(
    model = model,
    accuracy = mean(test_predictions == test_data$num)
  ))
}

run_svm <- function(train_data, test_data, kernel, cost, gamma) {
  if (is.na(gamma)) {
    model <- svm(num ~ ., data = train_data, kernel = kernel, cost = cost, probability = TRUE)
  } else {
    model <- svm(num ~ ., data = train_data, kernel = kernel, cost = cost, gamma = gamma, probability = TRUE)
  }
  
  test_predictions <- predict(model, newdata = test_data)
  test_predictions <- ifelse(test_predictions > 0.5, 1, 0)
  accuracy <- mean(test_predictions == test_data$num)
  
  return(list(
    model = model,
    accuracy = accuracy
  ))
}

####################
# LOAD DATA
####################

heart_data <- get_heart_data(clean_method = 0)

####################
# LOGISTIC REGRESSION
####################

res <- run_cross_validation(heart_data, run_logistic_regression)
mean(res)

####################
# COEFFICIENT EFFECTS IN LOGISTIC REGRESSION
####################

train_index <- createDataPartition(heart_data$num, p = 0.8, list = FALSE)
train_data <- heart_data[train_index, ]
test_data <- heart_data[-train_index, ]
model <- glm(num ~ ., data = train_data, family = binomial)
test_predictions <- predict(model, newdata = test_data, type = "response")
test_predictions <- ifelse(test_predictions > 0.5, 1, 0)

coefficients <- coef(model)
coeff_df <- data.frame(
  variable = names(coefficients),
  coefficient = coefficients
)
ggplot(coeff_df[-1, ], aes(x = reorder(variable, coefficient), y = coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Effect of Predictors on Outcome", x = "Predictor", y = "Coefficient")

####################
# SVM
####################

kernels <- c("linear", "radial", "polynomial")
cost_range <- c(0.01, 0.1, 1, 10)
gamma_range <- c(0.01, 0.1, 1, 10)

results <- list()
for (kernel in kernels) {
  for (cost in cost_range) {
    if (kernel == "linear") {
      accuracies <- run_cross_validation(heart_data, function(train, test) run_svm(train, test, kernel, cost, NA), k=5, repeats=1)
      avg_accuracy <- mean(accuracies)
      results[[paste(kernel, cost, "NA", sep = "_")]] <- avg_accuracy
    } else {
      for (gamma in gamma_range) {
        accuracies <- run_cross_validation(heart_data, function(train, test) run_svm(train, test, kernel, cost, gamma))
        avg_accuracy <- mean(accuracies)
        results[[paste(kernel, cost, gamma, sep = "_")]] <- avg_accuracy
      }
    }
  }
}
results

####################
# VISUALIZING EACH SVM KERNEL USING PCA
####################

predictors <- heart_data[, sapply(heart_data, is.numeric) & colnames(heart_data) != "num"]
target <- heart_data$num

pca_res <- prcomp(predictors, center = TRUE, scale. = TRUE)
pc_data <- data.frame(PC1 = pca_res$x[, 1], PC2 = pca_res$x[, 2], Target = as.factor(target))

for (kernel in kernels) {
  if (kernel == "linear") {
    svm_model <- svm(Target ~ PC1 + PC2, data = pc_data, kernel = kernel, cost = 1, probability = TRUE)
  } else {
    svm_model <- svm(Target ~ PC1 + PC2, data = pc_data, kernel = kernel, cost = 1, gamma = 0.1, probability = TRUE)
  }
  
  x_min <- min(pc_data$PC1) - 0.5
  x_max <- max(pc_data$PC1) + 0.5
  y_min <- min(pc_data$PC2) - 0.5
  y_max <- max(pc_data$PC2) + 0.5
  
  grid <- expand.grid(
    PC1 = seq(x_min, x_max, length.out = 100),
    PC2 = seq(y_min, y_max, length.out = 100)
  )
  grid$Prediction <- predict(svm_model, newdata = grid)
  
  plot <- ggplot(pc_data, aes(x = PC1, y = PC2, color = Target)) +
    geom_point(size = 2, alpha = 0.6) +
    geom_contour(data = grid, aes(x = PC1, y = PC2, z = as.numeric(Prediction)), color = "black", alpha = 0.5) +
    labs(title = "SVM Decision Boundary after Dimension Reduction", x = "PC1", y = "PC2") +
    theme_minimal()
  print(plot)
}

####################
# RANDOM FOREST
####################

run_random_forest <- function(train_data, test_data, ntree, mtry) {
  model <- randomForest(num ~ ., data = train_data, ntree = ntree, mtry = mtry)
  test_predictions <- predict(model, newdata = test_data)
  test_predictions <- ifelse(test_predictions > 0.5, 1, 0)
  accuracy <- mean(test_predictions == test_data$num)
  
  return(list(
    model = model,
    accuracy = accuracy
  ))
}

ntree_values <- c(50, 100, 200, 400)
mtry_values <- c(2, 4, 6, 8)

rf_results <- list()
for (ntree in ntree_values) {
  for (mtry in mtry_values) {
    accuracies <- run_cross_validation(heart_data, function(train, test) run_random_forest(train, test, ntree, mtry), k = 5, repeats = 1)
    avg_accuracy <- mean(accuracies)
    rf_results[[paste("ntree", ntree, "mtry", mtry, sep = "_")]] <- avg_accuracy
  }
}
rf_results

####################
# RANDOM FOREST IMPORTANCE PLOT
####################

rf_model <- randomForest(num ~ ., data = train_data, ntree = 50, mtry = 4)

importance_df <- data.frame(
  Variable = rownames(rf_model$importance),
  Importance = rf_model$importance[, 1]
)

ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(
    title = "Variable Importance in Random Forest",
    x = "Predictors",
    y = "Importance"
  )
