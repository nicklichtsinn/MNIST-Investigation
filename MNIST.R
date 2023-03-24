---
author: "Nicholas Lichtsinn"
date: "3/30/2022"
output: html_document
---

```{r echo=FALSE}
library(datasets)
library(class)
library(plyr)
library(dplyr)
library(lattice)
library(caret)
library(e1071)
#install.packages("gmodels")
library(gmodels)
library(randomForest)
library(MASS)
library(rpart)
library(rpart.plot)
#install.packages("proto")
library(proto)
library(readr)
#install.packages("naivebayes")
library(naivebayes)
#install.packages("statsr")
library(statsr)
#install.packages("statsExpressions")
library(statsExpressions)
library(tidyverse)
#install.packages("gbm")
library(gbm)
library(Rcpp)
library(Rtsne)
library(RColorBrewer)
library(rattle)
library(lattice)
library(caret)
library(cluster)
library(class)
library(foreign)
#install.packages("maptree")
library(maptree)
```

```{r}
# load in train and test csv files
train <- read.csv("/Users/nickl/Documents/Syracuse/IST 707 - Data Analytics/digit-train.csv")
test <- read.csv("/Users/nickl/Documents/Syracuse/IST 707 - Data Analytics/digit-test.csv")
```

```{r}
# inspect both files
nrow(train)
nrow(test)

head(train)
head(test)

str(train)
str(test)

dim(train)
dim(test)
```

```{r}
# creating distribution of values for train
hist(train$label, col = 'purple', breaks = seq(from=-0.5, to=9.5, by=1), main = "Train Distribution", xlab = 'Digits')

```

```{r}
# creating samples for train and test sets
sample_train <- train[seq(1, nrow(train), 10), ]
sample_test <- test[seq(1, nrow(train), 10), ]
```

```{r}
# Data Cleaning
# Removing samples with 0 in all images
train_clean <- sample_train[, colSums(sample_train != 0) > 0]

# Removing samples with low variance
all_var <- data.frame(apply(train_clean[-1], 2, var))
colnames(all_var) <- "Variances"
```

```{r}
# Sorting variance and creating number labels
all_var <- all_var[order(all_var$Variances), , drop = FALSE]

# creating number labels
num_lab <- c(1:661)
numbered_var <- cbind(all_var, num_lab)
summary(all_var)

# plotting the variance
plot(all_var$Variances, type = "l", xlab = "Pixel", ylab = "Variance", lwd=2)
abline(h=5000, col = "Purple", led=2)
# zoomed in plot
plot(all_var$Variances, type = "l", xlim = c(0,400), ylim = c(0,800))
abline(h=300, col = "Purple", led=2)

# creating good variance subset
good_var <- subset(all_var, all_var$Variances >= 300, "Variances")
good_var_pixels <- row.names(good_var)

# creating cleaned train dataframe
train_clean <- train_clean[, c("label", good_var_pixels)]

```

```{r}
# Normalizing the data
# creating min/max function
min_max_function <- function(x) {
  a <- (x - min(x))
  b <- (max(x) - min(x))
  return(a / b)
}
# removing label to normalize
train_clean_NL <- train_clean[,-1]
train_clean_NL_norm <- as.data.frame(lapply(train_clean_NL, min_max_function))
train_clean <- cbind(label = train_clean$label, train_clean_NL_norm)

head(train_clean)
```

```{r}
# Creating test and train sets for cleaned training set
train_clean$label <- as.factor(train_clean$label)
set.seed(123)
idx <- sample(1:nrow(train_clean), size = 0.8 * nrow(train_clean))
train_set <- train_clean[idx, ]
test_set <- train_clean[-idx, ]
```

```{r}
# Creating the Decision Tree Model
DT_Model <- rpart(train_set$label ~ . , data = train_set, method = 'class')


# evalutating the model
rsq.rpart(DT_Model)

#plotting the model
rpart.plot(DT_Model)

# predicting test set
predict(DT_Model, test_set, type = 'class')
```
```{r}
# decision tree prediction
#DT_pred <- predict(DT_model, test_set_num, type = 'class')

# creating the confusion matrix
#confusionMatrix(test_set_lab, DT_pred)
```

```{r}
# Building Naive Bayes Model
NB <- naiveBayes(label ~ . , data = train_set, laplace = 1, cost=100, scale=FALSE)
NB_pred <- predict(NB, test_set)

# Creating confusion matrix
confusionMatrix(test_set$label, NB_pred)
```

```{r}
# Creating the KNN Model
# separating train and test into numbers and labels only
train_set_num <- train_set[,-1]
train_set_lab <- train_set[,1]
test_set_num <- test_set[,-1]
test_set_lab <- test_set[,1]
```

```{r}
# choosing k
k <- round(sqrt(nrow(train_clean)))
# building knn model
KNN_Model <- knn(train = train_set_num, test = test_set_num, cl = train_set_lab, k = k, prob = TRUE)
# creating the confusion matrix
confusionMatrix(test_set_lab, KNN_Model)
```

```{r}
# Creating the SVM Model
SVM <- svm(label ~ . , data = train_set, kernel = 'radial', cost=100, scale=FALSE)
# raidal kernel is chosen due to non-linear data and no prior knowledge of data.
print(SVM)

```

```{r}
# predicting the test set
SVM_pred <- predict(SVM, test_set)

# Creating confusion matrix
confusionMatrix(test_set$label, SVM_pred)

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}
# Random Forest Model
# creating fully unpruned decision tree model
RF_model <- randomForest(label ~ . ,data = train_set, ntree = 200)
RF_Predict <- predict(RF_model, test_set)
# creating confusion matrix
confusionMatrix(test_set$label, RF_Predict)
```

```{r}
# plotting the random forest model
plot(RF_model)

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```

```{r}

```
