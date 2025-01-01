### Libraries

library(lubridate)
library(dplyr)
library(stringr)
library(randomForest)
library(pROC)
library(caret)
library(gbm)
library(e1071)
library(ggplot2)

### Dataframes

df1 <- read.csv("C:\\Users\\Nicho\\Desktop\\Superstore.csv") 
df2 <- read.csv("C:\\Users\\Nicho\\Desktop\\credit_card2.csv")

### Execution

df1$Order.Date <- as.Date(df1$Order.Date, format = "%m/%d/%Y")
df1 <- df1 %>% mutate(year = lubridate::year(Order.Date))
transactions_2017 <- df1 %>%
  filter(year == 2017) %>%
  nrow()
print(transactions_2017)

df1 <- df1 %>% mutate(month = lubridate::month(Order.Date))
december_transactions <- df1 %>%
  filter(month == 12) %>%
  nrow()
print(december_transactions)

df1 <- df1 %>%
  mutate(
    year = year(Order.Date),
    month = month(Order.Date),
    year_month = paste(year, month, sep="/")
  )
december_2017 <- df1 %>%
  filter(year_month == "2017/12") %>%
  nrow()
print(december_2017)

unit_price_9 <- df1 %>%
  mutate(Unit.Price.as.character = as.character(Unit.Price),
         Last.Digit = substring(Unit.Price.as.character, nchar(Unit.Price.as.character))) %>%
  filter(Last.Digit == "9") %>%
  nrow()
print(unit_price_9)

kensington <- length(grep("Kensington", df1$Item, ignore.case = TRUE))
print(kensington)

df1 <- df1 %>%
  mutate(
    last.name = word(Customer.Name, -1),
    starts_with_R = str_detect(last.name, "^R")
  )
rows_with_R <- sum(df1$starts_with_R, na.rm = TRUE)
print(rows_with_R)

rows_black_or_blue <- df1 %>%
  mutate(Item = tolower(Item)) %>%
  filter(str_detect(Item, "black|blue")) %>%
  nrow()
print(rows_black_or_blue)

df1 <- df1 %>%
  mutate(Item = tolower(Item),
         Order.Date = gsub("/", "-", Order.Date),
         identifier1 = paste(Item, format(as.Date(Order.Date, "%m/%d/%Y"), "%m-%d-%Y"), sep="-"))
num_unique_identifiers <- length(unique(df1$identifier1))
print(num_unique_identifiers)

df1 <- df1 %>%
  mutate(
    Item = tolower(Item),
    City = tolower(City),
    State = tolower(State),
    Order.Date = as.Date(Order.Date),
    Formatted.Date = format(Order.Date, "%m-%d-%Y"),
    identifier2 = paste(Item, Formatted.Date, City, State, sep="-")
  )
df1$identifier2 <- gsub("-(?=[^-]*$)", ",", df1$identifier2, perl = TRUE)
num_unique_identifiers2 <- length(unique(df1$identifier2))
print(num_unique_identifiers2)

###############################################


set.seed(100)

df2$card <- as.factor(df2$card)

set.seed(100)
train_indices <- sample(nrow(df2), size = 0.7 * nrow(df2))
train_data <- df2[train_indices, ]
test_data <- df2[-train_indices, ]

x = seq(20, 200, by = 20)
auc_values = vector("numeric", length = length(x))

for (i in seq_along(x)) {
  rf_model <- randomForest(card ~ ., data = train_data, ntree = x[i], maxnodes = 10)
  
  predictions <- predict(rf_model, newdata = test_data, type = "prob")
  roc_result <- roc(response = as.numeric(test_data$card) - 1, predictor = predictions[,2])
  
  auc_values[i] <- roc_result$auc
}
data = data.frame(Trees = x, AUC = auc_values)
ggplot(data, aes(x = Trees, y = AUC)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  ggtitle("AUC vs. Number of Trees") +
  xlab("Number of Trees") +
  ylab("AUC")

############################################################################


fitControl <- trainControl(method="repeatedcv", number=5, repeats=5, summaryFunction=twoClassSummary, classProbs=TRUE, savePredictions=TRUE)
boosted_glm_model <- train(card ~ ., data=train_data, method="gbm", trControl=fitControl, metric="F1", verbose=FALSE)
predictions <- predict(boosted_glm_model, newdata= test_data)
confusionMatrix <- confusionMatrix(predictions, test_data$card)
precision <- confusionMatrix$byClass['Pos Pred Value']
recall <- confusionMatrix$byClass['Sensitivity']
f1_score <- 2 * ((precision * recall) / (precision + recall))
print(round(f1_score, 3))


############################################################################


set.seed(100)

x = seq(0.1, 0.9, by = 0.1)
auc_values = numeric(length(x))

for (i in seq_along(x)) {
  index <- createDataPartition(df2$card, p=x[i], list=FALSE)
  train_data <- df2[index, ]
  test_data <- df2[-index, ]
  
  fitControl <- trainControl(method = 'repeatedcv', number = 5, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
  
  boosted_glm_model <- train(card ~ ., data = train_data, method = 'gbm', trControl = fitControl, metric = "ROC")
  
  predictions <- predict(boosted_glm_model, newdata = test_data, type = "prob")
  roc_result <- roc(response = test_data$card, predictor = predictions[,2])
  
  auc_values[i] <- roc_result$auc
}

data = data.frame(Proportion = x, AUC = auc_values)

ggplot(data, aes(x = Proportion, y = AUC)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  ggtitle("AUC vs. Proportion of Training Data") +
  xlab("Proportion of Training Data") +
  ylab("AUC")

ggsave("auc_vs_training_proportion.png", width = 8, height = 6)


#############################################################################################


set.seed(100)
df2$card <- ifelse(df2$card == "yes", 1, 0)
data_a <- df2[c("card", "age", "income")]
data_b <- df2[c("card", "age", "income", "months")]
data_c <- df2[c("card", "age", "income", "expenditure", "months")]
train_svm <- function(data) {
  index <- sample(1:nrow(data), 0.7 * nrow(data))
  svm_model <- svm(card ~ ., data= train_data, kernel= "radial", probability= TRUE)
  predictions <- predict(svm_model, test_data, probability= TRUE)
  probs <- attr(predictions, "probabilities")[,2]
  auc_score <- auc(test_data$card, probs)
  return(auc_score)
}
auc_a <- train_svm(data_a)
auc_b <- train_svm(data_b)
auc_c <- train_svm(data_c)
print(paste("AUC for age + income:", round(auc_a, 3)))
print(paste("AUC for age + income + months:", round(auc_b, 3)))
print(paste("AUC for age + income + expenditure + months:", round(auc_c, 3)))
max_auc <- max(auc_a, auc_b, auc_c)
if(max_auc == auc_a) {
  print("Highest AUC generated by combination: age + income")
} else if(max_auc == auc_b) {
  print("Highest AUC generated by combination: age + income + months")
} else {
  print("Highest AUC generated by combination: age + income + expenditure + months")
}



