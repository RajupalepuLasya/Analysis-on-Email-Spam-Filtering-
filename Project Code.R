setwd("C:/Users/user/Desktop/SEM7/Data Analytics/Project")

emails = read.csv('emails.csv', stringsAsFactors = FALSE)
str(emails)
table(emails$spam)

library(tm)
corpus = VCorpus(VectorSource(emails$text))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords("en"))
corpus = tm_map(corpus, stemDocument)

dtm = DocumentTermMatrix(corpus)
dtm

spdtm = removeSparseTerms(dtm, 0.95)
spdtm

emailsSparse = as.data.frame(as.matrix(spdtm))
colnames(emailsSparse) = make.names(colnames(emailsSparse))

sort(colSums(emailsSparse))

emailsSparse$spam = emails$spam
sort(colSums(subset(emailsSparse, spam == 0)))

sort(colSums(subset(emailsSparse, spam == 1)))

emailsSparse$spam = as.factor(emailsSparse$spam)
library(caTools)
set.seed(123)
spl = sample.split(emailsSparse$spam, 0.7)
train = subset(emailsSparse, spl == TRUE)
test = subset(emailsSparse, spl == FALSE)


# Build a logistic regression model


spamLog = glm(spam~., data=train, family="binomial")
summary(spamLog)


#Build a CART model

library(rpart)
library(rpart.plot)
spamCART = rpart(spam~., data=train, method="class")




# Build a random forest model


library(randomForest)
set.seed(123)
spamRF = randomForest(spam~., data=train)



#Prediction on training data

predTrainLog = predict(spamLog, type="response")
predTrainCART = predict(spamCART)[,2]
predTrainRF = predict(spamRF, type="prob")[,2] 


# Evaluate the performance of the logistic regression model on training set
table(train$spam, predTrainLog > 0.5)
# training set accuracy of logistic regression
(3052+954)/nrow(train)
# training set AUC of logistic regression
library(ROCR)
predictionTrainLog = prediction(predTrainLog, train$spam)
as.numeric(performance(predictionTrainLog, "auc")@y.values)



# Evaluate the performance of the CART model on training set
table(train$spam, predTrainCART > 0.5)
# training set accuracy of CART
(2885+894)/nrow(train)
# training set AUC of CART
predictionTrainCART = prediction(predTrainCART, train$spam)
as.numeric(performance(predictionTrainCART, "auc")@y.values)



# Evaluate the performance of the random forest model on training set
table(train$spam, predTrainRF > 0.5)
# training set accuracy of random forest
(3013+914)/nrow(train)
# training set AUC of random forest
predictionTrainRF = prediction(predTrainRF, train$spam)
as.numeric(performance(predictionTrainRF, "auc")@y.values)

#In terms of both accuracy and AUC, logistic regression is nearly 
#perfect and outperforms the other two models.



#Prediction on testing data

predTestLog = predict(spamLog, newdata=test, type="response")
predTestCART = predict(spamCART, newdata=test)[,2]
predTestRF = predict(spamRF, newdata=test, type="prob")[,2] 


# Evaluate the performance of the logistic regression model on testing set
table(test$spam, predTestLog > 0.5)
(1257+376)/nrow(test)
predictionTestLog = prediction(predTestLog, test$spam)
as.numeric(performance(predictionTestLog, "auc")@y.values)


# Evaluate the performance of the CART model on testing set
table(test$spam, predTestCART > 0.5)
(1228+386)/nrow(test)
predictionTestCART = prediction(predTestCART, test$spam)
as.numeric(performance(predictionTestCART, "auc")@y.values)



# Evaluate the performance of the random forest model on testing set
table(test$spam, predTestRF > 0.5)
(1290+385)/nrow(test)
predictionTestRF = prediction(predTestRF, test$spam)
as.numeric(performance(predictionTestRF, "auc")@y.values)



#The random forest outperformed logistic regression and CART in both 
#measures, obtaining an impressive AUC of 0.997 on the test set.


