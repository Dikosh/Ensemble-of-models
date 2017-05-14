heart.data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=FALSE,sep=",",na.strings = '?')
names(heart.data) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                        "thalach","exang", "oldpeak","slope", "ca", "thal", "num")
head(heart.data,3)
dim(heart.data)
heart.data$num[heart.data$num > 0] <- 1
barplot(table(heart.data$num),
        main="Fate", col="black")
# change a few predictor variables from integer to factors (make dummies)
chclass <-c("numeric","factor","factor","numeric","numeric","factor","factor","numeric","factor","numeric","factor","factor","factor","factor")
convert.magic <- function(obj,types){
  for (i in 1:length(obj)){
    FUN <- switch(types[i],character = as.character, 
                  numeric = as.numeric, 
                  factor = as.factor)
    obj[,i] <- FUN(obj[,i])
  }
  obj
}

heart.data <- convert.magic(heart.data,chclass)

heart = heart.data #add labels only for plot
levels(heart$num) = c("No disease","Disease")
levels(heart$sex) = c("female","male","")
mosaicplot(heart$sex ~ heart$num,
           main="Fate by Gender", shade=FALSE,color=TRUE,
           xlab="Gender", ylab="Heart disease")
boxplot(heart$age ~ heart$num,
        main="Fate by Age",
        ylab="Age",xlab="Heart disease")
s = sum(is.na(heart.data))
heart.data <- na.omit(heart.data)
#str(heart.data)
library(caret)
set.seed(10)
inTrainRows <- createDataPartition(heart.data$num,p=0.7,list=FALSE)
trainData <- heart.data[inTrainRows,]
testData <-  heart.data[-inTrainRows,]
nrow(trainData)/(nrow(testData)+nrow(trainData)) #checking whether really 70% -> OK

AUC = list()
Accuracy = list()

set.seed(10)
logRegModel <- train(num ~ ., data=trainData, method = 'glm', family = 'binomial')
logRegPrediction <- predict(logRegModel, testData)
logRegPredictionprob <- predict(logRegModel, testData, type='prob')[2]
logRegConfMat <- confusionMatrix(logRegPrediction, testData[,"num"])
#ROC Curve
install.packages("pROC")
library(pROC)
AUC$logReg <- roc(as.numeric(testData$num),as.numeric(as.matrix((logRegPredictionprob))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']  #found names with str(logRegConfMat)  

library(randomForest)
set.seed(10)
RFModel <- randomForest(num ~ .,
                        data=trainData,
                        importance=TRUE,
                        ntree=2000)
#varImpPlot(RFModel)
RFPrediction <- predict(RFModel, testData)
RFPredictionprob = predict(RFModel,testData,type="prob")[, 2]

RFConfMat <- confusionMatrix(RFPrediction, testData[,"num"])

AUC$RF <- roc(as.numeric(testData$num),as.numeric(as.matrix((RFPredictionprob))))$auc
Accuracy$RF <- RFConfMat$overall['Accuracy']  

library(caret)
set.seed(10)
objControl <- trainControl(method='cv', number=10,  repeats = 10)
gbmGrid <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode =10)
# run model
boostModel <- train(num ~ .,data=trainData, method='gbm',
                    trControl=objControl, tuneGrid = gbmGrid, verbose=F)
# See model output in Appendix to get an idea how it selects best model
#trellis.par.set(caretTheme())
#plot(boostModel)
boostPrediction <- predict(boostModel, testData)
boostPredictionprob <- predict(boostModel, testData, type='prob')[2]
boostConfMat <- confusionMatrix(boostPrediction, testData[,"num"])

#ROC Curve
AUC$boost <- roc(as.numeric(testData$num),as.numeric(as.matrix((boostPredictionprob))))$auc
Accuracy$boost <- boostConfMat$overall['Accuracy']  

# for this to work add names to all levels (numbers not allowed)
feature.names=names(heart.data)

for (f in feature.names) {
  if (class(heart.data[[f]])=="factor") {
    levels <- unique(c(heart.data[[f]]))
    heart.data[[f]] <- factor(heart.data[[f]],
                              labels=make.names(levels))
  }
}
set.seed(10)
inTrainRows <- createDataPartition(heart.data$num,p=0.7,list=FALSE)
trainData2 <- heart.data[inTrainRows,]
testData2 <-  heart.data[-inTrainRows,]


fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(10)
gbmModel <- train(num ~ ., data = trainData2,
                  method = "gbm",
                  trControl = fitControl,
                  verbose = FALSE,
                  tuneGrid = gbmGrid,
                  ## Specify which metric to optimize
                  metric = "ROC")
gbmPrediction <- predict(gbmModel, testData2)
gbmPredictionprob <- predict(gbmModel, testData2, type='prob')[2]
gbmConfMat <- confusionMatrix(gbmPrediction, testData2[,"num"])
#ROC Curve
AUC$gbm <- roc(as.numeric(testData2$num),as.numeric(as.matrix((gbmPredictionprob))))$auc
Accuracy$gbm <- gbmConfMat$overall['Accuracy']

set.seed(10)
svmModel <- train(num ~ ., data = trainData2,
                  method = "svmRadial",
                  trControl = fitControl,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
svmPrediction <- predict(svmModel, testData2)
svmPredictionprob <- predict(svmModel, testData2, type='prob')[2]
svmConfMat <- confusionMatrix(svmPrediction, testData2[,"num"])
#ROC Curve
AUC$svm <- roc(as.numeric(testData2$num),as.numeric(as.matrix((svmPredictionprob))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']  

row.names <- names(Accuracy)
col.names <- c("AUC", "Accuracy")
cbind(as.data.frame(matrix(c(AUC,Accuracy),nrow = 5, ncol = 2,
                           dimnames = list(row.names, col.names))
                    
                    summary(logRegModel)$coeff
                    
                    
                    boostImp =varImp(boostModel, scale = FALSE)
                    row = rownames(varImp(boostModel, scale = FALSE)$importance)
                    row = convert.names(row)
                    rownames(boostImp$importance)=row
                    plot(boostImp,main = 'Variable importance for heart failure prediction with boosted tree')
                    
                    logRegConfMat
                    RFConfMat
                    boostConfMat
                    gbmConfMat
                    svmConfMat
                    
                    
                    
