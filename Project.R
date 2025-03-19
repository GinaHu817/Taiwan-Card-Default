rm(list = ls())
library(kknn)
library(ROCR)
library(tree)
library(e1071)
library(rpart)
library(rpart.plot)
library(boot)


#count sample size of each category for variable X6-X11 (uneven sample)
category_counts = table(Default$X6)
print(category_counts)

category_counts1 = table(Default$X7)
print(category_counts1)

category_counts2 = table(Default$X8)
print(category_counts2)

category_counts3 = table(Default$X9)
print(category_counts3)

category_counts4 = table(Default$X10)
print(category_counts4)

category_counts5 = table(Default$X11)
print(category_counts5)

#factor 
Default$Y <- factor(Default$Y, levels = c(0, 1), labels = c("No", "Yes"))
Default$X2 <- factor(Default$X2, levels = c(1, 2), labels = c("male", "female"))
Default$X3 <- factor(Default$X3, levels = c(1, 2, 3, 4), labels = c("graduate school", "university", "high school", "others"))
Default$X4 <- factor(Default$X4, levels = c(1, 2, 3), labels = c("married", "single", "others"))
Default$X6 <- factor(Default$X6, levels = c(-2:9), labels = c("0 balance", "pay duly", "credit revolving facility", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above"))
Default$X7 <- factor(Default$X7, levels = c(-2:9), labels = c("0 balance", "pay duly", "credit revolving facility", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above"))
Default$X8 <- factor(Default$X8, levels = c(-2:9), labels = c("0 balance", "pay duly", "credit revolving facility", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above"))
Default$X9 <- factor(Default$X9, levels = c(-2:9), labels = c("0 balance", "pay duly", "credit revolving facility", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above"))
Default$X10 <- factor(Default$X10, levels = c(-2:9), labels = c("0 balance", "pay duly", "credit revolving facility", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above"))
Default$X11 <- factor(Default$X11, levels = c(-2:9), labels = c("0 balance", "pay duly", "credit revolving facility", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above", "payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above","payment delay for one month and above"))

subset_data <- Default[Default$Y == "Yes", ]
subset_data1 <- Default[Default$Y == "No", ]

#Figure 1
plot(Default$Y, Default$X1,
     xlab = "Default", ylab = "Given credit",
     pch = 19,
     main = "Boxplot of Default vs Given credit")
summary(subset_data$X1)
summary(subset_data1$X1)

#Figure 2
Default$TotalAmount = rowSums(Default[, c("X12", "X13", "X14", "X15", "X16", "X17")])
plot(Default$Y ,Default$TotalAmount,
     xlab = "Default",
     ylab = "Total amount of bill statement",
     col = c("lightblue"),
     main = "Boxplot of Y vs Total Amount of Bill Statements")

#Figure 3
default_x6 = table(Default$Y, Default$X6)
barplot(default_x6, legend = rownames(default_x6),
        beside = TRUE)
summary(subset_data$X6)
summary(subset_data1$X6)
plot(Default$Y, Default$X18,
     xlab = "Default", ylab = "previous payment in September",
     pch = 19,)

#Figure 4
Default$TotalPastPayment = rowSums(Default[, c("X18", "X19", "X20", "X21", "X22", "X23")])
plot(Default$Y ,Default$TotalPastPayment,
     xlab = "Default",
     ylab = "Total Past Payment from April to Sep",
     col = c("lightblue"),
     main = "Boxplot of Y vs Total Amount of Past Payment")


#data split
df=Default

set.seed(678349) 
ntrain=4853 #70/30 split
tr = sample(1:nrow(df),ntrain)  
train = df[tr,] # Training sample
test = df[-tr,] # Testing sample 

#logistic model with all vairables
glm_fit = glm(Y~I(X1/10000)+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+I((X12+X13+X14+X15+X16+X17)/10000)+I((X18+X19+X20+X21+X22+X23)/10000), data = train, family = binomial)
summary(glm_fit)$coefficients
glm_prob = predict(glm_fit, newdata = test, type = "response")
table(glm_prob >= 0.5, test$Y)
pred = prediction(glm_prob, test$Y)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perf = performance(pred, measure = "auc")@y.values[[1]]
plot(perf, lwd = 3, col = "black")
abline(0, 1, lwd = 1, lty = 2) 
text(0.2, 0.9, paste("AUC =", round(auc_perf, 2)))

glm_prob_train = predict(glm_fit, type = "response")
table(glm_prob_train >= 0.5, train$Y)
glm_pred = ifelse(glm_prob_train > 0.5, "Yes", "No")
mean(glm_pred != train$Y) # training error
glm_pred1 = ifelse(glm_prob > 0.5, "Yes", "No")
mean(glm_pred1 != test$Y) # Test error

#native bayes with TotalAmount and TotalPastPayment
nbfit<-naiveBayes(Y~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+TotalAmount+TotalPastPayment, data=train)
nbpred=predict(nbfit, test, type="class")
nbpred2=predict(nbfit, test, type="raw")
table(nbpred, test$Y)
pred3 = prediction(nbpred2[,2], test$Y)
perf3 = performance(pred3, measure = "tpr", x.measure = "fpr")
auc_perf3 = performance(pred3, measure = "auc") # Calculate AUC
plot(perf3, col = "steelblue", lwd = 2) # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.2, 0.8, paste("AUC =", round(auc_perf3@y.values[[1]], 2))) # Compute AUC and add text to ROC plot.
#native bayes with X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23 seperately 
nbfit<-naiveBayes(Y~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23, data=train)
nbpred3=predict(nbfit, test, type="class")
nbpred4=predict(nbfit, test, type="raw")
table(nbpred3, test$Y)
pred4 = prediction(nbpred4[,2], test$Y)
perf4 = performance(pred4, measure = "tpr", x.measure = "fpr")
auc_perf4 = performance(pred4, measure = "auc") # Calculate AUC
plot(perf4, col = "steelblue", lwd = 2) # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.2, 0.8, paste("AUC =", round(auc_perf4@y.values[[1]], 2))) # Compute AUC and add text to ROC plot.

#tree
treeGini = rpart(Y~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+TotalAmount+TotalPastPayment,data=train,method = "class", minsplit = 10, cp = .0001, maxdepth = 30)
bestcp=treeGini$cptable[which.min(treeGini$cptable[,"xerror"]),"CP"]
bestGini = prune(treeGini,cp=bestcp)
plot(bestGini)
text(bestGini,digits=4,bg='lightblue')
treepred1 = predict(bestGini,newdata = test)
pred5 = prediction(treepred1[,2], test$Y)
perf5 = performance(pred5, measure = "tpr", x.measure = "fpr")
auc_perf5 = performance(pred5, measure = "auc") # Calculate AUC
plot(perf5, col = "steelblue", lwd = 2, main="ROC for Tree") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.2, 0.8, paste("AUC =", round(auc_perf5@y.values[[1]], 2))) # Compute AUC and add text to ROC plot.

