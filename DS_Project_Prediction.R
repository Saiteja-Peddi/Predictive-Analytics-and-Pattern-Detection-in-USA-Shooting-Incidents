
#install.packages("arulesViz")
#install.packages("tidyverse")
#install.packages("readxml")
#install.packages("knitr")
#install.packages("RColorBrewer")
library(knitr)
library(ggplot2)
library(plyr)
library(dplyr)
library(RColorBrewer)
library(readxl)
library(tidyverse)
library(arulesViz)
library(mlbench)
library(e1071)
library(caret)

setwd("/Users/peddisaiteja/Desktop/UMBC/Course Work/Sem 1/Intro to Datascience/Group Project")
sampleShootings <- data.frame(read.csv('shootings.csv'))
shootings <- data.frame(read.csv('shootings.csv'))

head(shootings)
#Removing rows with missing values and NONE items
shootings <- shootings[complete.cases(shootings), ]

# Data Preparation
shootings$Armed <- shootings$arms_category
shootings$FleeType <- shootings$flee
shootings$FleeType[shootings$FleeType != 'Not fleeing'] <- 'Fleeing'
shootings$Armed[shootings$Armed != 'Unarmed'] <- 'Armed'
shootings$gender[shootings$gender == 'M'] <- 'Male'
shootings$gender[shootings$gender == 'F'] <- 'Female'
shootings$AgeGroup <- shootings$age
shootings$AgeGroup[shootings$AgeGroup < 40] <- '<40'
shootings$AgeGroup[shootings$AgeGroup != '<40'] <- '>40'
shootings$signs_of_mental_illness[shootings$signs_of_mental_illness == TRUE] <- 'M-Ill'
shootings$signs_of_mental_illness[shootings$signs_of_mental_illness == FALSE] <- 'No M-Ill'
shootings$body_camera[shootings$body_camera == TRUE] <- 'BCam'
shootings$body_camera[shootings$body_camera == FALSE] <- 'No BCam'

shootings <- subset(shootings, select = -c(id,name,date,armed,arms_category,age,flee) )

#Tables of individual columns
table(shootings$AgeGroup)
table(shootings$gender)
table(shootings$race)
table(shootings$armed)


#Factorising the columns
shootings$manner_of_death <- as.factor(shootings$manner_of_death)
shootings$Armed <- as.factor(shootings$Armed)
shootings$AgeGroup <- as.factor(shootings$AgeGroup)
shootings$gender <- as.factor(shootings$gender)
shootings$race <- as.factor(shootings$race)
shootings$city <- as.factor(shootings$city)
shootings$state <- as.factor(shootings$state)
shootings$signs_of_mental_illness <- as.factor(shootings$signs_of_mental_illness)
shootings$threat_level <- as.factor(shootings$threat_level)
shootings$FleeType <- as.factor(shootings$FleeType)
shootings$body_camera <- as.factor(shootings$body_camera)



################################################################################
#######################Association analysis#####################################
################################################################################


#combining items of single transaction with comma separated

shootings$Combined <- paste(
  shootings$Armed,
  shootings$race,
  shootings$signs_of_mental_illness,
  shootings$threat_level,
  shootings$FleeType,
  shootings$body_camera,
  shootings$AgeGroup,
  sep = ",")
shootings

#Creating a csv file with only items
shootings$manner_of_death <- NULL
shootings$Armed <- NULL
shootings$AgeGroup <- NULL
shootings$gender <- NULL
shootings$race <- NULL
shootings$city <- NULL
shootings$state <- NULL
shootings$signs_of_mental_illness <- NULL
shootings$threat_level <- NULL
shootings$flee <- NULL
shootings$body_camera <- NULL
colnames(shootings) <- c("Combined")
head(shootings)
write.csv(shootings,"shootingsDataTransactions.csv", quote = FALSE, row.names = FALSE)
shootingsTransactions <- read.transactions("shootingsDataTransactions.csv", format = 'basket', sep=',')
head(shootingsTransactions)

#Descriptive statistics of Bread Basket Transactions
summary(shootingsTransactions)


#Creating top 5 association rules.
association.rules <- apriori(shootingsTransactions, parameter = list(supp=0.1, conf=0.3,maxlen=10, minlen = 4))
inspect(association.rules)



################################################################################
#######################Naive Bayes Classification###############################
################################################################################

##################Predicting whether person is Armed or not(Partitioned Data set)##########################

## 75% of the sample size

shootingsWithArmed <- shootings[shootings$Armed == 'Armed',]
shootingsWithUnarmed <- shootings[shootings$Armed == 'Unarmed',]
smpSizeArmed <- floor(0.75 * nrow(shootingsWithArmed))
smpSizeUnarmed <- floor(0.75 * nrow(shootingsWithUnarmed))
typeof(shootingsWithArmed)

## set the seed to make your partition reproducible
set.seed(6)
train_ind_arm <- sample(seq_len(nrow(shootingsWithArmed)), size = smpSizeArmed)
train_ind_unarm <- sample(seq_len(nrow(shootingsWithUnarmed)), size = smpSizeUnarmed)
train <- rbind(shootingsWithArmed[train_ind_arm, ],shootingsWithUnarmed[train_ind_unarm, ] )
test <- rbind(shootingsWithArmed[-train_ind_arm, ],shootingsWithUnarmed[-train_ind_unarm, ] )

NVmodel <- naiveBayes(Armed ~ signs_of_mental_illness + AgeGroup + city,
                      data = train)
preds <- predict(NVmodel, newdata = test)
conf_matrix <- table(preds, test$Armed)

conf_matrix
table(shootings$Armed)

confusionMatrix(conf_matrix)

#NVmodel
NVmodel$tables



##################Predicting person Age group (Partitioned Data set)##########################

## 75% of the sample size

shootingsLess40 <- shootings[shootings$AgeGroup == '<40',]
shootingsLess99 <- shootings[shootings$AgeGroup == '>40',]
smpSizeLess40 <- floor(0.75 * nrow(shootingsLess40))
smpSizeLess99 <- floor(0.75 * nrow(shootingsLess99))


## set the seed to make your partition reproducible
set.seed(8)
train_ind_less40 <- sample(seq_len(nrow(shootingsLess40)), size = smpSizeLess40)
train_ind_less99 <- sample(seq_len(nrow(shootingsLess99)), size = smpSizeLess99)
train <- rbind(shootingsLess40[smpSizeLess40, ],shootingsLess99[smpSizeLess99, ] )
test <- rbind(shootingsLess40[-smpSizeLess40, ],shootingsLess99[-smpSizeLess99, ] )


NVmodel <- naiveBayes(AgeGroup ~ signs_of_mental_illness + Armed + city + state + FleeType,
                      data = train)
NVmodel
preds <- predict(NVmodel, newdata = test)
conf_matrix <- table(preds, test$AgeGroup)

conf_matrix
table(shootings$AgeGroup)

confusionMatrix(conf_matrix)


#NVmodel
NVmodel$tables


# The below generated models doesn't work as above as they have shown accuracy without
# showing any prediction in False Positive and False Negative columns

#############Predicting whether person is Armed or not##########################

## 75% of the sample size

smp_size <- floor(0.75 * nrow(shootings))

## set the seed to make your partition reproducible
set.seed(4)
train_ind <- sample(seq_len(nrow(shootings)), size = smp_size)

train <- shootings[train_ind, ]
test <- shootings[-train_ind, ]

NVmodel <- naiveBayes(Armed ~ FleeType + signs_of_mental_illness + AgeGroup,
                      data = train)
preds <- predict(NVmodel, newdata = test)
conf_matrix <- table(preds, test$Armed)

conf_matrix
confusionMatrix(conf_matrix)

#NVmodel
NVmodel$tables

#############Predicting person Age group########################################
## 75% of the sample size
smp_size <- floor(0.75 * nrow(shootings))

## set the seed to make your partition reproducible
set.seed(8)
train_ind <- sample(seq_len(nrow(shootings)), size = smp_size)

train <- shootings[train_ind, ]
test <- shootings[-train_ind, ]

NVmodel <- naiveBayes(AgeGroup ~ threat_level + signs_of_mental_illness + FleeType + Armed,
                      data = train)
preds <- predict(NVmodel, newdata = test)
conf_matrix <- table(preds, test$AgeGroup)

conf_matrix
confusionMatrix(conf_matrix)

#NVmodel
NVmodel$tables


