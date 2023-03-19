
loan <- read.csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv", header=TRUE)

library("skimr")
library("tidyverse")
library("ggplot2")
library("caret")
set.seed(123)

#Image visualization
skim(loan)
par(mfrow=c(1,2))
ggplot(loan, aes(x=Age, y=Income)) + geom_point(aes(color=Personal.Loan))
ggplot(loan, aes(x=Income, y=CCAvg)) + geom_point(aes(color=Personal.Loan))

DataExplorer::plot_bar(loan, ncol = 3)
DataExplorer::plot_histogram(loan, ncol = 3)
DataExplorer::plot_boxplot(loan, by = "Personal.Loan", ncol = 3)

#Data preprocessing
y <- loan$Personal.Loan
loan1 <-loan |>
  filter(Experience >= 0) |>
  select(-ZIP.Code) |>
  mutate(Personal.Loan = as.factor(Personal.Loan))

#########################################################################
#MLR 3
library("mlr3")
library("mlr3learners")
library("data.table")
#loan1$Personal.Loan<- as.factor(loan1$Personal.Loan)
task_loan <- TaskClassif$new(id = "loan",
                             backend = loan1,
                             target = "Personal.Loan",
                             positive = "1")

split_task <- partition(task_loan, ratio=0.8)

task_train <- task_loan$clone()$filter(split_task$train)
task_test <- task_loan$clone()$filter(split_task$test)

#Cross-Validation resampling strategies
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(task_train)
#bootstrap resampling 
boots <- rsmp("bootstrap")
boots$instantiate(task_train)

lrn_cart <- lrn("classif.rpart", predict_type = "prob",cp = 0)
lrn_lr <- lrn("classif.log_reg", predict_type = "prob")
lrn_rf <- lrn("classif.ranger", predict_type = "prob")
lrn_lda <- lrn("classif.lda", predict_type = "prob")
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")

##########################################################################
#bootstrap vs. cv
res_cv<- benchmark(data.table(
  task       = list(task_train),
  learner    = list(lrn_lr,
                    lrn_lda,
                    lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)


res_boot <- benchmark(data.table(
  task       = list(task_train),
  learner    = list(lrn_lr,
                    lrn_lda,
                    lrn_cart),
  resampling = list(boots)
), store_models = TRUE)

rbind(res_cv$aggregate(list(msr("classif.ce"),
                            msr("classif.auc")))[1],res_boot$aggregate(list(msr("classif.ce"),
                                                                            msr("classif.auc")))[1],
      res_cv$aggregate(list(msr("classif.ce"),
                            msr("classif.auc")))[2],res_boot$aggregate(list(msr("classif.ce"),
                                                                            msr("classif.auc")))[2],
      res_cv$aggregate(list(msr("classif.ce"),
                            msr("classif.auc")))[3],res_boot$aggregate(list(msr("classif.ce"),
                                                                            msr("classif.auc")))[3])
##########################################################################
#Pruning the tree
res <- benchmark(data.table(
  task       = list(task_train),
  learner    = list(lrn_lr,
                    lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)

###Trees
trees <- res$resample_result(2)
# first CV iteration
tree1 <- trees$learners[[1]]
# fitted rpart object
tree1_rpart <- tree1$model

plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
res_cart_cv <- resample(task_train, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[3]]$model)
lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.011)

##########################################################################
#Cross validation sampling refitting
res <- benchmark(data.table(
  task       = list(task_train),
  learner    = list(lrn_lr,
                    lrn_rf,
                    lrn_lda,
                    lrn_cart,
                    lrn_cart_cp,
                    lrn_xgboost),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

##########################################################################
#Random forest hyperparameter tuning
library(mlr3tuning)
ps_rf <- ParamSet$new(list(
  ParamInt$new("num.trees", lower = 100, upper = 1000),
  ParamInt$new("mtry", lower = 1, upper = 10),
  ParamInt$new("min.node.size", lower = 1, upper = 20)
))

ctrl_rf_random <- TuningInstanceSingleCrit$new(
  task = task_train,
  learner = lrn_rf,
  resampling = cv5,
  measure = msr("classif.ce"),
  terminator = trm("evals", n_evals = 20),
  search_space = ps_rf
)
tuner = tnr("random_search", batch_size = 10)

# Run tuning
tuner$optimize(ctrl_rf_random)
#as.data.table(ctrl_rf_random$archive)
ctrl_rf_random$result_learner_param_vals
#set the best parameter
lrn_rf$param_set$values <- ctrl_rf_random$result_learner_param_vals
res_rf <- resample(task_train, lrn_rf, cv5, store_models = TRUE)

############################################################################
#model performance
library(pROC)
library(mlr3viz)
res_rf$aggregate(list(msr("classif.ce"),
                      msr("classif.acc"),
                      msr("classif.auc"),
                      msr("classif.fpr"),
                      msr("classif.fnr")))

#roc curve
autoplot(res_rf, type = "roc")+ theme_bw()
#prc curve
autoplot(res_rf, type = "prc")+ theme_bw()
#confusion matrix
res_rf$prediction()$confusion


#calibration curve
library(ggsci)
prediction <- as.data.table(res_rf$prediction())

cv_pred <- lrn_rf$train(task_train)$predict(task_test)

cv_pred_df <- as.data.table(cv_pred)
head(cv_pred_df)

calibration_df <- cv_pred_df %>% 
  mutate(pass = if_else(truth == "1", 1, 0),
         pred_rnd = round(prob.1, 2)
  ) %>% 
  group_by(pred_rnd) %>% 
  summarize(mean_pred = mean(prob.1),
            mean_obs = mean(pass),
            n = n()
  )

ggplot(calibration_df, aes(mean_pred, mean_obs))+ 
  geom_point(aes(size = n), alpha = 0.5)+
  scale_color_lancet()+
  geom_abline(linetype = "dashed")+
  labs(x="Predicted Probability", y= "Observed Probability")+
  theme_minimal()


#learning curve
train_score <- c()
test_score <- c()
for (i in 1:1000){
  learn_loan <- TaskClassif$new(id = "learn1",
                                backend = loan1[1:i,],
                                target = "Personal.Loan")
  
  learn_loan2 <- TaskClassif$new(id = "learn2",
                                 backend = loan1[i+1:1000,],
                                 target = "Personal.Loan")
  
  pred1 <- lrn_rf$train(learn_loan)$predict(learn_loan2)
  train_score[i] <- pred1$score(msr("classif.ce"))
  
  pred2 <- lrn_rf$train(learn_loan2)$predict(learn_loan)
  test_score[i] <- pred2$score(msr("classif.ce"))
}

plot(1:1000, train_score, type = 'l', col = 'red', xlab = 'Number of samples', ylab = 'RMSE')
lines(1:1000, test_score, col = 'blue')
legend('topright', c('Train', 'Test'), col = c('red', 'blue'), lty = 1)











