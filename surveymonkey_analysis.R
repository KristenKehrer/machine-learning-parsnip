# SURVEY MONKEY DATA ----
install.packages("correlationfunnel")
install.packages("digest")

# 1.0 LIBRARIES ----

# Machine Learning
library(parsnip)    # Taught in 101   # Tidy machine learning
library(rsample)    # Taught in 101   # testing/training splits
library(recipes)    # Taught in 201*  # preprocessing
library(yardstick)  # Taught in 101   #  ROC/AUC
library(xgboost)    # Taught in 101
library(Ckmeans.1d.dp)  ## feature importance plot
library(rpart)      # decision tree
library(rpart.plot) # decision tree plot
library(digest)  # dependency

# EDA
library(correlationfunnel)  # New R Package - Teach how to make in 201

# Core
library(tidyverse)  # Taught in 101 (Foundations) & 201 (Advanced)


# 2.0 DATA ----

survey_data_tbl <- read_csv("[your path]/long_mods.csv")

survey_data_tbl %>% glimpse()

survey_data_tbl %>%
    select(-id) %>%
    binarize(n_bins = 4, thresh_infreq = 0.01) %>%
    correlate(changed_company_balance__Yes_changed_companies) %>%
    plot_correlation_funnel()


# 3.0 DATA PREPARATION ----
set.seed(222)
split_rsample <- survey_data_tbl %>%
    mutate(changed_company_yes = str_detect(changed_company_balance, "Yes") %>% as.numeric()) %>%
    select(changed_company_yes, everything()) %>%
    rsample::initial_split(prop = 0.80)

train_tbl <- rsample::training(split_rsample)    
test_tbl  <- rsample::testing(split_rsample)

train_tbl %>% glimpse()

preprocessing_pipeline <- recipe(changed_company_yes ~ ., data = train_tbl) %>%
  # removing id vars  
    step_rm(changed_company_balance, id) %>%
  # near zero-variance (i.e. constant) remove them.
    step_nzv(all_predictors()) %>%
  # Anything that is character (nominal) -> factor
    step_string2factor(all_nominal()) %>%
  # If a category in a factor variable is less than 3% of the distn
  # group it in an "other" category
    step_other(all_nominal(), threshold = 0.03, other = "MISC") %>%
  # get dummies
    step_dummy(all_nominal(), one_hot = TRUE) %>%
  # remove dummies with only a few non-zeros
    step_nzv(all_predictors()) %>%
  # Binary variable converting to a factor (may not be necessary)
    step_num2factor(changed_company_yes) %>%
    prep()

## The bake function actually performs the processing pipeline
train_processed_tbl <- bake(preprocessing_pipeline, train_tbl)
test_processed_tbl  <- bake(preprocessing_pipeline, test_tbl)

train_processed_tbl %>% glimpse()
test_processed_tbl %>% glimpse()


# 4.0 MACHINE LEARNING ----
set.seed(123)
model_xgb <- boost_tree(
           mode = "classification", 
           mtry = 80, 
           trees = 1000, 
           min_n = 3, 
           tree_depth = 8, 
           learn_rate = 0.01, 
           loss_reduction = 0.01) %>%
  # runs xgboost under the hood
    set_engine("xgboost") %>%
  # model_spec is an object.  fit similar to bake, in the sense
  # That it is applying the fit to the training data
    fit.model_spec(changed_company_yes ~ ., data = train_processed_tbl)



predictions_test_tbl <- model_xgb %>%
    predict.model_fit(new_data = test_processed_tbl, type = "prob") %>%
    bind_cols(test_tbl) %>%
    mutate(changed_company_yes = as.factor(changed_company_yes))


predictions_test_tbl[,1:4] %>% glimpse()

# 6.0 FEATURE IMPORTANCE -----

xgb.importance(model = model_xgb$fit) %>%
  xgb.ggplot.importance()




#########  Decision tree 
fit <- rpart(changed_company_yes ~ ., method = "class", data = train_processed_tbl)

print(fit)
printcp(fit)

###  prune tree
pfit<-prune(fit, cp=0.015)




rpart.plot(pfit, extra=104, 
           main="Changing Employer Tree - pruned")


# two_class_example %>% glimpse()

# 5.0 METRICS ----

predictions_test_tbl %>% 
    roc_auc(changed_company_yes, .pred_0)

# two_class_example %>% roc_curve(truth, Class1)

predictions_test_tbl %>%
    roc_curve(changed_company_yes, .pred_0) %>%
    ggplot(aes(x = 1-specificity, y = sensitivity)) +
    geom_path() +
    geom_point() +
    labs(title = "ROC Curve")



# 6.0 FEATURE IMPORTANCE -----

xgb.importance(model = model_xgb$fit) %>%
    xgb.ggplot.importance()

# ADVANCED TECHNIQUES - DS4B 201-R
# - H2O AutoML - 5-Fold Cross Validation
# - LIME - Local Feature Importance

# WEB APPS - SHINY DS4B-102-R
