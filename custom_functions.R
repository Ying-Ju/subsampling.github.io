# finding total number of numeric variables (including integers)
nvars <- function(x){
  A <- data.frame(numeric=0, factor=0, integer=0)
  temp <- table(sapply(x, class))
  A[names(temp)] <- temp
  A["numeric"] <- A$numeric + A$integer
  A$integer <- NULL
  A$numeric <- as.numeric(A$numeric)
  return(A)
}

# Suppressing output from cat() by Hadeley Wickham per
# https://r.789695.n4.nabble.com/Suppressing-output-e-g-from-cat-tp859876p859882.html
quiet = function(x) { 
  sink(tempfile()) 
  on.exit(sink()) 
  invisible(force(x)) 
} 


# Unzipping and reading from csv
unzip_read = function(fileName, ext = ".dat"){
  # read all file names in the subfolder
  path = paste0(here::here("Data/unzipDatasets"), "/" )
  all_files = gsub(path, "", as.character(unzip(fileName, exdir=here::here("Data/unzipDatasets"))))[1]
  
  # find the .dat data file
  find_ext = str_detect(all_files, ext)
  
  # the following code needs to modify if there are more than one dataset in the folder
  if (length(find_ext[find_ext==TRUE])==1){
    # find data and names for predictors
    filetext = readLines(unz(fileName,  all_files[find_ext==TRUE]))
    unlink(unz(fileName,  all_files[find_ext==TRUE]))
    lines2Skip = which(str_detect(filetext, "@data"))
    find_predictor_names = str_remove_all(filetext[str_detect(filetext, "@inputs|@input")], "@inputs |@input ")
    find_predictor_names = str_replace_all(find_predictor_names, " ", "")
    predictor_names = unlist(str_split(find_predictor_names, ","))
    predictor_names = predictor_names[predictor_names != ""]
    
    # read the dataset
    df = read.csv(unz(fileName,  all_files[find_ext==TRUE]), 
                     skip = lines2Skip, header=F, stringsAsFactors = T)
    colnames(df) = c(predictor_names, "Class")
    df %<>% mutate_if(is.factor, ~factor(trimws(.)))
    return(df)
  }else{
    return(NA)
  }
}


# Generating a list of data.frames containing all keelDatasets
keelData = function(kDataFullLinks, datasetNames, numAttributes, dest = here::here("Data/imbalancedDatasets")){
    # Downloading the data into 
  Map(function(u, d) download.file(u, d, mode="wb"), 
      kDataFullLinks, paste0(dest,"/" , datasetNames, ".zip") )

  subfolder_names = list.files(path = dest, pattern = ".zip", full.names = T)
  
  List_of_dfs = list()
  for (counter in 1:length(subfolder_names)) {
    List_of_dfs[[counter]] = unzip_read(subfolder_names[counter])
    df_name = gsub(".zip", "", gsub(".*/", "", subfolder_names[counter]))
    names(List_of_dfs)[counter] = df_name
  }
  return(List_of_dfs)
}


# Creating a function for applying multiple machine learning models and
# obtaining results from 5 fold cross validation

mainfunction <- function(i, dataset, cases){
  trainData = dataset$data[[cases$ID[i]]]
  best_metric <- cases$metrics[i]
  subsampling <- cases$subsampling[i]
  if (subsampling=="NULL") subsampling <- NULL
  
  # * Required Packages and Custom Functions --------------------------------
  
  # * * Packages ------------------------------------------------------------
  pacman::p_load(tidyverse, magrittr, janitor,
                 caret, caretEnsemble, recipes, yardstick,
                 rpart, glmnet, nnet, PRROC, conflicted)
  
  # * * Custom Functions ----------------------------------------------------
  
  get_all <- function(model){
    Accuracy <- model$pred %>% group_by(Resample) %>% yardstick::accuracy(obs, pred)
    Sensitivity <- model$pred %>% group_by(Resample) %>% yardstick::sens(obs, pred)
    Specificity <- model$pred %>% group_by(Resample) %>% yardstick::spec(obs, pred)
    AUROC <- model$pred %>% group_by(Resample) %>% yardstick::roc_auc(obs, positive)
    Precision <- model$pred %>% group_by(Resample) %>% yardstick::precision(obs, pred)
    F1 <- model$pred %>% group_by(Resample) %>% yardstick::f_meas(obs, pred)
    AUPRC <- model$pred %>% group_by(Resample) %>% yardstick::pr_auc(obs, positive)
    
    Resample_result <- cbind.data.frame(Resample = Sensitivity$Resample, Accuracy = Accuracy$.estimate,
                                        Sensitivity = Sensitivity$.estimate,
                                        Specificity = Specificity$.estimate,  
                                        AUROC = AUROC$.estimate, 
                                        Precision = Precision$.estimate, F1 = F1$.estimate,
                                        AUPRC = AUPRC$.estimate)
    colnames(Resample_result)[-1] <- paste0(model$method, "~", colnames(Resample_result)[-1])
    return(Resample_result)
  }
  
  prSummary0 <- function (data, lev = NULL, model = NULL) {
    caret:::requireNamespaceQuietStop("yardstick")
    if (length(levels(data$obs)) > 2) 
      stop(paste("Your outcome has", length(levels(data$obs)), 
                 "levels. `prSummary0`` function isn't appropriate.", 
                 call. = FALSE))
    if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
      stop("Levels of observed and predicted data do not match.", 
           call. = FALSE)
    auprc <- try(yardstick::pr_auc(data, obs, lev[1]), silent = TRUE)
    
    
    if (inherits(auprc, "try-error")){
      auprc <- NA
    }else{
      auprc <- auprc$.estimate
    }
    
    return(c(AUC = auprc, Precision = caret:::precision.default(data = data$pred, 
                                                                reference = data$obs, relevant = lev[1]), 
             Recall = caret:::recall.default(data = data$pred, reference = data$obs, relevant = lev[1]), 
             F = caret:::F_meas.default(data = data$pred, reference = data$obs, relevant = lev[1])))
  }
  
  trainData %<>% mutate_if(is.character, as.factor) %>% 
    clean_names() %>% 
    mutate(class = relevel(class, ref = "positive"))
  
  #table(trainData$class) %>% prop.table()
  
  # * Data Preprocessing ----------------------------------------------------
  recSteps = recipe(class ~ ., data = trainData) %>% 
    step_nzv( all_predictors() ) %>% 
    step_YeoJohnson(all_numeric()) %>%
    step_normalize( all_numeric() ) %>% 
    step_dummy(all_nominal(), -all_outcomes() ) # create dummy variables for categorical predictors
  
  if (best_metric=="Accuracy") outputFun = defaultSummary
  if (best_metric=="ROC") outputFun = twoClassSummary
  if (best_metric=="AUC") outputFun = prSummary0
  
  if (cases$subsampling[i]=="NULL") set.seed((11111+cases$ID[i]))
  if (cases$subsampling[i]=="down") set.seed((22222+cases$ID[i]))
  if (cases$subsampling[i]=="up") set.seed((33333+cases$ID[i]))
  if (cases$subsampling[i]=="smote") set.seed((44444+cases$ID[i]))
  
  seeds <- vector(mode = "list", length = 5)
  for(j in 1:5) seeds[[j]] <- sample.int(nrow(trainData), 1)
  
  ## For the last model:
  seeds[[6]] <- sample.int(nrow(trainData), 1)
  
  # * Cross Validation Setup ------------------------------------------------
  
  fitControl = trainControl(
    method = "cv", # k-fold cross validation
    number = 5, # Number of Folds
    search = "grid", # Default grid search for parameter tuning
    sampling = subsampling, # If none, please set to NULL
    summaryFunction = outputFun, # see custom function in this file
    classProbs = T, # should class probabilities be returned
    selectionFunction = "best", # best fold
    savePredictions = 'final',
    index = createResample(trainData$class, 5), seeds = seeds)
  
  
  models = caretList(recSteps, data = trainData, metric=best_metric, 
                     tuneList= list(
                       cart = caretModelSpec(method="rpart", tuneLength=30),
                       glm = caretModelSpec(method = "glm", family= "binomial", maxit = 100)
                     ),  
                     trControl=fitControl, continue_on_fail = T
  )
  
  # How to extract the cross validation Results ----
  temp = cbind(get_all(models$cart), get_all(models$glm)[,-1])
  temp$subsampling = toupper(cases$subsampling[i]) # needs to be automatically read from the input to fitControl
  temp$metric = best_metric # needs to be automatically read for the input to caretList
  temp$dataset = dataset$dataset[cases$ID[i]] # needs to be automatically read for the dataset name
  temp$imbalance_ratio = dataset$imbalance_ratio[cases$ID[i]]
  
  temp2 = temp %>% pivot_longer(cols = contains('~'))
  temp2$method = str_detect(temp2$name, 'glm')
  temp2$method = ifelse(temp2$method == 'TRUE', 'GLM', 'CART')
  temp2$measure = str_replace(temp2$name, '.*~', '')
  temp2 %<>% select(dataset, imbalance_ratio, subsampling, metric, method, Resample, measure, value)
  
  return(temp2)
}


# allinone function returns either the performance measures (index=0) based on the testdata 
# given a method (either "glm" or "rpart") or the prediction result (index=1).
# This is our custom function for a list of datasets and a test dataset. 

allinone <- function(i, datasets, testdata, cases, IR, method, index=0){
  trainData = datasets[[cases$ID[i]]]
  best_metric <- cases$metrics[i]
  subsampling <- cases$subsampling[i]
  
  if (!(method %in% c("glm", "rpart"))) return("Error: This algorithm only works with 'glm' or 'rpart'.")
  if (!(subsampling %in% c("NULL", "down", "up", "smote"))) return("Error: This algorithm only works with subsampling strategies: 'NULL', 'down', 'up', 'smote'.")
  
  
  if (subsampling=="NULL") subsampling <- NULL
  
  # * Required Packages and Custom Functions --------------------------------
  
  # * * Packages ------------------------------------------------------------
  # remotes::install_github("cran/DMwR")
  pacman::p_load(tidyverse, magrittr, janitor, 
                 caret, caretEnsemble, recipes, yardstick,
                 rpart, glmnet, nnet, PRROC, conflicted)
  
  
  # * Data Preprocessing ----------------------------------------------------
  recSteps = recipe(State ~ ., data = trainData) %>% 
    step_nzv( all_predictors() ) %>% 
    step_YeoJohnson(all_numeric()) %>%
    step_normalize( all_numeric() )
  
  
  if (best_metric=="Accuracy") outputFun = defaultSummary
  if (best_metric=="ROC") outputFun = twoClassSummary
  
  if (cases$subsampling[i]=="NULL") set.seed((11111+cases$ID[i]))
  if (cases$subsampling[i]=="down") set.seed((22222+cases$ID[i]))
  if (cases$subsampling[i]=="up") set.seed((33333+cases$ID[i]))
  if (cases$subsampling[i]=="smote") set.seed((44444+cases$ID[i]))
  
  seeds <- vector(mode = "list", length = 5)
  for(j in 1:5) seeds[[j]] <- sample.int(nrow(trainData), 100)
  
  ## For the last model:
  seeds[[6]] <- sample.int(nrow(trainData), 1)
  
  # * Cross Validation Setup ------------------------------------------------
  
  fitControl = trainControl(
    method = "cv", # k-fold cross validation
    number = 5, # Number of Folds
    search = "grid", # Default grid search for parameter tuning
    sampling = subsampling, # If none, please set to NULL
    summaryFunction = outputFun, # see custom function in this file
    classProbs = T, # should class probabilities be returned
    selectionFunction = "best", # best fold
    savePredictions = 'final',
    index = createResample(trainData$State, 5), seeds = seeds)
  
  # Model Training ----------------------------------------------------------
  
  if (method == "glm"){
    model = train(recSteps, data = trainData, metric=best_metric, 
                  method="glm", family= "binomial", maxit = 100,
                  trControl=fitControl
    )
  }
  
  if (method == "rpart"){
    model = train(recSteps, data = trainData, metric=best_metric, 
                  method="rpart", tuneLength = 30,
                  trControl=fitControl
    )
  }

  pred <- predict(model, testdata)
  diseased <- predict(model, testdata, type = "response")$diseased
  obs <- testdata$State
  result <- data.frame(obs, pred, diseased)
  
  if (index==1) return(result)
  
  Accuracy <- yardstick::accuracy(result, obs, pred)
  Sensitivity <- yardstick::sens(result, obs, pred)
  Specificity <- yardstick::spec(result, obs, pred)
  AUROC <- yardstick::roc_auc(result, obs, diseased)
  Precision <- yardstick::precision(result, obs, pred)
  F1 <- yardstick::f_meas(result, obs, pred)
  AUPRC <- yardstick::pr_auc(result, obs, diseased)
  
  final_result <- data.frame(cases[i,1:2], IR = IR[cases$ID[i]] , Accuracy = Accuracy$.estimate,
                             Sensitivity = Sensitivity$.estimate,
                             Specificity = Specificity$.estimate,  
                             AUROC = AUROC$.estimate, 
                             Precision = Precision$.estimate, F1 = F1$.estimate,
                             AUPRC = AUPRC$.estimate)
  return(final_result)
}


# The modeling() function requires at least three arguments:
# trainData: a dataset that is used for training a giving model
# testData: a dataset that is used for testing the performance of the model
# response: a string, name of the binary response variable
# subsampling: subsampling method (it takes one of: "NULL", "down", "up", "smote")
# best_metric: the metric used to evaluate the performance of the model ("Accuracy" or "ROC", "AUC")
# method: either "glm" or "rpart" (logistic regression or decision tree)
# index = 0: returns the performance measures based on the test dataset
# index = 1: returns the prediction result based on the test dataset
# index = 2: returns both the preformance measures and prediction results based on the test dataset
# seed: random seed 

modeling <- function(trainData, testData, response, subsampling, best_metric, method, index=0, seed=2021){
  if (subsampling=="NULL") subsampling <- NULL
  
  # Required Packages and Custom Functions 
  # if DMwR is not available, use the following line to install it.
  # remotes::install_github("cran/DMwR")
  pacman::p_load(tidyverse, magrittr, janitor, 
                 caret, recipes, yardstick,
                 rpart, glmnet, nnet, PRROC, conflicted)
  
  if (!(method %in% c("glm", "rpart"))) return("Error: This algorithm only works with 'glm' or 'rpart'.")
  if (!(subsampling %in% c("NULL", "down", "up", "smote"))) return("Error: This algorithm only works with subsampling strategies: 'NULL', 'down', 'up', 'smote'.")

  # custom function to obtain the area under the precision-recall curve
  prSummary0 <- function (data, lev = NULL, model = NULL) {
    caret:::requireNamespaceQuietStop("yardstick")
    if (length(levels(data$obs)) > 2) 
      stop(paste("Your outcome has", length(levels(data$obs)), 
                 "levels. `prSummary0`` function isn't appropriate.", 
                 call. = FALSE))
    if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
      stop("Levels of observed and predicted data do not match.", 
           call. = FALSE)
    auprc <- try(yardstick::pr_auc(data, obs, lev[1]), silent = TRUE)
    
    
    if (inherits(auprc, "try-error")){
      auprc <- NA
    }else{
      auprc <- auprc$.estimate
    }
    
    return(c(AUC = auprc, Precision = caret:::precision.default(data = data$pred, 
                                                                reference = data$obs, relevant = lev[1]), 
             Recall = caret:::recall.default(data = data$pred, reference = data$obs, relevant = lev[1]), 
             F = caret:::F_meas.default(data = data$pred, reference = data$obs, relevant = lev[1])))
  }
  
  i1 <- which(colnames(trainData)==response)
  i2 <- which(colnames(testData)==response)
  colnames(trainData)[i1] <- "class"
  colnames(testData)[i2] <- "class"
  
  trainData %<>% mutate_if(is.character, as.factor) %>% 
    mutate(class = relevel(class, ref = levels(trainData$class)[1]))
  

  # * Data Preprocessing ----------------------------------------------------
  recSteps = recipe(class ~ ., data = trainData) %>% 
    step_nzv( all_predictors() ) %>% 
    step_YeoJohnson(all_numeric()) %>%
    step_normalize( all_numeric() ) %>% 
    step_dummy(all_nominal(), -all_outcomes() ) # create dummy variables for categorical predictors
  
  set.seed <- seed
  seeds <- vector(mode = "list", length = 5)
  for(j in 1:5) seeds[[j]] <- sample.int(nrow(trainData), 100)
  
  ## For the last model:
  seeds[[6]] <- sample.int(nrow(trainData), 1)
  
  if (best_metric=="Accuracy") outputFun = defaultSummary
  if (best_metric=="ROC") outputFun = twoClassSummary
  if (best_metric=="AUC") outputFun = prSummary0
  
  # * Cross Validation Setup ------------------------------------------------
  
  fitControl = trainControl(
    method = "cv", # k-fold cross validation
    number = 5, # Number of Folds
    search = "grid", # Default grid search for parameter tuning
    sampling = subsampling, # If none, please set to NULL
    summaryFunction = outputFun, # see custom function in this file
    classProbs = T, # should class probabilities be returned
    selectionFunction = "best", # best fold
    savePredictions = 'final',
    index = createResample(trainData$class, 5), seeds = seeds)
  
  # Model Training ----------------------------------------------------------
  
  if (method == "glm"){
    model = train(recSteps, data = trainData, metric=best_metric, 
                  method="glm", family= "binomial", maxit = 100,
                  trControl=fitControl
    )
  }
  
  if (method == "rpart"){
    model = train(recSteps, data = trainData, metric=best_metric, 
                  method="rpart", tuneLength = 30,
                  trControl=fitControl
    )
  }
  pred <- predict(model, testData)
  prob <- predict(model, testData, type = "response")[levels(trainData$class)[1]]
  obs <- testData$class
  result <- data.frame(obs, pred, prob)
  colnames(result)[3] <- "prob"
  
  if (index==1) return(result)
  
  Accuracy <- yardstick::accuracy(result, obs, pred)
  Sensitivity <- yardstick::sens(result, obs, pred)
  Specificity <- yardstick::spec(result, obs, pred)
  AUROC <- yardstick::roc_auc(result, obs, prob)
  Precision <- yardstick::precision(result, obs, pred)
  F1 <- yardstick::f_meas(result, obs, pred)
  AUPRC <- yardstick::pr_auc(result, obs, prob)
  
  final_result <- data.frame(Accuracy = Accuracy$.estimate,
                             Sensitivity = Sensitivity$.estimate,
                             Specificity = Specificity$.estimate,  
                             AUROC = AUROC$.estimate, 
                             Precision = Precision$.estimate, F1 = F1$.estimate,
                             AUPRC = AUPRC$.estimate)
  
  if (index==1) return(final_result)
  if (index==2) return(list(Performance=final_result, Prediction=result))
}
