library(tidyverse)
library(data.table)
library(rstudioapi)
library(skimr)
library(car)
library(h2o)
library(rlang)
library(glue)
library(highcharter)
library(lime)

data <- fread("mushrooms.csv")
data %>% skim()
data %>% View()

func <- function(data){
  gsub("\\'","", data)
  gsub("\\'" ,"", data)
}
data <- apply(data, 2, func) %>% as.data.frame()

data$class %>% table() %>% prop.table()
data$class <- data$class %>% recode(" 'e' =1 ; 'p' = 0") %>% as.factor()
class(data$class)

names(data) <- names(data) %>% 
  str_replace_all(" ","_") %>%
  str_replace_all("-","_") %>%
  str_replace_all("\\(","_") %>% 
  str_replace_all("\\)","") %>%
  str_replace_all("\\/","_") %>% 
  str_replace_all("\\:","_") %>% 
  str_replace_all("\\.","_") %>% 
  str_replace_all("\\,","_") %>% 
  str_replace_all("\\%","_")


colnames(data)
# datada olan problemə görə (və modeldə yaranan problemə görə)
data$cap_shape <- data$cap_shape %>% as.factor()
data$cap_surface <- data$cap_surface %>% as.factor()
data$cap_color <- data$cap_color %>% as.factor()
data$bruises_3F <- data$bruises_3F %>% as.factor()
data$odor <- data$odor %>% as.factor()
data$gill_attachment <- data$gill_attachment %>% as.factor()
data$gill_spacing <- data$gill_spacing %>% as.factor()
data$gill_size <- data$gill_size %>% as.factor()
data$gill_color <- data$gill_color %>% as.factor()
data$stalk_shape <- data$stalk_shape %>% as.factor()
data$stalk_root <- data$stalk_root %>% as.factor()
data$stalk_color_above_ring <- data$stalk_color_above_ring %>% as.factor()
data$stalk_color_below_ring <- data$stalk_color_below_ring %>% as.factor()
data$stalk_surface_above_ring <- data$stalk_surface_above_ring %>% as.factor()
data$stalk_surface_below_ring<- data$stalk_surface_below_ring %>% as.factor()
data$veil_type <- data$veil_type %>% as.factor()
data$veil_color <- data$veil_color %>% as.factor()
data$ring_number <- data$ring_number %>% as.factor()
data$ring_type <- data$ring_type %>% as.factor()
data$spore_print_color <- data$spore_print_color %>% as.factor()
data$population <- data$population %>% as.factor()
data$habitat <- data$habitat %>% as.factor()

data %>% glimpse()

# 1. Build classification model with h2o.automl();
# 2. Apply Cross-validation;
h2o.init()
h2o_data <- data %>% as.h2o()

h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'class'
features <- data %>%  select(-class) %>% names()


model <- h2o.automl(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  nfolds = 10,
  seed = 123, max_runtime_secs = 120)


model@leaderboard %>% as.data.frame()
model@leader 


# Predicting the Test set results ----
pred <- model@leader %>% h2o.predict(test) %>% as.data.frame()


# Threshold / Cutoff ----  
model@leader %>% 
  h2o.performance(test) %>% 
    h2o.find_threshold_by_max_metric('f1') -> treshold


# ----------------------------- Model evaluation -----------------------------

# Confusion Matrix----
model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))


model@leader %>% 
  h2o.performance(test) %>% 
  h2o.metric() %>% 
  select(threshold,precision,recall,tpr,fpr) %>% 
  add_column(tpr_r=runif(nrow(.),min=0.001,max=1)) %>% 
  mutate(fpr_r=tpr_r) %>% 
  arrange(tpr_r,fpr_r) -> deep_metrics

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.auc() %>% round(2) -> auc


highchart() %>% 
  hc_add_series(deep_metrics, "scatter", hcaes(y=tpr,x=fpr), color='green', name='TPR') %>%
  hc_add_series(deep_metrics, "line", hcaes(y=tpr_r,x=fpr_r), color='red', name='Random Guess') %>% 
  hc_add_annotation(
    labels = list(
      point = list(xAxis=0,yAxis=0,x=0.3,y=0.6),
      text = glue('AUC = {enexpr(auc)}'))
  ) %>%
  hc_title(text = "ROC Curve") %>% 
  hc_subtitle(text = "Model is performing much better than random guessing") 


# Check overfitting ----
model@leader %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)


