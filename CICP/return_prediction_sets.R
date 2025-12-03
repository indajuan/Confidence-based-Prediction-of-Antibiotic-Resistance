library(dplyr)
library(grid)
library(tidyr)


#setwd("Documents/GitHub/Confidence-based-Prediction-of-Antibiotic-Resistance")
calibration_list <- readRDS("CICP/calibration.rds")

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[[1]]
output_file <- args[[2]]
model_name <- args[[3]]

pathogens <- c("E. coli", "K. pneumoniae", "P. aeruginosa")
names(pathogens) <- c("ESCCOL", "KLEPNE", "PSEAER")

if (!(model_name %in% c("full", "no patient"))) {
  stop(paste("ERROR: Value", v, "is not an allowed model. Exiting."), call. = FALSE)
}

#if (!(0 < confidence < 1 )) {
#  stop(paste("ERROR: Value", confidence, "is not an allowed confidence value. Exiting."), call. = FALSE)
#}

read_results <- function(file){
  df <- read.csv(file, header = F)
  df <- df %>% 
    rename(index = V1,
           antibiotic = V2,
           AST_true = V3,
           AST_pred = V4,
           Antibiotic_predictors = V5,
           Patient_data = V6,
           Output_neural_networks = V7)
  df <- df %>% 
    mutate(pathogen = as.vector(pathogens[sapply(strsplit(Antibiotic_predictors, split = " "), function(x) x[1])]))
  df <- df %>%
    mutate(
      nums = regmatches(Output_neural_networks,
                        gregexpr("-?\\d+\\.?\\d*", Output_neural_networks)),
      score0 = as.numeric(sapply(nums, `[`, 1)),
      score1 = as.numeric(sapply(nums, `[`, 2)),
      soft0 =  exp(score0)/(exp(score0) + exp(score1)), 
      soft1 = exp(score1)/(exp(score0) + exp(score1))) %>%
    select(-nums) 
  return(df)
}

return_prediction_set <- function(calibration_list, soft0, soft1, model, pathogen, antibiotic, pheno){
  
  if(pheno == "susceptible"){
    cal_susc <- calibration_list[[model]][[pathogen]][[antibiotic]][["susceptible"]]
    scores_le <- which(cal_susc$score <= soft0)
    scores_eq <- which(cal_susc$score == soft0)
    ids <- if(length(scores_le) > 0) {
      min(max(scores_le), if(length(scores_eq) > 0) min(scores_eq) else Inf)
    } else {
      Inf
    }
    #ids <- min(max(which(cal_susc$score <= soft0)), min(which(cal_susc$score == soft0)))
    ps <- ifelse(is.infinite(ids), 
                 1 / (cal_susc$n[1] + 1), 
                 (cal_susc$id[ids] + 1) / (cal_susc$n[ids] + 1))
  } else{
    cal_res <- calibration_list[[model]][[pathogen]][[antibiotic]][["resistant"]]
    scores_le <- which(cal_res$score <= soft1)
    scores_eq <- which(cal_res$score == soft1)
    ids <- if(length(scores_le) > 0) {
      min(max(scores_le), if(length(scores_eq) > 0) min(scores_eq) else Inf)
    } else {
      Inf
    }
    #ids <- min(max(which(cal_res$score <= soft1)), min(which(cal_res$score == soft1)))
    ps <- ifelse(is.infinite(ids), 
                 1 / (cal_res$n[1] + 1), 
                 (cal_res$id[ids] + 1) / (cal_res$n[ids] + 1))
  }
  return(ps)
}

df <- read_results(input_file)




df <- df %>% 
  rowwise() %>%
  mutate(ps = return_prediction_set(calibration_list, soft0, soft1, "full", pathogen, antibiotic, "susceptible"),
         pr = return_prediction_set(calibration_list, soft0, soft1, "full", pathogen, antibiotic, "resistant")) 

conf <- 0.85
df <- df %>% mutate(cp_85 = ifelse(ps > (1-conf) & pr > (1-conf), "S/R",
                             ifelse(ps > (1-conf) & pr <= (1-conf), "S",
                                    ifelse(ps <= (1-conf) & pr > (1-conf), "R", ""))))
conf <- 0.9
df <- df %>% mutate(cp_90 = ifelse(ps > (1-conf) & pr > (1-conf), "S/R",
                      ifelse(ps > (1-conf) & pr <= (1-conf), "S",
                      ifelse(ps <= (1-conf) & pr > (1-conf), "R", ""))))
conf <- 0.95
df <- df %>% mutate(cp_95 = ifelse(ps > (1-conf) & pr > (1-conf), "S/R",
                             ifelse(ps > (1-conf) & pr <= (1-conf), "S",
                                    ifelse(ps <= (1-conf) & pr > (1-conf), "R", ""))))
conf <- 0.975
df <- df %>% mutate(cp_975 = ifelse(ps > (1-conf) & pr > (1-conf), "S/R",
                             ifelse(ps > (1-conf) & pr <= (1-conf), "S",
                                    ifelse(ps <= (1-conf) & pr > (1-conf), "R", ""))))
df <- df %>% 
  ungroup() %>%
  select(-c(pathogen, score0, score1, soft0, soft1))

write.csv(df, file = output_file)
