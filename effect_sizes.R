# install.packages(c("MASS", "akima", "robustbase"))
# install.packages(c("akima", "cobs", "robust", "mgcv", "scatterplot3d", "quantreg", "rrcov", "lars", "pwr", "trimcluster", "mc2d", "psych", "Rfit", "DepthProc", "class", "fda", "rankFD"))
# install.packages("devtools")
# library(devtools)
# install_github("nicebread/WRS", subdir="pkg")

library("WRS")

filenames = c(
              "C:\\Users\\martijnsparnaa\\surfdrive\\PhD\\Activity scheduler\\Simulations\\2021_12\\transfer\\results_20211215\\base_case_customer_customer_1_5_weight_over_contacts.csv")

# "C:\\Users\\martijnsparnaa\\surfdrive\\PhD\\Activity scheduler\\Simulations\\2021_12\\transfer\\results_20211215\\base_case_customer_customer_1_5_contacts_over_agents.csv",
# "C:\\Users\\martijnsparnaa\\surfdrive\\PhD\\Activity scheduler\\Simulations\\2021_12\\transfer\\results_20211215\\base_case_customer_customer_1_5_weight_over_agents.csv",
# "C:\\Users\\martijnsparnaa\\surfdrive\\PhD\\Activity scheduler\\Simulations\\2021_12\\transfer\\results_20211215\\base_case_customer_staff_1_5_contacts_over_agents.csv",
# "C:\\Users\\martijnsparnaa\\surfdrive\\PhD\\Activity scheduler\\Simulations\\2021_12\\transfer\\results_20211215\\base_case_customer_staff_1_5_weight_over_agents.csv",
# "C:\\Users\\martijnsparnaa\\surfdrive\\PhD\\Activity scheduler\\Simulations\\2021_12\\transfer\\results_20211215\\base_case_customer_staff_1_5_weight_over_contacts.csv",
# "C:\\Users\\martijnsparnaa\\surfdrive\\PhD\\Activity scheduler\\Simulations\\2021_12\\transfer\\results_20211215\\base_case_staff_customer_1_5_contacts_over_agents.csv",
# "C:\\Users\\martijnsparnaa\\surfdrive\\PhD\\Activity scheduler\\Simulations\\2021_12\\transfer\\results_20211215\\base_case_staff_customer_1_5_weight_over_agents.csv",
# "C:\\Users\\martijnsparnaa\\surfdrive\\PhD\\Activity scheduler\\Simulations\\2021_12\\transfer\\results_20211215\\base_case_staff_customer_1_5_weight_over_contacts.csv",

combinations <- list(c("nn_nts_bvd", "n_nts_bvd"),c("nn_nts_bvd", "nn_ts_bvd"),c("nn_nts_bvd", "nn_nts_svd"),c("nn_nts_bvd", "nn_nts_lvd"),
                     c("n_nts_bvd", "nn_ts_bvd"),c("n_nts_bvd", "nn_nts_svd"),  c("n_nts_bvd", "nn_nts_lvd"),
                     c("nn_ts_bvd", "nn_nts_svd"),   c("nn_ts_bvd", "nn_nts_lvd"),
                     c("nn_nts_svd", "nn_nts_lvd"))


for (filename in filenames) {
  print(filename)
  df <- read.csv(filename)
  effect_sizes = NULL
  
  for (combination in combinations) {
    X <- df[!is.na(df[,combination[1]]),combination[1]]
    Y <- df[!is.na(df[,combination[2]]),combination[2]]
    print(sprintf("%s (%s) - %s (%s)", combination[1], length(X), combination[2], length(Y)))
    n <- as.numeric(length(X))*as.numeric(length(Y))
    print(n)
    if (n > 2.025e+09) {
      factor = 2.025e+09/n*0.99
      print(factor)
      sample_size_x = floor(length(X)*factor)
      sample_size_y = floor(length(Y)*factor)
      X <- X[sample(length(X), sample_size_x)]     
      Y <- Y[sample(length(Y), sample_size_y)]
      print(sprintf("%s (%s) - %s (%s)", combination[1], length(X), combination[2], length(Y)))
    }
    
    out <- ES.summary(X,Y)
    
    if (is.null(effect_sizes)) {
        effect_sizes <- as.data.frame(out)
        effect_sizes <- effect_sizes[,c(2,3,4,5,1)]
    } else {
      effect_sizes <- cbind(effect_sizes, Est = out[,1])
    }
    names(effect_sizes)[names(effect_sizes) == 'Est'] <- sprintf("%s - %s", combination[1], combination[2])
  }
  save_filename = sprintf("%s_ES.csv", substr(filename, 1, nchar(filename) - 4))
  write.csv(effect_sizes,save_filename)
}


