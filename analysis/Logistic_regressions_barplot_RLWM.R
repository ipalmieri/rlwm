# Opens relevant libraries using pacman
pacman::p_load(dplyr, psych, car, MASS, DescTools, QuantPsyc, ggplot2, rstatix, lmtest, scatterplot3d)

# Other relevant libraries
library ('foreign')
library('NCmisc')



#### LOGISTIC REGRESSION FOR MODEL DATA_______________________________________


# Open dataframes - trial-by-trial behavioral data or simulation data 
setwd("")
data <- read.csv("trial_by_trial_data_for_each_subject.csv", header = T)

glimpse(data)


# Transform variables to z scores - ZBlock_order is not considered for simulation data
data$Zrpred <- as.integer(scale(data$rpred, center = TRUE, scale = TRUE))
data$Zblock_size <- as.factor(scale(data$block_size, center = TRUE, scale = TRUE))
data$ZBlock_order <- as.integer(scale(data$block, center = TRUE, scale = TRUE))
data$Zdelay <- as.integer(scale(data$delay, center = TRUE, scale = TRUE))


#### Initiates Logistic Regression
logistic <- glm(reward ~ Zrpred + Zblock_size + ZBlock_order + Zdelay, family = "binomial"(link = 'logit'), data = data)
summary(logistic)


# Tests for multicollinearity 
vif(logistic)

# Tests for leveraged standardized residuals and Cook's Distance
plot(logistic, which = 4)
plot(logistic, which = 5)

summary(stdres(logistic))


# Removes influential cases using Cook's Distance usual criteria 4/n (n = trials for the current data frame)
cooksd <- cooks.distance(logistic)
influential <- as.numeric(names(cooksd)[(cooksd > (4/1620000))])
data_reglog <- data[-influential, ]

  
# Logistic regression without influential cases
logistic_2 <- glm(reward ~ Zrpred + Zblock_size + Zdelay, family = "binomial"(link = 'logit'), data = data_reglog)
summary(logistic_2)


# Tests again for leveraged standardized residuals and Cook's Distance
plot(logistic_2, which = 4)
plot(logistic_2, which = 5)

summary(stdres(logistic_2))

# Shows pseudo R^² - Choose the model with highest prediction if assumptions were met
PseudoR2(logistic, which = 'Nagelkerke')
PseudoR2(logistic_2, which = 'Nagelkerke')

# Reveal values for Akaike and Bayesian Information Criteria  - Models must have the same length for comparison
# Choose the model with the lower values if assumptions were met
#AIC(logistic, logistic_2)
AIC(logistic)
AIC(logistic_2)
#BIC(logistic, logistic_2)
BIC(logistic)
BIC(logistic_2)


# Compare models using chi-square - Models must have the same length
anova(logistic_2, logistic, test = "Chisq")


# Classification table - Choose the model with the best results if assumptions were met
ClassLog(logistic, data$reward)
ClassLog(logistic_2, data_reglog$reward) #best model


# Do regression again for the best model
logistic_2 <- glm(reward ~ Zrpred + Zblock_size + Zdelay, family = "binomial"(link = 'logit'), data = data_reglog)
summary(logistic_2)




#### END OF LOGISTIC REGRESSION FOR MODEL DATA_______________________________________





#### BARPLOT FOR AIC COMPARISON_______________________________________

if(!require(devtools)) install.packages("devtools")
devtools::install_github("kassambara/ggpubr")


library(xlsx)
library(ggpubr)

# Open data frame with AIC values for each model
setwd("")
data_p <- read.xlsx("data_frame_with_model_values_each_subject.xlsx", sheetIndex = 1) 


# Tranform IDs into factor
data_p$caseid <- factor(data_p$caseid)
is.factor(data_p$caseid)


# Create delta scores - Difference between AIC values between models
data_p$Classic.RLWMi <- data_p$Classic - data_p$RLWMi
data_p$Classic.WM <- data_p$Classic - data_p$WM
data_p$WM.RLWMi <- data_p$WM - data_p$RLWMi


# Transform data frame to long format for the graphic
data_p_long <- gather(data_p, models, value, 11:13, factor_key=TRUE) 


# Merge bar plots
barplot <- ggbarplot(data_p, 
          y = c("Classic.RLWMi", "Classic.WM", "WM.RLWMi"),
          merge = TRUE, 
          ylab = "AIC (Mean and Standard Error)", 
          add = "mean_se",
          ylim = c(0, 50),
          size = 0.5,
          width = 0.5,
          palette = c("black", "darkblue", "darkred")
          )

ggpar(barplot, 
      font.family = 'serif',
      font.ytickslab = 14, 
      font.legend = c(14, "bold", "black"),
      legend = "top",
      legend.title = "Computational Models:",
      font.y = c(14, "bold", "black"),
      ylab(c("Delta Classic - RLWMi", "Delta_C_WM", "Delta_WM_RL")),
      xlab = FALSE
      )

#### END OF BARPLOT FOR AIC COMPARISON_______________________________________
