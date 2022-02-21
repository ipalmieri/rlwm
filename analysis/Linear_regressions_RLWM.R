# Opens relevant libraries using pacman
pacman::p_load(dplyr, psych, car, MASS, DescTools, QuantPsyc, ggplot2, rstatix, lmtest, scatterplot3d)


# Other relevant libraries
library ('foreign')
library('NCmisc')
library(xlsx)


#### LOGISTIC REGRESSION FOR MODEL DATA_______________________________________


setwd("")
data <- read.xlsx("trial_by_trial_data_for_each_subject.xlsx", sheetIndex = 1)

glimpse(data)


# Transform variables to integer and factors
data$Zrpred <- as.integer(scale(data$rpred, center = TRUE, scale = TRUE))
data$Zblock_size <- as.factor(scale(data$block_size, center = TRUE, scale = TRUE))
data$ZBlock_order <- as.integer(scale(data$block, center = TRUE, scale = TRUE))
data$Zdelay <- as.integer(scale(data$delay, center = TRUE, scale = TRUE))


# Verify response time distribution
par(mfrow = c(1,2))

hist(data$response_time_c, 
     main="Corrected Reaction Time", 
     xlab="Response time", 
     border="black", 
     col="darkgray", 
     family = "serif", 
     font = 2, 
     font.lab = 2, 
     font.axis = 2)

hist(log(data$response_time_c), 
     main="Corrected Reaction Time (log transformed)", 
     xlab="Response time", 
     border="black", 
     col="darkgray", 
     family = "serif", 
     font = 2, 
     font.lab = 2, 
     font.axis = 2)



#### Initiates Linear Regression_____________________________________

linear <- lm(log(response_time_c) ~ Zrpred + Zblock_size + Zdelay + ZBlock_order, data = data)
summary(linear)


# Tests for leveraged standardized residuals and Cook's Distance
par(mfrow = c(2,2))

plot(linear)

summary(stdres(linear))


# Tests for multicollinearity
vif(linear)


# Removes the specific rows considered outliers on Cook's distance - If needed
#data_reglin <- data[-c(CASEID_X, CASEID_Y, CASEID_Z), ]


# Removes influential cases using Cook's Distance usual criteria 4/n (n = trials for the current data frame)
cooksd_l <- cooks.distance(linear)
influential_l <- as.numeric(names(cooksd_l)[(cooksd_l > (4/16200))])
data_reglin <- data[-influential_l, ]


# Linear regression without influential cases
linear_2 <- lm(log(response_time_c) ~ Zrpred + Zblock_size + Zdelay + ZBlock_order, data = data_reglin)
summary(linear_2)


# Tests again for leveraged standardized residuals and Cook's Distance
par(mfrow = c(2,2))

plot(linear_2)

summary(stdres(linear_2))


# Tests again the best model - Choose the on with the highest R^2, if assumptions were met
linear_2 <- lm(Log_RT ~ Zrpred + Zblock_size + Zdelay + ZBlock_order, data = data_reglin)
summary(linear_2)



