# R code for STA302/STA1001 Assignment 2

a2 = read.table("DataA2.txt",sep=" ",header=T) # Load the data set
fev <- a2$fev; age <- a2$age
q = "1a" # question number to execute: 1a, 1b, 2a, or 2d

# Question 1
if (q=="1a") {
  # Q1(a) plot FEV vs age and the scale vs location
  mod1 = lm(fev~age)
  par(mfrow=c(1,2))
  plot(age,fev,type="p",col="blue",pch=21, main="FEV vs age")
  plot(mod1,which=3) # 'which' selects from among the 4 lm plots
  # Comment: The scatter plot indicates a linear association between FEV and age.
  # The scale-location plot indicates that the variance in the response variable 
  # increases with age
} 

if (q=="1b") {
  # Q1(b) plot effects of three transformations
  mod2 = lm(sqrt(fev)~sqrt(age))
  mod3 = lm(log(fev)~log(age))
  a=1/fev; b=1/age; mod4 = lm(a~b)
  par(mfcol=c(2,3))
  plot(sqrt(age),sqrt(fev),type="p",col="blue",pch=21, main="SQRT")
  plot(mod2,which=3)
  plot(log(age),log(fev),type="p",col="blue",pch=21, main="LOG")
  plot(mod3,which=3)
  plot(1/age,1/fev,type="p",col="blue",pch=21, main="RECIPROCAL")
  plot(mod4,which=3)
} # Log looks best

# Question 2
mod3 = lm(log(fev)~log(age))
if (q=="2a") {
  # Q2(a) The estimated regression line is hat{log(fev)} = 0.8462 log(age) - 0.9877
  mod3$coefficients
}

# Q2(b) The residuals vs fitted line is greatly improved, but the variance still isn't
# constant: intermediate fitted values (around y-hat = 0.8) have relatively low 
# variance. Separately, the QQ plot of the residuals (not shown) looks good.

# Q2(c) The slope indicates that for a k-fold increase in age, FEV increases by a 
# factor of k^0.8462. So if age doubles, FEV increases by 80%.

if (q=="2d") {
  # Q2(d) CIs and PIs
  CI<-predict.lm(mod3,newdata=data.frame(age=c(8,17,21)), interval=c("confidence"))
  PI<-predict.lm(mod3,newdata=data.frame(age=c(8,17,21)), interval=c("prediction"))
  print(exp(CI)); print(exp(PI))
}
