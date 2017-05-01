
#Calculating MSE by Monte-Carlo Simulation
MSEs <- function(lambda, n){
    theta <- exp(-lambda)
    samp <- matrix(rpois(n*1e5, lambda), ncol=n)
    xBar <- apply(samp, 1, mean)
    thetaHat1 <- exp(-xBar) 
    thetaHat2 <- (1-1/n)^(n*xBar) 
    MSE1 <- mean((thetaHat1-theta)^2) 
    MSE2 <- mean((thetaHat2-theta)^2) 
    return(c(MSE1,MSE2))
}



n <- 5
Vals1 <- t(sapply(.1*c(1:40), MSEs, n=n))
plot(.1*c(1:40), Vals1[,1], type='l')
lines(.1*c(1:40), Vals1[,2], lty=2, col=2)
