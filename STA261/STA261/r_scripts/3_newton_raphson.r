# X_i ~ Geom(p) find mle of p with New-ton Raphson algorithm starting frmom p = 0.9


x <- c(2,9,6,3,1,8,1,2,1,3)	# 10 observations 
xBar <- mean(x)
pHat <- 0.9	# starting value for 

epsilon <- 5e-16	# threshold for exiting, when 
			# change in value of mle estimated cross iteration is minimal
delta <- 1		# change in estimator cross each iteration 

while(delta > epsilon){
	cat(paste(pHat, "\n"))
	pHatNew <- pHat + pHat*(1-pHat)*(1-pHat*xBar) / (xBar*pHat^2-2*pHat+1)
	delta <- abs(pHat - pHatNew)
	pHat <- pHatNew
}



