
observed <- c(18, 28, 56, 105, 126, 146, 164, 161, 
		123, 101, 74, 53, 23, 15, 9, 5)

names <- c("0-2", paste(3:16), "17+")

bp <- barplot(observed, names=names, axes=TRUE, xlab="No of particles", ylab="Frequenc")

bp
