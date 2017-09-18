# Sample R script for STA302
# See if you can run this in RStudio!

a<-4
3->b
print(a+b)

c<-seq(1,20,2)
print(c)

write.csv(c,"myfile.csv") # Can also add: row.names=FALSE
d<-read.csv("myfile.csv")

d<-t(d[,2]) # Transpose of d's second column
e<-rnorm(length(d))
plot(d,e)

?rnorm
