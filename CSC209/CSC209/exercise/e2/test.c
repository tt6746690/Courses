#include <stdio.h>

int size
int check_sum;


for(int i=0;i<size;i+=2){
    check_sum -= command[i];
}
for(int i=1;i<size;i+=2){
    check_sum += command[i];
}
printf("%d", check_sum);
