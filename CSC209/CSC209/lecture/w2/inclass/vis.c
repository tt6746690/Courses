#include <stdio.h>


int main() {
    int before[4] = {-1, -1, -1, -1};
    int ages[4] = {5, 7, 18, 20};
    int after[4] = {-1, -1, -1, -1};

    for(int i = 0; i <= 4; i++) {
        ages[i]++;
    }

    for(int j = 0; j < 4; j++) {
        printf("before[%d] has value %d\n", j, before[j]);
    }
    for(int j = 0; j < 4; j++) {
        printf("ages[%d] has value %d\n", j, ages[j]);
    }
    for(int j = 0; j < 4; j++) {
        printf("after[%d] has value %d\n", j, after[j]);
    }

    return 0;
}
