
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv){

    // file starts at 0x000000 
    int rodata = strtol(argv[1], NULL, 16);
    int size = strtol(argv[2], NULL, 10);

    FILE *input = fopen("hello", "rb"); 
    // must check for error; otherwise segmentation fault
    if(input == NULL){
        fprintf(stderr, "could not open file");
        perror("fopen (hello) ");    // relevant error message
        exit(1);
    }

    if(fseek(input, rodata, SEEK_SET) == -1){
        perror("fseek");
        exit(1);
    }
   
    // have to allocate memory for strings 
   char *strings = malloc(size);
   fread(strings, size, 1, input); 
    
   // print 
   for(int i =0; i < size; i++){
       if(strings[i] == '\0'){
           strings[i] = '\n';
       }
   }
   return 0;

}
