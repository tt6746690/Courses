#include <stdio.h>
#include <string.h>

int main() {
    struct student {
        char first_name[14];
        char last_name[20];
        int year;
        float gpa;
    };
  
    FILE *student_file;
    int error;
    struct student s;

    student_file = fopen("student_data", "wb");
    if (student_file == NULL) {
        fprintf(stderr, "Error: could not open file\n");
        return 1;
    }
  
    strcpy(s.first_name, "Betty");
    strcpy(s.last_name, "Holberton");
    s.year = 4;
    s.gpa = 3.8;
    
    fwrite(&s, sizeof(struct student), 1, student_file);
  
    strcpy(s.first_name, "Grace");
    strcpy(s.last_name, "Hopper");
    s.year = 6;
    s.gpa = 3.9;
  
    fwrite(&s, sizeof(struct student), 1, student_file);

    error = fclose(student_file);
    if (error != 0) {
        fprintf(stderr, "Error: fclose failed\n");
        return 1;
    }

    return 0;
}