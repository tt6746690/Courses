#include <stdio.h>

#define NUM_STUDENTS 5

int main() {
    struct student {
      char first_name[20];
      char last_name[20];
      int year;
      float gpa;
    };
  
    FILE *student_file;
    int error;
    struct student s[5] = {
      {"Betty", "Holberton", 4, 3.8},
      {"Michelle", "Craig", 4, 3.5},
      {"Andrew", "Petersen", 3, 0.0},
      {"Daniel", "Zingaro", 1, 2.2},
      {"Grace", "Hopper", 6, 3.9}
    };
  
    student_file = fopen("five_students", "wb");
    if (student_file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }
  
    fwrite(s, sizeof(struct student), NUM_STUDENTS, student_file);
    error = fclose(student_file);
    if (error != 0) {
        fprintf(stderr, "fclose failed\n");
        return 1;
    }

    return 0;
}