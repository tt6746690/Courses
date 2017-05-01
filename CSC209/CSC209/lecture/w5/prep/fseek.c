#include <stdio.h>

int main() {
    struct student {
        char first_name[20];
        char last_name[20];
        int year;
        float gpa;
    };
  
    FILE *student_file;
    int error, student_num;
    struct student s;
  
    student_file = fopen("five_students", "rb");
    if (student_file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }
  
    printf("Type -1 to exit.\n");
    printf("Enter the index of the next student to view: ");

    scanf("%d", &student_num);
    while (student_num >= 0) {
        fseek(student_file, student_num * sizeof(struct student), SEEK_SET);
    
        error = fread(&s, sizeof(struct student), 1, student_file);
        if (error == 1) {  
            printf("Name: %s %s\n", s.first_name, s.last_name);
            printf("Year: %d. GPA: %.2f\n", s.year, s.gpa);
        }
        else {
            fprintf(stderr, "Error: student could not be read.\n");
        }
        printf("Enter the index of the next student to view: ");
        scanf("%d", &student_num);
    }
    
    error = fclose(student_file);
    if (error != 0) {
        fprintf(stderr, "fclose failed\n");
        return 1;
    }

    return 0;
}