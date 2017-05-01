#include <stdio.h>
#include <string.h>

int main() {
    struct student {
        char first_name[20];
        char last_name[20];
        int year;
        float gpa;
    };
  
    struct student s;

    // Cannot be dereferenced until it points to
    // allocated space:
    struct student *p;
  
    // Initialize the members of s:
    strcpy(s.first_name, "Jo");
    strcpy(s.last_name, "Smith");
    s.year = 2;
    s.gpa = 3.2;
  
    // p now points to the address of struct student s:
    p = &s;

    // Dereference p to get the struct, then use the
    // dot operator to access the "gpa" member:
    (*p).gpa = 3.8;

    p->year = 1;    // this is the same as (*p).year = 1;
    strcpy(p->first_name, "Henrick");
  
    printf("Name: %s %s\n",
           s.first_name, s.last_name);
    printf("Year: %d. GPA: %.2f\n", s.year, s.gpa);
  
    return 0;
}













