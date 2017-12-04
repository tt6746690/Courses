-- 1. Report full name and student number of all students in the database 
create view q1_solution as 
select id as student_number, first_name||' '||last_name as full_name 
from Student;

-- display
select * from q1_solution;