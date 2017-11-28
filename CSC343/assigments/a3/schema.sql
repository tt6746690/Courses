drop schema if exists quizschema cascade;
create schema quizschema;
set search_path to quizschema;



-- creating tables
create table Student (
    student_id   int         not null, 
    first_name   varchar(50) not null, 
    last_name    varchar(50) not null, 
    primary key (student_id)
    constraint ck_student_id_10_digits  -- student id is a 10-digit number
        check (student_id between 0 and 9999999999)
)

create table Class (
    class_id    int         not null,
    room        varchar(50) not null, 
    grade       varchar(50) not null,    -- grade not unique, so can have multiple grade for a single Class
    teacher     varchar(50) not null,
    primary key (class_id),
    constraint ck_room_cannot_have_more_than_one_teacher
        unique (room, teacher)
)

-- Student 
create table StudentInClass (
    student_id   int         not null,
    class_room   varchar(50) not null,
    primary key (student_id, class_room),
    constraint ck_student_id_in_Student
        foreign key (student_id) references Student(student_id),
    constraint ck_class_room_in_Class
        foreign key (class_room) references Class(room)
)



-- 3 types of questions 
create domain QuestionType as (3)
