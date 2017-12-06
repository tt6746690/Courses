-- 1. What constraints from the domain could not be enforced?
--  * class has at least 1 student 
--      in this case cardinality of Class has min = 1
--      Can create a foreign key for Class(id) references StudentInClass(class_id)
--      class_id must be made unique, but there can be many student in a class. 
--      So cannot construct foreign key to satisfy this constraint
--  * quiz has at least 1 question
--      For same reason as above
-- 2. What constraints that could have been enforced were not enforced? Why not?
--  * correct answers do not have hints, 
--      question and hints are in different tables, a check constraint would involve subqueries, 
--      which is not allowed according to a piazza post
--  * only a student in the class that was assigned a quiz can answer a question
--      Cross table constraint requiring a trigger, but is not allowed according to a piazza list

drop schema if exists quizschema cascade;
create schema quizschema;
set search_path to quizschema;

-- A student can be in classes and take quizzes
create table Student (
    id           varchar(10) not null, 
    first_name   varchar(45) not null, 
    last_name    varchar(45) not null, 
    primary key (id),
    constraint ck_student_id_digits  -- student id is a 10-digit number
        check (id ~ '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'),
    constraint ck_student_id_digits_len_10  -- student id is a 10-digit number
        check (length(id)=10)
);


-- A class holds holds one or more students
--  there are multiple classes for the same grade (not unique)
--  A room can have at most 2 classes in it but never more than 1 teacher, 
--      So combination of room and teacher is unique
create table Class (
    id          int         not null,
    room        int         not null, 
    grade       int         not null,    -- grade not unique, so can have multiple grade for a single Class
    teacher     varchar(45) not null,    
    primary key (id),
    constraint ck_room_cannot_have_more_than_one_teacher
        unique (room, teacher)
);        


-- Relationship set describing that a student is in a particular class
--  Student --(0, N)-- StudentInClass --(1, N)-- Class
create table StudentInClass (
    student_id   varchar(10) not null,  -- cannot be unique, since a student can be in multiple classes 
    class_id     int         not null,  -- cannot be unique, since a student can have multiple students
    primary key (student_id, class_id),
    constraint ck_student_id_in_Student
        foreign key (student_id) references Student(id),
    constraint ck_class_id_in_Class
        foreign key (class_id) references Class(id)
);

-- constraint that to satisfy cardinality min=1 for Class 
-- alter table Class 
-- add constraint ck_class_has_one_or_more_Student
-- foreign key (id) references StudentInClass(student_id);


-- 3 types of questions 
--  multiple choice questions have answer of type VARCHAR
--  true false questions have answer of type BOOLEAN
--  numeric questions have answer of type INT 
create type question_type as 
enum('multiple_choice', 'true_false', 'numeric');

-- ER relation 
--  Question --(1,1)-- MultipleChoiceQuestion
--  Question --(1,1)-- TrueFalseQuestion
--  Question --(1,1)-- NumericQuestion
--  MultipleChoiceQuestion --(2,N) --(1,1)-- MultipleChoiceOption
--  NumericQuestion --(0,1)--(1,1)-- NumericQuestionHint

-- Questions has text and a single correct answer 
create table Question (
    id      int             not null,
    type    question_type   not null,
    text    varchar(200)    not null,
    primary key (id)
);

-- Multiple choice options 
--  An incorrect answer may have one hint associated with it
create table MultipleChoiceOption (
    qid     int             not null,
    text    varchar(100)    not null,               
    hint    varchar(100),   -- can be null, ensures option--(1,1)--hint
    primary key (qid, text),
    constraint ck_qid_in_Question
        foreign key (qid) references Question(id)
);


-- Multple choice question 
--  A multiple choice question has at least 2 options and solution must be one of them
--  Correct answers do no have hints, incorrect question may have at most 1 hints
create table MultipleChoiceQuestion (
    qid     int             not null,
    answer  varchar(100)    not null,
    primary key (qid),
    constraint ck_qid_in_Question
        foreign key (qid) references Question(id),
    constraint ck_answer_is_one_of_options 
        foreign key (qid, answer) references MultipleChoiceOption(qid, text)
    -- constraint ck_correct_answer_do_not_have_hints not checked
);


-- True False Question 
--  A true false question has answer with value either TRUE of FALSE
create table TrueFalseQuestion (
    qid     int         not null,
    answer  boolean     not null,   -- 1 true, 0 false
    primary key (qid),
    constraint ck_qid_in_Question 
        foreign key (qid) references Question(id)
);


-- Numeric Question hints 
--  A hint is specific to a range [lower_bound, upper_bound) of values 
create table NumericQuestionHint (
    qid             int             not null, 
    lower_bound     int             not null, 
    upper_bound     int             not null, 
    text            varchar(100)    not null, 
    primary key (qid, lower_bound, upper_bound),
    constraint ck_qid_in_Question
        foreign key (qid) references Question(id)
);


-- Numeric Question 
--  Incorrect answer to a numeric question has 0 or 1 hint associated with it 
--  Correct answer does not have hints
create table NumericQuestion (
    qid         int     not null, 
    answer      int     not null, 
    primary key (qid),
    constraint ck_qid_in_Question
        foreign key (qid) references Question(id)
    -- constraint ck_correct_answer_do_not_have_hints not implemented
);

-- ER relation 
--  Quiz --(1,N)--(1,1)--Question

-- Quiz 
--  A quiz has 1 or more questions 
create table Quiz (
    id          varchar(50) not null, 
    title       varchar(50) not null, 
    due         date        not null,
    class_id    int         not null, 
    hint_flag   boolean     not null, 
    primary key (id),
    constraint ck_class_id_in_Class 
        foreign key (class_id) references Class(id)
);

-- QuizQuestion
--  questions that appear in a quiz 
create table QuizQuestion (
    id          int         not null,
    quiz_id     varchar(50) not null, 
    qid         int         not null, 
    weight      int         not null, 
    primary key (id),
    unique (quiz_id, qid),
    constraint ck_quiz_id_in_Quiz 
        foreign key (quiz_id) references Quiz(id),
    constraint ck_qud_in_Question 
        foreign key (qid) references Question(id)
);


-- Response to Multiple choice questions 
create table MultipleChoiceResponse (
    qqid        int             not null, 
    student_id  varchar(10)     not null, 
    answer      varchar(100)    not null, 
    primary key (qqid, student_id),
    constraint ck_qid_in_QuizQuestion 
        foreign key (qqid) references QuizQuestion(id),
    constraint ck_student_id_in_Student
        foreign key (student_id) references Student(id)
);


-- Response to True False questions 
create table TrueFalseResponse (
    qqid        int         not null, 
    student_id  varchar(10) not null, 
    answer      boolean     not null, 
    primary key (qqid, student_id),
    constraint ck_qid_in_QuizQuestion 
        foreign key (qqid) references QuizQuestion(id),
    constraint ck_student_id_in_Student
        foreign key (student_id) references Student(id)
);



-- Response to numeric questions 
create table NumericResponse (
    qqid        int         not null, 
    student_id  varchar(10) not null, 
    answer      int         not null, 
    primary key (qqid, student_id),
    constraint ck_qid_in_QuizQuestion 
        foreign key (qqid) references QuizQuestion(id),
    constraint ck_student_id_in_Student
        foreign key (student_id) references Student(id)
);
