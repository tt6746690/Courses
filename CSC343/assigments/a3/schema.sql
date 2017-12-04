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
    id           bigint      not null, 
    first_name   varchar(45) not null, 
    last_name    varchar(45) not null, 
    primary key (id),
    constraint ck_student_id_10_digits  -- student id is a 10-digit number
        check (id between 0 and 9999999999)
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
    student_id   bigint      not null,  -- cannot be unique, since a student can be in multiple classes 
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
    student_id  bigint          not null, 
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
    student_id  bigint      not null, 
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
    student_id  bigint      not null, 
    answer      int         not null, 
    primary key (qqid, student_id),
    constraint ck_qid_in_QuizQuestion 
        foreign key (qqid) references QuizQuestion(id),
    constraint ck_student_id_in_Student
        foreign key (student_id) references Student(id)
);



-- Populate tables
insert into Student(id, first_name, last_name)
values 
( 998801234, 'Lena', 'Headey'),
(  10784522, 'Peter', 'Dinklage'),
( 997733991, 'Emilia', 'Clarke'),
(5555555555, 'Kit', 'Harrington'),
(1111111111, 'Sophie', 'Turner'),
(2222222222, 'Maisie', 'Williams');


insert into Class(id, room, grade, teacher)
values 
(0, 120, 8, 'Mr Higgins'),
(1, 366, 5, 'Miss Nyers');

insert into StudentInClass(student_id, class_id)
values
( 998801234, 0),
(  10784522, 0),
( 997733991, 0),
(5555555555, 0),
(1111111111, 0),
(2222222222, 1);



insert into Question(id, type, text)
values 
(782, 'multiple_choice', 'What do you promise when you take the oath of citizenship?'),
(566, 'true_false', 'The Prime Minister, Justin Trudeau, is Canada''s Head of State.'),
(601, 'numeric', 'During the "Quiet Revolution," Quebec experienced rapid change. In what
decade did this occur? (Enter the year that began the decade, e.g., 1840.)'),
(625, 'multiple_choice', 'What is the Underground Railroad?'),
(790, 'multiple_choice', 'During the War of 1812 the Americans burned down the Parliament Buildings in
York (now Toronto). What did the British and Canadians do in return?');



insert into MultipleChoiceOption(qid, text, hint)
values 
(782, 'To pledge your loyalty to the Sovereign, Queen Elizabeth II', null),
(782, 'To pledge your allegiance to the flag and fulfill the duties of a Canadian', 'Think regally.'),
(782, 'To pledge your loyalty to Canada from sea to sea', null),
(625, 'The first railway to cross Canada', 'The Underground Railroad was generally south to north, not east-west.'),
(625, 'The CPR''s secret railway line', 'The Underground Railroad was secret, but it had nothing to do with trains.'),
(625, 'The TTC subway system', 'The TTC is relatively recent; the Underground Railroad was in operation over 100 years ago.'),
(625, 'A network used by slaves who escaped the United States into Canada', null),
(790, 'They attacked American merchant ships', null),
(790, 'They expanded their defence system, including Fort York', null),
(790, 'They burned down the White House in Washington D.C.', null),
(790, 'They captured Niagara Falls', null);



insert into MultipleChoiceQuestion(qid, answer)
values
(782, 'To pledge your loyalty to the Sovereign, Queen Elizabeth II'),
(625, 'A network used by slaves who escaped the United States into Canada'),
(790, 'They burned down the White House in Washington D.C.');


insert into TrueFalseQuestion(qid, answer)
values 
(566, false);


insert into NumericQuestion(qid, answer)
values 
(601, 1960);

insert into NumericQuestionHint(qid, lower_bound, upper_bound, text)
values 
(601, 1800, 1900, 'The Quiet Revolution happened during the 20th Century.'),
(601, 2000, 2010, 'The Quiet Revolution happened some time ago.'),
(601, 2020, 3000, 'The Quiet Revolution has already happened!');


insert into Quiz(id, title, due, class_id, hint_flag)
values 
('Pr1-220310', 'Citizenship Test Practise Questions', '2017-10-01 1:30:00', 0, true);


insert into QuizQuestion(id, quiz_id, qid, weight)
values 
(0, 'Pr1-220310', 601, 2),
(1, 'Pr1-220310', 566, 1),
(2, 'Pr1-220310', 790, 3),
(3, 'Pr1-220310', 625, 2);


insert into MultipleChoiceResponse(qqid, student_id, answer)
values 
(2,   998801234, 'They expanded their defence system, including Fort York'),
(2,    10784522, 'They burned down the White House in Washington D.C.'),
(2,   997733991, 'They burned down the White House in Washington D.C.'),
(2,  5555555555, 'They captured Niagara Falls'),
-- (2,  1111111111, null),
(3,   998801234, 'A network used by slaves who escaped the United States into Canada'),
(3,    10784522, 'A network used by slaves who escaped the United States into Canada'),
(3,   997733991, 'The CPR''s secret railway line');
-- (3,  5555555555, null),
-- (3,  1111111111, null),


insert into TrueFalseResponse(qqid, student_id, answer)
values 
(1,   998801234, False),
(1,    10784522, False),
(1,   997733991, True),
(1,  5555555555, False);
-- (1,  997733991,  null),


insert into NumericResponse(qqid, student_id, answer)
values 
(0,   998801234, 1950),
(0,    10784522, 1960),
(0,   997733991, 1960);
-- (0,  5555555555, null),
-- (0,  997733991, null),




-- 1. Report full name and student number of all students in the database 
create view q1_solution as 
select id as student_number, first_name||' '||last_name as full_name 
from Student;

-- 2. For all questions in the database, report 
--  question id, text, number of hints associated with it 
--  report null for True-False questions, since they cannot have hints 
create view MCQuestionHints as 
select q.id as question_id, q.text as question_text, count(o.hint) as num_hints
from Question q join MultipleChoiceQuestion mcq on q.id=mcq.qid
                join MultipleChoiceOption o on mcq.qid=o.qid
where q.type='multiple_choice'
group by q.id;


create view TFQuestionHints as 
select q.id as question_id, q.text as question_text, null as num_hints
from Question q join TrueFalseQuestion tfq on q.id=tfq.qid
where q.type='true_false';

-- 3 hints for 601
create view NQuestionHints as 
select q.id as question_id, q.text as question_text, count(*) as num_hints
from Question q join NumericQuestion nq on q.id=nq.qid
                join NumericQuestionHint nqh on q.id=nqh.qid
where q.type='numeric' 
group by q.id;


create view q2_solution as 
select * from MCQuestionHints union 
select * from TFQuestionHints union 
select * from NQuestionHints;


-- 3. compute total grade on quiz Pr1-220310 for every student in grade 8 in room 120 with mr higgins 
--  Report student number, last name and total grade

create view StudentQuizQuestions as 
select s.id as student_id, qq.id as qqid, qq.qid as qid, qq.weight as qqweight
from Student s join StudentInClass sic on s.id=sic.student_id 
               join Class c on c.id=sic.class_id
               join Quiz q on q.class_id=c.id
               join QuizQuestion qq on qq.quiz_id=q.id
where c.room=120 and c.grade=8 and c.teacher='Mr Higgins' 
    and q.id='Pr1-220310';



-- Compute weighted grade for MC response
create view MCCheckCorrect as 
select sqq.student_id, sqq.qid,
    case
      when mcr.answer is null       then 0
      when mcq.answer=mcr.answer    then sqq.qqweight 
      else 0
    end as mark
from StudentQuizQuestions sqq left join MultipleChoiceResponse mcr on sqq.student_id=mcr.student_id and sqq.qqid=mcr.qqid
                                   join MultipleChoiceQuestion mcq on sqq.qid=mcq.qid;

-- Compute weighted grade for true-false responses
create view TFCheckCorrect as 
select sqq.student_id, sqq.qid, 
    case
      when tfr.answer is null       then 0
      when tfq.answer=tfr.answer    then sqq.qqweight 
      else 0
    end as mark
from StudentQuizQuestions sqq left join TrueFalseResponse tfr on sqq.student_id=tfr.student_id and sqq.qqid=tfr.qqid
                                   join TrueFalseQuestion tfq on sqq.qid=tfq.qid;


-- Compute weighted grade for numeric responses
create view NumericCheckCorrect as 
select sqq.student_id, sqq.qid,
    case
      when nr.answer is null       then 0
      when nq.answer=nr.answer    then sqq.qqweight 
      else 0
    end as mark
from StudentQuizQuestions sqq left join NumericResponse nr on sqq.student_id=nr.student_id and sqq.qqid=nr.qqid
                                   join NumericQuestion nq on sqq.qid=nq.qid;

-- Union all question marks for the quiz
create view AllMarks as 
select * from MCCheckCorrect union 
select * from TFCheckCorrect union 
select * from NumericCheckCorrect;

-- Aggregate marks relation to compute the total grade for each student
create view TotalWeightedGrade as 
select student_id, sum(mark) as total_grade
from AllMarks am
group by am.student_id;

-- Add last name attribute
create view q3_solution as 
select student_id, s.last_name as last_name, total_grade
from TotalWeightedGrade am join Student s on am.student_id=s.id;



-- 4. For each student in grade 8 class in room 120 with Mr Higgins, and 
--      every question from quiz Pr1-220310 that they did not answer, 
--      report student id, question id, and question text 
create view StudentQuizQuestions as 
select s.id as student_id, qq.id as qqid, qq.qid as qid
from Student s join StudentInClass sic on s.id=sic.student_id 
               join Class c on c.id=sic.class_id
               join Quiz q on q.class_id=c.id
               join QuizQuestion qq on qq.quiz_id=q.id
where c.room=120 and c.grade=8 and c.teacher='Mr Higgins' 
    and q.id='Pr1-220310';

-- Find student that did answer multiple choice questions for the quiz 
create view MCDidDo as 
select q.student_id, q.qid
from StudentQuizQuestions q left join MultipleChoiceResponse res on 
    q.student_id=res.student_id and q.qqid=res.qqid
where res.answer is not null;

-- Find student that did answer true false questions for the quiz 
create view TFDidDo as 
select q.student_id, q.qid
from StudentQuizQuestions q left join TrueFalseResponse res on 
    q.student_id=res.student_id and q.qqid=res.qqid
where res.answer is not null;

-- Find student that did answer numeric questions for the quiz 
create view NDidDo as 
select q.student_id, q.qid
from StudentQuizQuestions q left join NumericResponse res on 
    q.student_id=res.student_id and q.qqid=res.qqid
where res.answer is not null;

-- Union (student_id, qid) pair such that student did do question 
create view AllDidDo as 
select * from MCDidDo union 
select * from TFDidDo union 
select * from NDidDo;

-- Find (student_id, qid) pair such that the student did not do question 
create view DidNotDoQuestion as 
(select student_id, qid from StudentQuizQuestions) 
except 
(select * from AllDidDo);

-- Add additional attribute question text 
create view q4_solution as 
select student_id, qid as question_id, q.text as question_text
from DidNotDoQuestion ddq join Question q on ddq.qid=q.id;



-- 5.  For each question on quiz Pr1-220310, report 
--      the number of student in grade 8 class in room 120 with Mr Higgins who got the question right, 
--      the number who got it wrong 
--      the number who did not answer it

create view StudentQuizQuestions as 
select s.id as student_id, qq.id as qqid, qq.qid as qid
from Student s join StudentInClass sic on s.id=sic.student_id 
               join Class c on c.id=sic.class_id
               join Quiz q on q.class_id=c.id
               join QuizQuestion qq on qq.quiz_id=q.id
where c.room=120 and c.grade=8 and c.teacher='Mr Higgins' 
    and q.id='Pr1-220310';


-- A tally of if the student got the multiple choice question right, wrong, or did not answer 
create view MCSummary as 
select sqq.student_id, sqq.qid,
    case
        when res.answer is null     then 1
        else 0
    end as did_not_answer,
    case
        when res.answer is null     then 0
        when q.answer=res.answer    then 1
        else 0
    end as answer_right,
    case
        when res.answer is null      then 0
        when q.answer!=res.answer    then 1
        else 0
    end as answer_wrong
from StudentQuizQuestions sqq left join MultipleChoiceResponse res on sqq.student_id=res.student_id and sqq.qqid=res.qqid
                                   join MultipleChoiceQuestion q   on sqq.qid=q.qid;

-- A tally of if the student got the true false question right, wrong, or did not answer 
create view TFSummary as 
select sqq.student_id, sqq.qid,
    case
        when res.answer is null     then 1
        else 0
    end as did_not_answer,
    case
        when res.answer is null     then 0
        when q.answer=res.answer    then 1
        else 0
    end as answer_right,
    case
        when res.answer is null      then 0
        when q.answer!=res.answer    then 1
        else 0
    end as answer_wrong
from StudentQuizQuestions sqq left join TrueFalseResponse res on sqq.student_id=res.student_id and sqq.qqid=res.qqid
                                   join TrueFalseQuestion q   on sqq.qid=q.qid;

-- A tally of if the student got the numeric question right, wrong, or did not answer 
create view NSummary as 
select sqq.student_id, sqq.qid,
    case
        when res.answer is null     then 1
        else 0
    end as did_not_answer,
    case
        when res.answer is null     then 0
        when q.answer=res.answer    then 1
        else 0
    end as answer_right,
    case
        when res.answer is null      then 0
        when q.answer!=res.answer    then 1
        else 0
    end as answer_wrong
from StudentQuizQuestions sqq left join NumericResponse res on sqq.student_id=res.student_id and sqq.qqid=res.qqid
                                   join NumericQuestion q   on sqq.qid=q.qid;


-- Union of the previous tallies for all questions 
create view AllSummary as 
select * from MCSummary union 
select * from TFSummary union 
select * from NSummary;


-- Aggregate to get sum of number of student who got it wrong, right, did not answer for each question
create view q5_solution as 
select qid, sum(did_not_answer) as num_did_not_answer,
            sum(answer_right)   as num_answer_correct,
            sum(answer_wrong)   as num_answer_incorrect
from AllSummary
group by qid;