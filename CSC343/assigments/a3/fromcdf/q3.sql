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



-- display
select * from q3_solution;