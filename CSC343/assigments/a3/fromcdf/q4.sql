-- 4. For each student in grade 8 class in room 120 with Mr Higgins, and 
--      every question from quiz Pr1-220310 that they did not answer, 
--      report student id, question id, and question text 
create view StudentQuizQuestions4 as 
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
from StudentQuizQuestions4 q left join MultipleChoiceResponse res on 
    q.student_id=res.student_id and q.qqid=res.qqid
where res.answer is not null;

-- Find student that did answer true false questions for the quiz 
create view TFDidDo as 
select q.student_id, q.qid
from StudentQuizQuestions4 q left join TrueFalseResponse res on 
    q.student_id=res.student_id and q.qqid=res.qqid
where res.answer is not null;

-- Find student that did answer numeric questions for the quiz 
create view NDidDo as 
select q.student_id, q.qid
from StudentQuizQuestions4 q left join NumericResponse res on 
    q.student_id=res.student_id and q.qqid=res.qqid
where res.answer is not null;

-- Union (student_id, qid) pair such that student did do question 
create view AllDidDo as 
select * from MCDidDo union 
select * from TFDidDo union 
select * from NDidDo;

-- Find (student_id, qid) pair such that the student did not do question 
create view DidNotDoQuestion as 
(select student_id, qid from StudentQuizQuestions4) 
except 
(select * from AllDidDo);

-- Add additional attribute question text 
create view q4_solution as 
select student_id, qid as question_id, q.text as question_text
from DidNotDoQuestion ddq join Question q on ddq.qid=q.id;



-- display
select * from q4_solution;