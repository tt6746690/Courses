-- 5.  For each question on quiz Pr1-220310, report 
--      the number of student in grade 8 class in room 120 with Mr Higgins who got the question right, 
--      the number who got it wrong 
--      the number who did not answer it

create view StudentQuizQuestions5 as 
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
from StudentQuizQuestions5 sqq left join MultipleChoiceResponse res on sqq.student_id=res.student_id and sqq.qqid=res.qqid
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
from StudentQuizQuestions5 sqq left join TrueFalseResponse res on sqq.student_id=res.student_id and sqq.qqid=res.qqid
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
from StudentQuizQuestions5 sqq left join NumericResponse res on sqq.student_id=res.student_id and sqq.qqid=res.qqid
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


-- display
select * from q5_solution;