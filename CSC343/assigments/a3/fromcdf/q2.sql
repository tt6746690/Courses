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


-- display
select * from q2_solution;