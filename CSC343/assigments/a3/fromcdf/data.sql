
-- Populate tables
insert into Student(id, first_name, last_name)
values 
('0998801234', 'Lena', 'Headey'),
('0010784522', 'Peter', 'Dinklage'),
('0997733991', 'Emilia', 'Clarke'),
('5555555555', 'Kit', 'Harrington'),
('1111111111', 'Sophie', 'Turner'),
('2222222222', 'Maisie', 'Williams');


insert into Class(id, room, grade, teacher)
values 
(0, 120, 8, 'Mr Higgins'),
(1, 366, 5, 'Miss Nyers');

insert into StudentInClass(student_id, class_id)
values
('0998801234', 0),
('0010784522', 0),
('0997733991', 0),
('5555555555', 0),
('1111111111', 0),
('2222222222', 1);



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
(2,  '0998801234', 'They expanded their defence system, including Fort York'),
(2,  '0010784522', 'They burned down the White House in Washington D.C.'),
(2,  '0997733991', 'They burned down the White House in Washington D.C.'),
(2,  '5555555555', 'They captured Niagara Falls'),
-- (2,  '1111111111', null),
(3,  '0998801234', 'A network used by slaves who escaped the United States into Canada'),
(3,  '0010784522', 'A network used by slaves who escaped the United States into Canada'),
(3,  '0997733991', 'The CPR''s secret railway line');
-- (3,  '5555555555', null),
-- (3,  '1111111111', null),


insert into TrueFalseResponse(qqid, student_id, answer)
values 
(1,  '0998801234', False),
(1,  '0010784522', False),
(1,  '0997733991', True),
(1,  '5555555555', False);
-- (1, '0997733991',  null),


insert into NumericResponse(qqid, student_id, answer)
values 
(0,  '0998801234', 1950),
(0,  '0010784522', 1960),
(0,  '0997733991', 1960);
-- (0,  '5555555555', null),
-- (0, '0997733991', null),
