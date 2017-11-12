-- VoteRange

SET SEARCH_PATH TO parlgov;
drop table if exists q1 cascade;

-- You must not change this table definition.

create table q1(
    year INT,
    countryName VARCHAR(50),
    voteRange VARCHAR(20),
    partyName VARCHAR(100)
);

DROP VIEW IF EXISTS ElectionResults CASCADE;

-- Select years where at least an election has happened in the past 10 years
CREATE VIEW ElectionResults AS
SELECT 
    date_part('year', e_date) as year, 
    election.id as e_id, 
    election_result.id as r_id,
    country_id, 
    party_id
FROM Election, Election_result
WHERE 
    election.id = election_id and 
    1996 <= date_part('year', e_date) and date_part('year', e_date) <= 2016;

-- Collect votes data for elections in the past 10 years
CREATE VIEW VotesResults AS
SELECT 
    r_id, 
    votes_valid, 
    votes,
    votes::float / votes_valid as ratio
FROM Election, Election_result, ElectionResults
WHERE e_id = election.id and r_id = election_result.id ;

-- Select election results which fall in vote range '(0,5]'
CREATE VIEW bracket1 AS
select 
    r_id, 
    '(0,5]'::varchar(50) as voteRange
from VotesResults
Where 0 < ratio and ratio <=0.05;

-- Select election results which fall in vote range '(5,10]'
CREATE VIEW bracket2 AS
select 
    r_id, 
    '(5,10]'::varchar(50) as voteRange
from VotesResults
Where 0.05 < ratio and ratio <=0.10;

-- Select election results which fall in vote range '(10,20]'
CREATE VIEW bracket3 AS
select 
    r_id, 
    '(10,20]'::varchar(50) as voteRange
from VotesResults
Where 0.10 < ratio and ratio <= 0.20;

-- Select election results which fall in vote range '(20,30]'
CREATE VIEW bracket4 AS
select 
    r_id, 
    '(20,30]'::varchar(50) as voteRange
from VotesResults
Where 0.20 < ratio and ratio <=0.30;

-- Select election results which fall in vote range '(30,40]'
CREATE VIEW bracket5 AS
select 
    r_id, 
    '(30,40]'::varchar(50) as voteRange
from VotesResults
Where 0.30 < ratio and ratio <=0.40;

-- Select election results which fall in vote range '(40,100]'
CREATE VIEW bracket6 AS
select 
    r_id, 
    '(40,100]'::varchar(50) as voteRange
from VotesResults
Where 0.40 < ratio and ratio <= 1.00;

-- Union all vote range groups together
CREATE VIEW voteRangeGroups AS
select * from bracket1 union 
select * from bracket2 union 
select * from bracket3 union 
select * from bracket4 union 
select * from bracket5 union 
select * from bracket6;

-- Final answer before insertion
CREATE VIEW FinalAnswer AS
select 
    year, 
    c.name as countryName, 
    voteRange, 
    p.name as partyName
From 
    ElectionResults er, 
    Country c, 
    voteRangeGroups v,
    party p
WHERE 
    er.country_id = c.id and 
    er.party_id = p.id and 
    v.r_id = er.r_id;


--  select * from FinalAnswer;

-- the answer to the query 
insert into q1
(select * from FinalAnswer);


