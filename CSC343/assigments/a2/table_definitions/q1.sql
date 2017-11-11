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


-- You may find it convenient to do this for each of the views
-- that define your intermediate steps.  (But give them better names!)
DROP VIEW IF EXISTS intermediate_step CASCADE;

-- Define views for your intermediate steps here.


-- the answer to the query 
insert into q1 

