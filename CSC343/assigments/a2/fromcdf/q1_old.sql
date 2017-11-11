
-- VoteRange

SET SEARCH_PATH TO parlgov;
drop table if exists q1 cascade;

-- You must not change this table definition.

create table q1(
    year            INT,
    countryName     VARCHAR(50),
    voteRange       VARCHAR(20),
    partyName       VARCHAR(100)
);


-- You may find it convenient to do this for each of the views
-- that define your intermediate steps.  (But give them better names!)
DROP VIEW IF EXISTS election10y CASCADE;
DROP VIEW IF EXISTS solution CASCADE;

-- election where e_date is within 1996 to 2016 inclusive
create view election10y as 
select * 
from election e
where date_part('year', e.e_date) >= 1996 and date_part('year', e.e_date) <= 2016;


create view votesPercent as
select e.id as election_id, 
       date_part('year', e.e_date) as year,
       c.name as countryName,
       p.name as partyName,
       r.votes ::numeric / e.votes_valid as vp
from election10y e join election_result r on e.id=r.election_id
                   join party p on r.party_id=p.id
                   join country c on e.country_id=c.id;

select election_id, year, countryName, partyName,
       case when vp>0  and vp<=5   then "(0, 5]"
            when vp>5  and vp<=10  then "(5, 10]"
            when vp>10 and vp<=20  then "(10, 20]"
            when vp>20 and vp<=30  then "(20, 30]"
            when vp>30 and vp<=40  then "(30, 40]"
            else "(40, 100]"
       end voteRange
from votesPercent;




