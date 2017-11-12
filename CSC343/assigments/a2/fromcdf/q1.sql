
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

-- election where e_date is within 1996 to 2016 inclusive
create view election10y as 
select * 
from election e
where date_part('year', e.e_date) >= 1996 and date_part('year', e.e_date) <= 2016;


-- calculate percentage votes
create view votesPercent as
select 
    e.id as election_id, 
    date_part('year', e.e_date) as year,
    c.name as countryName,
    p.name_short as partyName,
    r.votes ::numeric / e.votes_valid as vp
from election10y e join election_result r on e.id=r.election_id
                   join party p on r.party_id=p.id
                   join country c on e.country_id=c.id;

-- average average votes percentage for each year, countryName and partyName combination
create view votesPercentAvg as
select 
    year, 
    countryName, 
    avg(vp) as vpavg,
    partyName
from votesPercent
group by year, countryName, partyName;


-- put votes percent into buckets
-- filters such that only country with votes percent are left
create view votesPercentInBucket as
select
    year, 
    countryName, 
    case when vpavg > 0    and vpavg <= 0.05 then  '(0-5]'
         when vpavg > 0.05 and vpavg <= 0.10 then '(5-10]'
         when vpavg > 0.10 and vpavg <= 0.20 then '(10-20]'
         when vpavg > 0.20 and vpavg <= 0.30 then '(20-30]'
         when vpavg > 0.30 and vpavg <= 0.40 then '(30-40]'
         when vpavg > 0.40 and vpavg <= 1.00 then '(40-100]' end as voteRange,
    partyName 
from votesPercentAvg
where vpavg is not null;

 
-- insert to answer
insert into q1
select * from votesPercentInBucket;





