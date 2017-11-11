-- Left-right

SET SEARCH_PATH TO parlgov;
drop table if exists q4 cascade;

-- You must not change this table definition.

create table q4(
    countryName VARCHAR(50),
    r0_2 INT,
    r2_4 INT,
    r4_6 INT,
    r6_8 INT,
    r8_10 INT
);

-- You may find it convenient to do this for each of the views
-- that define your intermediate steps.  (But give them better names!)
drop view if exists joined_table cascade;


-- join tables to get countryname and the left_right value
create view joined_table as
select 
    c.name as countryName, 
    pp.left_right as left_right, 
    p.name as partyName
from country c join party p on c.id=p.country_id
               join party_position pp on p.id=pp.party_id;


--  select * from joined_table order by countryName, left_right;

-- view that tallies party count for each bucket
create view in_buckets as
select
    countryName, 
    count(case when left_right >= 0 and left_right < 2 then 1 end) as r0_2, 
    count(case when left_right >= 2 and left_right < 4 then 1 end) as r2_4, 
    count(case when left_right >= 4 and left_right < 6 then 1 end) as r4_6, 
    count(case when left_right >= 6 and left_right < 8 then 1 end) as r6_8, 
    count(case when left_right >= 8 and left_right < 10 then 1 end) as r8_10
from joined_table
group by countryName;


-- the answer to the query 
insert into q4
select * from in_buckets


