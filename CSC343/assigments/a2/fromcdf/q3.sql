-- Participate

SET SEARCH_PATH TO parlgov;
drop table if exists q3 cascade;

-- You must not change this table definition.

create table q3(
    countryName varchar(50),
    year int,
    participationRatio real
);

-- You may find it convenient to do this for each of the views
-- that define your intermediate steps.  (But give them better names!)
drop view if exists election_aug cascade;
drop view if exists election_avg cascade;
drop view if exists election_filtered cascade;
drop view if exists not_non_desc_country cascade;
drop view if exists election_non_desc_ratio cascade;

-- augment table election with participation ratio and year
-- count(*) = 141
create view election_aug as 
select country_id,
       date_part('year', e.e_date) as year, 
       e.votes_cast::float  / e.electorate as participationRatio
from election e
where e.votes_cast is not null; -- need to filter or not?

-- averaged participation ratio over country + year combination
-- count(*) = 136
create view election_avg as 
select country_id,
       year,
       avg(participationRatio) as participationRatio
from election_aug e
group by country_id, year;

-- countries with at least one election from 2001 to 2016
-- count(*)
create view election_filtered as 
select * 
from election_avg
where year>=2001 and year<=2016;

-- countries whose averaged participation ratio are not monotonically non-desc
-- count(*) = 5
create view not_non_desc_country as 
select e1.country_id as country_id
from election_filtered e1, election_filtered e2
where e1.country_id = e2.country_id and 
      e1.year < e2.year and 
      e1.participationRatio > e2.participationRatio
group by e1.country_id;


-- for inspecting non-desc 
-- select * from election_filtered order by country_id, year;

-- remove the not 
create view election_non_desc_ratio as
select * 
from election_filtered e
where e.country_id NOT IN (
    select * from not_non_desc_country
);


-- replace country id with coutry name
create view election_non_desc_ratio_withname as 
select 
    c.name as countryName, 
    x.year as year, 
    x.participationRatio as participationRatio
from election_non_desc_ratio x join country c on x.country_id = c.id;




-- the answer to the query 
insert into q3
select * from election_non_desc_ratio_withname;


