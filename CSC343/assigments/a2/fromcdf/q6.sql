SET SEARCH_PATH TO parlgov;
drop table if exists q6 cascade;

-- You must not change this table definition.

CREATE TABLE q6(
    countryName VARCHAR(50),
    cabinetId INT, 
    startDate DATE,
    endDate DATE,
    pmParty VARCHAR(100)
);

-- You may find it convenient to do this for each of the views
-- that define your intermediate steps.  (But give them better names!)
drop view if exists cabinet_with_pmparty cascade;
drop view if exists cabinet_with_startend cascade;

-- find PM's party if exists
-- count(*) = 286, note cabinet has 289 rows, some cabinet pmParty is null
create view cabinet_with_pmparty as
select 
    c.id as cabinet_id,
    p.party_id as pm_party
from cabinet c join cabinet_party p on c.id=p.cabinet_id
where p.pm = TRUE;

-- use start_date of current cabinet to end_date of previous cabinet
-- count(*) = 284
create view cabinet_with_startend as 
select
    prev.id as cabinet_id,
    cur.start_date as end_date
from
    cabinet cur join cabinet prev
    on cur.previous_cabinet_id = prev.id;

-- add pm party info to cabinet table
-- note 3 rows have null pmparty which is OK
create view cabinet_added_attr as 
select 
    country.name as countryName,
    c.id as cabinetId,
    c.start_date as startDate,
    s.end_date as endDate,
    party.name as pmParty
from cabinet c left join cabinet_with_pmparty p on c.id=p.cabinet_id
               left join cabinet_with_startend s on c.id=s.cabinet_id
               join country on c.country_id=country.id
               join party on p.pm_party=party.id;

-- the answer to the query 
insert into q6
select * from cabinet_added_attr;

