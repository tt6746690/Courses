SET SEARCH_PATH TO parlgov;
drop table if exists q7 cascade;

-- You must not change this table definition.

DROP TABLE IF EXISTS q7 CASCADE;
CREATE TABLE q7(
    countryId INT, 
    alliedPartyId1 INT, 
    alliedPartyId2 INT

);



-- Find all alliances in all elections
-- count(*) = 276
DROP VIEW IF EXISTS AllAlliances CASCADE;
CREATE VIEW AllAlliances AS 
select 
    e.country_id as countryId, 
    e1.id as alliedPartyId1, 
    e2.id as alliedPartyId2, 
    count(e1.election_id) as pairsAmount
from 
    election_result e1 join election_result e2 on e1.election_id=e2.election_id
                       join election e on e.id=e1.election_id
where 
    e1.id < e2.id and 
    (e1.alliance_id = e2.id or e2.alliance_id = e1.id)
group by e.country_id, alliedPartyId1, alliedPartyId2;


-- Calculate the total number of elections in each country
DROP VIEW IF EXISTS electionStats CASCADE;
CREATE VIEW electionStats AS
select 
    country_id, 
    count(id) as electionNum
from election
group by country_id;

-- Final answer
DROP VIEW IF EXISTS FinalAnswer CASCADE;
CREATE VIEW FinalAnswer AS
select 
    countryId, 
    alliedPartyId1, 
    alliedPartyId2
from 
    AllAlliances join electionStats on AllAlliances.countryId=electionStats.country_id
where 
    pairsAmount >= 0.3 * electionNum;


-- the answer to the query 
--  insert into q7
--  (select * from FinalAnswer);



