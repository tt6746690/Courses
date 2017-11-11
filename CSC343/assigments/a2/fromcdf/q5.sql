-- Committed

SET SEARCH_PATH TO parlgov;
drop table if exists q5 cascade;

-- You must not change this table definition.

CREATE TABLE q5(
    countryName VARCHAR(50),
    partyName VARCHAR(100),
    partyFamily VARCHAR(50),
    stateMarket REAL
);

DROP VIEW IF EXISTS Cabinets CASCADE;

-- Constructe a view to store all cabinets over the past 20 years
CREATE VIEW Cabinets AS
select country_id, cabinet_id, party_id 
from cabinet, cabinet_party 
where cabinet_id = cabinet.id and start_date >= '1986-01-01';

-- Prepare a check list that records all combinations of cabinets 
-- and parties in some country
CREATE VIEW ExpectedCommitted AS
select c.country_id as country_id, cabinet_id, id as party_id
from Cabinets c, party
where c.country_id = party.country_id;

-- Construct a table to store those parties that are not committed parties
CREATE VIEW Notcommitted AS
((select * from ExpectedCommitted ) except (select * from Cabinets));

-- Find all committed parties
CREATE VIEW CommittedParties AS 
((select * from Cabinets) except (select * from NotCommitted));


-- Final answer
CREATE VIEW FinalAnswer AS
select
    country.name as countryName, 
    party.name as partyName, 
    pf.family as partyFamily, 
    party_position.state_market as stateMarket
from country, party, party_position, CommittedParties c, party_family pf
where c.party_id = party.id and c.country_id = country.id 
	and c.party_id = party_position.party_id and 
    pf.party_id = party.id;

-- the answer to the query 
insert into q5 
(select * from FinalAnswer);


