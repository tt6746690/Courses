-- Winners

SET SEARCH_PATH TO parlgov;
drop table if exists q2 cascade;

-- You must not change this table definition.

create table q2(
    countryName VARCHaR(100),
    partyName VARCHaR(100),
    partyFamily VARCHaR(100),
    wonElections INT,
    mostRecentlyWonElectionId INT,
    mostRecentlyWonElectionYear INT
);


DROP VIEW IF EXISTS maxVotes CASCADE;

-- Figure out winners' votes in each election
CREATE VIEW maxVotes AS
SELECT election_id as e_id, max(votes) as maximum
FROM election_result
GROUP BY e_id;

-- Winners' table
CREATE VIEW Winners AS
SELECT e_id, party_id as p_id
FROM maxVotes, election_result
WHERE e_id = election_id;

-- Calculate the correponding amounts of each winners' winning elections
CREATE VIEW WinningAmount AS
SELECT p_id, count(e_id) as winNum
FROM Winners
GROUP BY p_id;

-- Figure out the totel wins in each country
CREATE VIEW CountryWinningAmount AS
SELECT 
    country_id as c_id, 
    count(winNum) as cWinNum, 
    count(p_id) as cWinPNum
FROM 
    WinningAmount, party
WHERE p_id = party.id
GROUP BY c_id;

-- Find parties that have won more than 3 times the average number of 
-- wining elections of parties of the same country
CREATE VIEW Dominants AS
SELECT c_id, p_id
FROM CountryWinningAmount, WinningAmount, party
WHERE p_id = party.id and c_id = country_id and winNum > 3.0 * cWinNum / cWinPNum;

-- Find the most recently won election date of above dominants
CREATE VIEW mostRecentlyWonDate AS
SELECT 
    d.p_id AS partyID, 
    max(e_date) as nearestWon
FROM Winners w, Dominants d, election e
WHERE w.p_id = d.p_id and e_id = e.id
group by d.p_id;

-- Find the most recently won election id and year of above dominants
CREATE VIEW mostRecentlyWon AS
SELECT partyID as p_id, id as mostRecentlyWonElectionId, 
	date_part('year', nearestWon) as mostRecentlyWonElectionYear
FROM mostRecentlyWonDate, election
WHERE nearestWon = e_date;

-- Generate the final view before insertion
CREATE VIEW FinalAnswer AS
Select 
    country.name as countryName, 
    party.name as partyName, 
    pf.family as partyFamily, 
    winNum as wonElections, 
	mostRecentlyWonElectionId, 
    mostRecentlyWonElectionYear
FROM country, party, WinningAmount, mostRecentlyWon m, party_family pf
WHERE 
    m.p_id = party.id and 
    m.p_id = WinningAmount.p_id and 
    country.id = party.country_id and 
    pf.party_id = party.id;

-- the answer to the query 
insert into q2 
(select * from FinalAnswer);



