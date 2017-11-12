--  select * from q2 order by countryName, wonElections, partyName;
select * 
from election, election_result on election.id=election_result.election_id;
