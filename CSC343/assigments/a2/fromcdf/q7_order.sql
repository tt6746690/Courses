--  select * from q7 order by countryID DESC, alliedPartyID1 DESC, alliedPartyID2 DESC;


select 
    e.country_id as countryId, 
    e1.id as alliedPartyId1, 
    e2.id as alliedPartyId2, 
    e1.election_id
from 
    election_result e1 join election_result e2 on e1.election_id=e2.election_id
                       join election e on e.id=e1.election_id
where 
    e1.id < e2.id and 
    (e1.alliance_id = e2.id or e2.alliance_id = e1.id);

