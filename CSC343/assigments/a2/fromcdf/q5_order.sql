--  select * from q5 order by countryName, partyName, stateMarket DESC;

select * 
from cabinet c join cabinet_party cp on c.id=cp.cabinet_id
               join party p on c.country_id=p.country_id;

