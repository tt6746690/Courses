SET SEARCH_PATH TO parlgov;


select 
    e.id as election_id,   -- elections in country whose name is specified
    cab.id as cabinet_id  -- the cabinets that formed after each election

    --  e.e_type as election_type,
    --  e.e_date as election_date,          -- sanity checking if ordered by election date desc
    --  cab.start_date as cabinet_start      -- sanity checking if for multiple cabinets, their startdate is before the next election
from election e join country c on e.country_id=c.id
                join cabinet cab on e.id=cab.election_id
where c.name='United Kingdom'
order by e.e_date desc;

