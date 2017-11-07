-- The Liberal, Green, and Rhinocerous parties of Canada formed an alliance in an election in 2020.
-- In that same election, the NDP and Conservative parties formed an alliance also.

INSERT INTO country(id, name, oecd_accession_date)
VALUES
(1, 'Canada', '2015-11-27');

INSERT INTO party(id, country_id, name_short, name)
VALUES
(1, 1, 'LIB', 'Liberal Party'),
(2, 1, 'GRN', 'Green Party'),
(3, 1, 'RHI', 'Rhinocerous Party'),
(4, 1, 'NDP', 'National Democratic Party'),
(5, 1, 'CON', 'Convervative Party');


INSERT INTO election(id, country_id, e_date, seats_total, electorate, e_type)
VALUES
(1, 1, '2020-11-27', 300, 250, 'Parliamentary election');


INSERT INTO election_result(id, election_id, party_id, alliance_id)
VALUES
(1, 1, 1, NULL),
(2, 1, 2, 1),
(3, 1, 3, 1),
(4, 1, 4, NULL),
(5, 1, 5, 4);