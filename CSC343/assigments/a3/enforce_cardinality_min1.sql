drop schema if exists circularCourses cascade; 
create schema circularCourses; 
set search_path to circularCourses;

-- Here is a very trimmed down version of our Courses database
CREATE TABLE Student( 
    sID integer primary key, 
    other varchar(5) );

CREATE TABLE Offering ( 
    oID integer primary key, 
    other varchar(5) );

CREATE TABLE Took ( 
    sID integer, 
    oID integer, 
    grade integer, 
    primary key (sID, oID), 
    foreign key (sID) references Student, 
    foreign key (oID) references Offering);

-- Suppose we change a domain rule so that a -- student must take a mimimum of 1 offering
-- In RA we write: Student[sid] \subseteq Took[sid]
-- Is that enforceable in SQL at all?
-- Inside Student we can't write "foreign key (sID) references Took" because 
-- by default, this assumes we are referencing the key of Took, and 
-- the key of Took is BOTH sID and oID. 
-- We have to be more specific about what we are referencing. 
-- This would be expressed by writing "foreign key (sID) references Took(sID)".
-- With this additional constraint, we can't define Student before Took is defined. 
-- But we can't define Student after Took either, because Took also 
-- references Student. In short, neither table can come first.
-- There is a workaround: We can leave out one of the constraints, allowing 
-- one table to be defined before the other, and then add the constraint 
-- after both tables have been defined. Here we add the constraint to the 
-- Student table:
ALTER TABLE Student ADD foreign key (sID) references Took(sID);
-- But we still have a problem: sID is not unique in Took, so we can't reference 
-- it in another table. We get this error when we try to add the constraint: 
-- ERROR: there is no unique constraint matching given keys for referenced table "took"
-- Conclusion -- you can't use the foreign key concept in SQL to enforce -- the fact that a student must take at least one offering.