# solutions.py
"""SQL 2.
Jonathan Merrill
"""
import sqlite3 as sql
import itertools


def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #find the students who have a B in any course
            cur.execute("SELECT SI.StudentName FROM StudentInfo as SI INNER JOIN StudentGrades as SG ON SI.StudentID == SG.StudentID WHERE SG.Grade == 'B'")
            #return the list
            A = cur.fetchall()
    finally:
        conn.close()
    #convert the tuple into a list
    return list(itertools.chain(*A))


def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #find the students grade in calculus if they are in calculus
            cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade FROM StudentInfo as SI LEFT OUTER JOIN MajorInfo as MI ON SI.MajorID == MI.MajorID INNER JOIN StudentGrades as SG ON SI.StudentID == SG.StudentID WHERE SG.CourseID == 1")
            A = cur.fetchall()
    finally:
        conn.close()
    #convert and return it into a list
    return A



def prob3(db_file="students.db"):
    """Query the database for the list of the names of courses that have at
    least 5 students enrolled in them.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        ((list): a list of strings, each of which is a course name.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #find the list of classes with more than 5 people in them
            cur.execute("SELECT CI.CourseName FROM StudentGrades as SG INNER JOIN CourseInfo as CI ON SG.CourseID == CI.CourseID GROUP BY SG.CourseID HAVING COUNT(*) >= 5")
            A = cur.fetchall()
    finally:
        conn.close()
    #convert and return the list (from a tuple)
    return list(itertools.chain(*A))



def prob4(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #i'll be honest, these comments seem redundant. Read the docstring above
            cur.execute("SELECT MI.MajorName, COUNT(*) as num_students FROM MajorInfo as MI INNER JOIN StudentInfo as SI ON MI.MajorID == SI.MajorID GROUP BY MI.MajorID ORDER BY num_students ASC")
            A = cur.fetchall()
    finally:
        conn.close()
    #convert the tuple into a list and return 
    return A


def prob5(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, MajorName) where
    the last name of the specified student begins with the letter C.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #see docstring for description
            cur.execute("SELECT SI.StudentName, MI.MajorName FROM StudentInfo as SI LEFT OUTER JOIN MajorInfo as MI ON SI.MajorID == MI.MajorID WHERE StudentName LIKE '% C%';")
            A = cur.fetchall()
    finally:
        conn.close()
    #convert from tuple to list and return 
    return A


def prob6(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name, COUNT(*) as num_courses, AVG(gpa) " #return the name, number of courses, and gpa
                        "FROM (SELECT SI.StudentName as Name, CASE SG.Grade " #sub select
                        #write the point statements for the gpa numbers
                        "WHEN 'A+' THEN 4.0 "
                        "WHEN 'A' THEN 4.0 "
                        "WHEN 'A-' THEN 3.7 "
                        "WHEN 'B+' THEN 3.4 "
                        "WHEN 'B' THEN 3.0 "
                        "WHEN 'B-' THEN 2.7 "
                        "WHEN 'C+' THEN 2.4 "
                        "WHEN 'C' THEN 2.0 "
                        "WHEN 'C-' THEN 1.7 "
                        "WHEN 'D+' THEN 1.4 "
                        "WHEN 'D' THEN 1.0 "
                        "ELSE 0.7 END as gpa "
                        #Left outer join student grades and student info
                        "FROM StudentGrades as SG LEFT OUTER JOIN StudentInfo as SI "
                        "ON SI.StudentID == SG.StudentID) "  #end of sub category
                    "GROUP BY name "  #group by name
                    "ORDER BY gpa DESC ")  #order in descending order by gpa
            A = cur.fetchall()
    finally:
        conn.close()
    return A  #return the list of tuples
            


