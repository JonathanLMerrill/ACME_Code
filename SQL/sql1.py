# sql1.py
"""SQL Introduction
Jonathan Merrill
"""
import sqlite3 as sql
import csv
from matplotlib import pyplot as plt
import numpy as np


def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file.
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #delete said tables if they exist
            cur.execute("DROP TABLE IF EXISTS MajorInfo");  
            cur.execute("DROP TABLE IF EXISTS CourseInfo");
            cur.execute("DROP TABLE IF EXISTS StudentInfo");
            cur.execute("DROP TABLE IF EXISTS StudentGrades");
            #create said tables 
            cur.execute("CREATE TABLE MajorInfo(MajorID INTEGER, MajorName TEXT)");
            cur.execute("CREATE TABLE CourseInfo(CourseID INTEGER, CourseName TEXT)");
            cur.execute("CREATE TABLE StudentInfo(StudentID INTEGER, StudentName TEXT, MajorID INTEGER)");
            cur.execute("CREATE TABLE StudentGrades(StudentID INTEGER, CourseID INTEGER, Grade TEXT)");
    finally:
        conn.close()
    
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #create the MajorInfo table
            rows = [(1,'Math'),(2,'Science'),(3,'Writing'),(4,'Art')]
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?)",rows)
            #create the CourseInfo table
            rows = [(1,'Calculus'),(2,'English'),(3,'Pottery'),(4,'History')]
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?)",rows)
            #open the files and create the table StudentInfo with the data
            with open("student_info.csv", 'r') as infile:
                rows = list(csv.reader(infile))
                cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?)",rows)
            #open the file and create the table StudentGrades with the data
            with open("student_grades.csv", 'r') as infile:
                rows = list(csv.reader(infile))
                cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?)",rows)
    finally: 
        conn.close()
        
    try: 
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #replace the -1 entries with NULL
            cur.execute("UPDATE StudentInfo SET MajorID = NULL WHERE MajorID == -1")
    finally:
        conn.close()

        
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    try: 
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #delete the table if it exists
            cur.execute("DROP TABLE IF EXISTS USEarthquakes")
            #create the table USEarthquakes
            cur.execute("CREATE TABLE USEarthquakes(Year INTEGER, Month INTEGER, Day INTEGER,Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL)")
            #open the file and create a table using the data
            with open("us_earthquakes.csv", 'r') as infile:
                rows = list(csv.reader(infile))
                cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?)",rows)
    finally:
        conn.close()
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #delete the earthquakes with magnitude 0 from the table
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude == 0")
            #change the 0 values in Day, Hour, Minute, and Second to NULL
            cur.execute("UPDATE USEarthquakes SET Day = NULL WHERE Day == 0")
            cur.execute("UPDATE USEarthquakes SET Hour = NULL WHERE Hour == 0")
            cur.execute("UPDATE USEarthquakes SET Minute = NULL WHERE Minute == 0")
            cur.execute("UPDATE USEarthquakes SET Second = NULL WHERE Second == 0")
    finally:
         conn.close()


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Select all the Student names and course names for all the classes in which a student has an A or an A+
            cur.execute("SELECT SI.StudentName, CI.CourseName FROM CourseInfo as CI, StudentInfo as SI, StudentGrades as SG WHERE CI.CourseID == SG.CourseID AND SI.StudentID == SG.StudentID AND (SG.Grade == 'A' OR SG.Grade == 'A+')")
            #set the result as a list and return it
            A = cur.fetchall()
    finally:
        conn.close()
    return A


def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #get a list of tuples of the magnitude of the earthquakes from 1800-1899
            cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year >= 1800 AND Year < 1900")
            nineteen = cur.fetchall()
            #get a list of tuples of the magnitude of the earthquakes from 1900-1999
            cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year >= 1900 AND Year < 2000")
            twenty = cur.fetchall()
            #find the average magnitude of all earthquakes
            cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes")
            average = cur.fetchall()
    finally:
        conn.close()
    
    #create a histogram for the magnitude of earthquakes in the 19th century
    A1 = plt.subplot(121)
    A1.title.set_text('Magnitude of Earthquakes in the 19th Century            ')
    A1.set_xlabel('Magnitude')
    A1.set_ylabel('Number of Earthquakes')
    A1.hist(np.ravel(nineteen))
    #create a histogram for the magnitude of the earthquakes in the 20th century
    A2 = plt.subplot(122)
    A2.title.set_text('          20th Century')
    A2.set_xlabel('Magnitude')
    A2.hist(np.ravel(twenty))
    #return the averge magnitude of all earthquakes
    return average[0][0]
    #raise NotImplementedError("Problem 6 Incomplete")

    
    
#test functions
def test1():
    student_db()
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM StudentInfo;")
        print([d[0] for d in cur.description])

def test2():
    student_db()
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM MajorInfo;"):
            print(row)
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM CourseInfo;"):
            print(row)
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM StudentInfo;"):
            print(row)
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM StudentGrades;"):
            print(row)
            
def test3():
    earthquakes_db()
    with sql.connect("earthquakes.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM USEarthquakes;"):
            print(row)         
            

