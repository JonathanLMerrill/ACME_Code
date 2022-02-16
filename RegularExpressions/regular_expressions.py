# regular_expressions.py
"""Regular Expressions Package
Jonathan Merrill
"""

import re
from collections import defaultdict as dd


def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile("python")


def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$")


def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")



def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return  re.compile(r"^[a-zA-Z|_][\w|_]*\s*(=\s*(\d*\.?\d*|'[^']*'|[a-zA-Z|_][\w|_]*))?$")
                       
    

def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    words = ['if', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'def', 'class']
    lines = code.split("\n")  #split the word file by the lines
    for i in range(len(lines)):     #for each line
        for j in range(len(words)):   #check each word in the test words
            A = re.compile(words[j])
            if bool(A.search(lines[i])) == True:   #if there is not a colon to finish the statement, then add one to the line
                lines[i] += ":"
    return '\n'.join(lines)  #return the lines put back together
         

    
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    file = open(filename)
    code = file.read()
    lines = code.split("\n")
    lines.pop()
    names = []
    bday = []
    email = []
    phone = []
    
    for i in range(len(lines)):
        #the compiling codes to isolate and extract the specific pieces of information
        pattern = re.compile("(\S*@\S*)")
        pattern2 = re.compile("(\d*)/(\d*)/(\d*)")
        pattern3 = re.compile(r"\(*(\d{3,3})[\)|-]*(\d{3,3})-(\d{4,4})")
        pattern4 = re.compile(r"^[a-zA-Z]*\s*[A-Z]{0,1}\.{0,1}\s[a-zA-Z]*")
        
        #add all the emails to a list (adding None if no email is listed)
        if pattern.findall(lines[i]) == []:
            email.append(None)
        else:
            email.append(pattern.findall(lines[i])[0]) 
            
        #add all phone numbers to a list (adding None if no email is listed)   
        if pattern3.findall(lines[i]) == []:
            phone.append(None)
        else:
            first = pattern3.findall(lines[i])[0][0]  #get the first 3 digits of the phones number
            second = pattern3.findall(lines[i])[0][1]  #then the next 3
            third = pattern3.findall(lines[i])[0][2]  #then the last 4
            number = '(' + first + ')' + second + '-' + third  #add them together in the format: (XXX)YYY-ZZZZ
            phone.append(number)
        
        #make sure all the birthdays are in correct format and add to the list of birthdays
        if pattern2.findall(lines[i]) == []:
            bday.append(None)
        else:
            month = pattern2.findall(lines[i])[0][0]  #isolate the months
            day = pattern2.findall(lines[i])[0][1]   #isolate the days
            year = pattern2.findall(lines[i])[0][2]  #isolate the years
            #add zeros to the front of the day and month if they need and 20 if no century is given
            if len(month) != 2:
                month = str(0) + month
            if len(day) != 2:
                day = str(0) + day
            if len(year) != 4:
                year = str(20) + year
            birthday = month + '/' + day + '/' + year
            bday.append(birthday)
        
        #add all the names to a list
        names.append(pattern4.findall(lines[i])[0])
    
    #use the zip function to make a tuple of the names, birthdays,emails, and phone numbers 
    s = zip(names, bday, email, phone)
    s = list(s)
    
    dict1 = dd(dict) #create an empty dictionary
    for name,birth,em,phon in s:
        dict1[name]["birthday"] = birth #add birthday, email, and phone with their respected elements to a dictionary from the names
        dict1[name]["email"] = em
        dict1[name]["phone"] = phon
    return dict1     #return the dictionary

#test functions
def test2():
    A = prob2()
    print(bool(A.search("^{@}(?)[%]{.}(*)[_]{&}$")))
    
def test3():
    A = prob3()
    print(bool(A.search("Book store")))
    print(bool(A.search("Mattress store")))
    print(bool(A.search("Grocery store")))
    print(bool(A.search("Book supplier")))
    print(bool(A.search("Mattress supplier")))
    print(bool(A.search("Grocery supplier")))
    print(bool(A.search("Book Grocery")))
    print(bool(A.search("Jet fuel can't melt steel beams")))
    
    
def test4():
    A = prob4()
    test = ["Mouse", "compile", "_123456789", "__x__", "while",
            "max=4.2", "string= ''", "num_guesses", "x = 3.14", "x ", "compile = 'hold on to your butts'", "_"]
    test2 = ["3rats", "err*r", "sq(x)", "sleep()", " x",
             "300", "is_4=(value==4)", "pattern = r'^one|two fish$'",
             "x = 3.14.1", "x  =", "()"]
    
    for i in test:
        B = bool(A.search(i))
        if B == False:
            print(i,B)
            
    for i in test2:
        print(i, bool(A.search(i)))
    
def test5():
    code = """
    k, i, p = 999, 1, 0
    while k > i
        i *= 2
        p += 1
        if k != 999
            print("k should not have changed")
        else
            pass
    print(p)
    
    
    Should it have a colon after it?
    while YES
    if    YES
    elif  YES
    butts
    with  YES
    def   YES
    class YES
    sandwich
    full house
    try   YES
   except YES
    motel
    """
    print(prob5(code))