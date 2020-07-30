#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# CMPE365 Assignment 4
# @author Stefan Robb 20027476
# I certify that this submission contains my own work, except as noted.

# Decodes files into their Huffman code and checks two files against eachother in the most efficient manner possible. 


# ## Command Line Arguments

# In[62]:


import sys

filenames = ["Dijkstra_py3.py", "Dijkstra.py", "Three_Bears.v1.txt", "Three_Bears.v2.txt"]
file1 = None
file2 = None

# Prompt user for valid file name input
while not file1:
    input1 = input("File 1: ")
    if input1 not in filenames:
        print(f"{input1} is an invalid file name")
    else:
        file1 = input1
while not file2:
    input2 = input("File 2: ")
    if input2 not in filenames:
        print(f"{input2} is an invalid file name")
    else:
        file2 = input2
        
path = r"" # Specify path to folder containing possible files
filepath1 = f"{path}\{file1}"
filepath2 = f"{path}\{file2}"


# ## Table Building Using Recursion

# In[63]:


with open(filepath1, "r") as f:
    lines1 = f.read().splitlines()
    hashlines1 = [hashstr(line) for line in lines1]
with open(filepath2, "r") as f:
    lines2 = f.read().splitlines()
    hashlines2 = [hashstr(line) for line in lines1]
# Initialize 2d list for storing tabular results
result = [[-1 for i in lines2] for j in lines1]
# Fill result table with longest common sublines
for i in range(len(result)):
    for j in range(len(result[i])):
        result[i][j] = LCSL(i, j, result)


# ## String Builidng and Output

# In[77]:


i = len(result) - 1
j = len(result[0]) - 1
i_end = i
j_end = j
match = comparelines(i, j)
outstrs = []
# Go through the result table in reverse finding sequences of matches and mismatches
while i >= 0 and j >= 0:
    if comparelines(i, j):
        if match is False:
            outstrs.append(f"Mismatch:\t{file1}: {writeline(i, i_end)}\t\t{file2}: {writeline(j, j_end)}")
            i_end = i
            j_end = j
        match = True
        i -= 1
        j -= 1
    else:
        if match is True:
            outstrs.append(f"Match:\t\t{file1}: {writeline(i, i_end)}\t\t{file2}: {writeline(j, j_end)}")
            i_end = i
            j_end = j
        match = False
        if i <= 0 and j > 0:
            j -= 1
        elif i > 0 and j <= 0:
            i -= 1
        elif i <= 0 and j <= 0:
            i -= 1
            j -= 1
        else:
            if result[i-1][j] == result[i][j]:
                i -= 1
            elif result[i][j-1] == result[i][j]:
                j -= 1
            else:
                i -= 1
                j -= 1
if match:
    outstrs.append(f"Match:\t\t{file1}: {writeline(i, i_end)}\t\t{file2}: {writeline(i, j_end)}")
else:
    outstrs.append(f"Mismatch:\t\t{file1}: {writeline(i, i_end)}\t\t{file2}: {writeline(j, j_end)}")
outstrs.reverse()
for s in outstrs:
    print(s)


# ## Functions

# In[16]:


def writeline(start, end):
    if start == end:
        return "None"
    return f"<{start+2} .. {end+1}>"


# In[5]:


def LCSL(i, j, table):
    if table[i][j] != -1:
        return table[i][j]
    # base cases
    if i == 0 and j == 0:
        if comparelines(0, 0):
            return 1
        else: 
            return 0
    elif i == 0:
        if LCSL(0, j-1, table) == 1 or comparelines(0, j):
            return 1
        else:
            return 0
    elif j == 0:
        if LCSL(i-1, 0, table) == 1 or comparelines(i, 0):
            return 1
        else:
            return 0
    # recurssive relationship
    else:
        if comparelines(i, j):
            return 1 + LCSL(i-1, j-1, table)
        else:
            return max(LCSL(i-1, j, table), LCSL(i, j-1, table), LCSL(i-1, j-1, table))


# In[21]:


# An efficient method to determine if two lines are equal
def comparelines(i, j):
    if hashlines1[i] != hashlines2[j]:
        return False
    else:
        return lines1[i] == lines2[j]


# In[60]:


# A hash function for converting strings into almost unique integers
def hashstr(string):
    result = 0
    for c in string:
        result += (7*result + ord(c)) % 100000
    return result

