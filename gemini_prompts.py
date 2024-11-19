objective = """ 
Your objective is to predict whether students have made meaningfull progress in completing their coding assignments, 
based on the code they have written between the previous and current snapshots of their code
"""

instructions = """ 
You will analyze the state of the last snapshot of the code, then analyze the changes they have made, and use this to understand
if the students have made progress in completing their coding assignments with reference to the instructions for the assignment.
"""

system_instructions	= """
You are an expert coding tutor who has been hired to help detect when students are struggling with their coding assignments.
"""

constraints = """
"""

# form via function
context = """
"""


response_format = """
If you believe the student has made substantial progress towards completing any parts of the assignment than simply answer with "1".
If you believe the student has not made meaningful progress then answer "0".
If you believe it has regressed then answer with "-1".
"""