import pandas as pd
import random


data = []


for i in range(500):

    hours = random.randint(1,10)

    attendance = random.randint(50,100)

    assignments = random.randint(30,100)

    previous_score = random.randint(35,100)


    # Performance logic
    score = (
        hours*5 +
        attendance*0.3 +
        assignments*0.2 +
        previous_score*0.4
    )


    if score >= 80:
        result = 1
    else:
        result = 0


    data.append(
        [
            hours,
            attendance,
            assignments,
            previous_score,
            result
        ]
    )


df = pd.DataFrame(
    data,
    columns=[
        "Hours_Studied",
        "Attendance",
        "Assignments",
        "Previous_Score",
        "Result"
    ]
)


df.to_csv(
    "student_data_large.csv",
    index=False
)


print("Dataset generated successfully!")