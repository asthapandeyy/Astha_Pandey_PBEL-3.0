import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Student Performance Prediction & Analytics System")

st.markdown(
"""
This AI system predicts student performance using:
- Study habits
- Attendance
- Assignment completion
- Previous academic performance

It also provides personalized improvement suggestions.
"""
)


st.divider()
st.sidebar.header("🤖 Model Information")

st.sidebar.success(
"""
Algorithm:
Decision Tree Classifier

Dataset:
500 Student Records

Features:
✔ Hours Studied
✔ Attendance
✔ Assignments
✔ Previous Score

Model Accuracy:
90%

Task:
Binary Classification
"""
)
st.sidebar.header("👤 Student Profile")

student_name = st.sidebar.text_input(
    "Student Name",
    "Student"
)

student_id = st.sidebar.text_input(
    "Student ID",
    "001"
)


col1, col2 = st.columns(2)


with col1:

    hours_studied = st.number_input(
        "📚 Hours Studied Per Day",
        0,
        24,
        5
    )

    attendance = st.number_input(
        "📅 Attendance Percentage",
        0,
        100,
        75
    )


with col2:

    assignments = st.number_input(
        "📝 Assignment Completion %",
        0,
        100,
        80
    )

    previous_score = st.number_input(
        "📊 Previous Exam Score",
        0,
        100,
        60
    )


if st.button("🚀 Analyze Student Performance"):


    input_data = pd.DataFrame(
        [[
            hours_studied,
            attendance,
            assignments,
            previous_score
        ]],
        columns=[
            "Hours_Studied",
            "Attendance",
            "Assignments",
            "Previous_Score"
        ]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)

    confidence = max(probability[0])*100

    # Performance Score

    performance_score = (
        (hours_studied/10)*25 +
        (attendance/100)*25 +
        (assignments/100)*25 +
        (previous_score/100)*25
    )

    # Limit score

    performance_score = min(
        performance_score,
        100
    )

    st.divider()

    st.header("📌 AI Prediction Result")


    if performance_score >= 85:

        category = "🌟 Excellent"

    elif performance_score >= 70:

        category = "🎉 Good"

    elif performance_score >= 50:

        category = "🙂 Average"

    else:

        category = "⚠️ Needs Improvement"

    if prediction == 1:

        st.success(
            f"{student_name} is predicted to have {category} performance"
        )

    else:

        st.warning(
            f"{student_name} requires academic improvement"
        )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Performance Score",
        f"{performance_score:.1f}%"
    )

    c2.metric(
        "AI Confidence",
        f"{confidence:.1f}%"
    )

    c3.metric(
        "Prediction",
        "Positive" if prediction==1 else "Needs Attention"
    )

    st.divider()

    st.header("📊 Performance Analytics")


    chart_data = pd.DataFrame(
        {
            "Area":[
                "Study",
                "Attendance",
                "Assignments",
                "Previous Score"
            ],

            "Score":[
                hours_studied*10,
                attendance,
                assignments,
                previous_score
            ]
        }
    )


    st.bar_chart(
        chart_data.set_index("Area")
    )

    st.header("🧠 Personalized AI Feedback")


    strengths=[]

    weaknesses=[]


    if hours_studied >=6:
        strengths.append(
            "Strong study discipline"
        )
    else:
        weaknesses.append(
            "Increase daily study hours"
        )


    if attendance >=85:
        strengths.append(
            "Excellent attendance"
        )
    else:
        weaknesses.append(
            "Maintain better attendance"
        )


    if assignments >=80:
        strengths.append(
            "Good assignment consistency"
        )
    else:
        weaknesses.append(
            "Submit assignments regularly"
        )


    if previous_score >=75:
        strengths.append(
            "Strong academic foundation"
        )
    else:
        weaknesses.append(
            "Revise previous concepts"
        )



    st.subheader("💪 Strengths")

    for s in strengths:
        st.write("✅",s)



    st.subheader("🚀 Improvement Areas")

    for w in weaknesses:
        st.write("🔹",w)



    st.header("📚 Recommended Action Plan")


    if attendance <75:

        st.write(
            "• Attend more classes and maintain minimum 75% attendance."
        )


    if hours_studied <5:

        st.write(
            "• Follow a minimum 5 hour daily study schedule."
        )


    if assignments <80:

        st.write(
            "• Complete assignments weekly to strengthen concepts."
        )


    if previous_score <60:

        st.write(
            "• Practice previous year questions and mock tests."
        )


    if len(weaknesses)==0:

        st.write(
            "✨ Maintain your current performance level."
        )



    st.header("⚠️ Academic Risk Analysis")


    risk=0


    if attendance<75:
        risk+=1

    if assignments<70:
        risk+=1

    if previous_score<60:
        risk+=1

    if hours_studied<4:
        risk+=1



    if risk==0:

        st.success(
            "Low Risk: Student is performing consistently."
        )

    elif risk<=2:

        st.warning(
            "Medium Risk: Improvement required."
        )

    else:

        st.error(
            "High Risk: Immediate academic attention required."
        )



    report=f"""

AI STUDENT PERFORMANCE REPORT

Name: {student_name}
ID: {student_id}

Performance Score:
{performance_score:.1f}%

Prediction:
{"Good Performance" if prediction==1 else "Needs Improvement"}

Confidence:
{confidence:.1f}%


Strengths:
{strengths}


Improvement Areas:
{weaknesses}


Generated:
{datetime.now()}

"""


    st.download_button(
        "📄 Download Student Report",
        report,
        file_name=f"{student_name}_report.txt"
    )
    st.divider()

    st.header("🧠 AI Model Insights")

    st.write(
    """
    The AI model identifies which factors influence student performance the most.
    Higher importance means that factor has a stronger impact on prediction.
    """
    )


    try:

        image = Image.open(
            "feature_importance.png"
        )

        st.image(
            image,
            caption="Factors Affecting Student Performance",
            use_container_width=True
        )


    except:

        st.info(
            "Feature importance graph not found. Run feature_importance.py first."
        )
    st.divider()

    st.header("📊 Model Evaluation")


    try:

        cm_image = Image.open(
            "confusion_matrix.png"
        )

        st.image(
            cm_image,
            caption="Confusion Matrix of AI Model",
            use_container_width=True
        )


    except:

        st.info(
            "Confusion matrix image not found."
        )    
