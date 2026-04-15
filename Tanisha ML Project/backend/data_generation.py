import pandas as pd
import numpy as np
import os
import random

def generate_student_data(num_students=200):
    subjects = ['Math', 'Science', 'English', 'History', 'Art']
    classes = ['Grade 9', 'Grade 10', 'Grade 11', 'Grade 12']
    participation_levels = ['Low', 'Medium', 'High']
    
    data = []
    
    first_names = ["Arjun", "Tanisha", "Rahul", "Priya", "Siddharth", "Ananya", "Vikram", "Ishita", "Rohan", "Sanya"]
    last_names = ["Sharma", "Verma", "Gupta", "Malhotra", "Kapoor", "Joshi", "Mehta", "Singh", "Reddy", "Patel"]

    for i in range(1, num_students + 1):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        cls = random.choice(classes)
        
        for subject in subjects:
            attendance = np.random.uniform(60, 100)
            study_hours = np.random.uniform(2, 30)
            assignment_completion = np.random.uniform(50, 100)
            participation = random.choice(participation_levels)
            
            # Base scores influenced by study hours and attendance
            score1 = 40 + (study_hours * 1.5) + (attendance * 0.2) + np.random.normal(0, 5)
            score2 = score1 + np.random.normal(2, 3) # Slight improvement generally
            score3 = score2 + np.random.normal(1, 4)
            
            # Ensure scores are within 0-100
            score1 = max(0, min(100, score1))
            score2 = max(0, min(100, score2))
            score3 = max(0, min(100, score3))
            
            # Target Final Score (what we want to predict)
            # Highly correlated with previous scores, attendance, and study hours
            final_score = (0.2 * score1 + 0.2 * score2 + 0.3 * score3 + 
                           0.15 * (attendance) + 0.15 * (study_hours * 3)) + np.random.normal(0, 3)
            final_score = max(0, min(100, final_score))
            
            data.append({
                'Student_ID': f'STU_{i:03d}',
                'Name': name,
                'Class': cls,
                'Subject': subject,
                'Exam_Score_1': round(score1, 2),
                'Exam_Score_2': round(score2, 2),
                'Exam_Score_3': round(score3, 2),
                'Attendance_Percentage': round(attendance, 2),
                'Study_Hours_Per_Week': round(study_hours, 2),
                'Assignment_Completion_Rate': round(assignment_completion, 2),
                'Participation_Level': participation,
                'Final_Score': round(final_score, 2)
            })
            
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/student_performance.csv', index=False)
    print(f"Generated data for {num_students} students across {len(subjects)} subjects.")
    return df

if __name__ == "__main__":
    generate_student_data()
