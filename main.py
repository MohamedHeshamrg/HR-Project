import streamlit as st
st.set_page_config(page_title="Employee Insights", layout="wide")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


# إعداد الستايل
sns.set(style='whitegrid')

# تحميل البيانات
dfE = pd.read_csv("https://raw.githubusercontent.com/MohamedHeshamrg/HR-Project/main/Data/employee.csv")
dfE = dfE.rename(columns={'id': 'employee_id'})
dfDE = pd.read_csv("Data/department_employee.csv")
dfDE = dfDE.rename(columns={'from_date': 'from_date_de','to_date' :"to_date_de"})
dfD = pd.read_csv("Data/department.csv")
dfD = dfD.rename(columns={'id': 'department_id'})

url = 'https://drive.google.com/uc?export=download&id=1Dn5K1a5SCFKgeA54CSqQtYCJ0xYX6020'
dfS = pd.read_csv("url")
dfS = dfS.rename(columns={'from_date': 'from_date_s','to_date' :"to_date_s"})
dfDM = pd.read_csv("Data/department_manager.csv")
dfDM = dfDM.rename(columns={'from_date': 'from_date_dm','to_date' :"to_date_dm"})
dfT = pd.read_csv("Data/title.csv")
dfT = dfT.rename(columns={'from_date': 'from_date_t','to_date' :"to_date_t"})
dfCE = pd.read_csv("https://raw.githubusercontent.com/MohamedHeshamrg/HR-Project/main/Data/current_employee_snapshot.csv")

dfDE_full = dfDE.merge(dfD, on='department_id', how='left')

df_main = dfDE_full.merge(dfE, on='employee_id', how='left')

df_main = df_main.merge(dfS, on='employee_id', how='left')

df_main = df_main.merge(dfT, on='employee_id', how='left')

df_main = df_main.merge(dfDM, on=['employee_id', 'department_id'], how='left')

df_main = df_main.merge(dfCE, on='employee_id', how='left')


for col in ['to_date_de', 'to_date_s', 'to_date_t', 'to_date_dm']:
    df_main[col] = pd.to_datetime(df_main[col], errors='coerce')

df_main['hire_date'] = pd.to_datetime(df_main['hire_date'], errors='coerce')


today = pd.to_datetime('today')

df_main['is_current'] = df_main['to_date_de'].isna() | (df_main['to_date_de'] > today)
df_main['exited'] = ~df_main['is_current']




# عنوان الصفحة
st.title("📊 Employee Data Insights Dashboard")

# قسم اختياري: نظرة عامة على البيانات
if st.checkbox("Data Fream "):
    st.dataframe(df_main.head())



st.title("📊 Employee Data Insights")

st.markdown("---")

st.header("🔍 Key Insights")

st.markdown("""
- **Salary & Pressure**: The most common reasons for employee exits are low salaries or high work pressure.
- **Department Sizes**: Customer Service has an unusually high number of Senior Staff (161K) and Staff (152K), suggesting major service expansion.
- **Salary Discrepancy**:
  - Highest salaries are in **Sales**.
  - Departments like **Development** and **Production** need salary increases.
- **Salary Fairness**: Salaries are fairly distributed between men and women.
- **Job Titles**: Engineers are the most common job title.
- **Largest Department**: Development has the highest number of employees.
- **Turnover**:
  - Most employees exited in the year **2000**.
  - **Development, Production, and Sales** have the highest turnover rates.
  - **Customer Service** has the lowest turnover.
- **Hiring Trends**:
  - Hiring occurs in all seasons, suggesting constant expansion.
  - The hiring rate is high, indicating rapid company growth.
- **Tenure**:
  - Most employees stay more than **10 years** in the company.
- **Age Distribution**:
  - 📌 Average age: **43.5**
  - 🔻 Youngest: **37**
  - 🔺 Oldest: **50**
  - 🔢 Standard Deviation: **3.77**
- **Loyalty Recommendation**: Salaries should increase more consistently with tenure to promote employee loyalty.
""")

st.markdown("---")
st.success("✅ Summary: The company is expanding rapidly with a stable employee base. Salaries need to be balanced across departments and tenure should be rewarded.")


# Load and parse dates, handling '9999-01-01'
dfE['hire_date'] = pd.to_datetime(dfE['hire_date'])
dfDE['from_date_de'] = pd.to_datetime(dfDE['from_date_de'])
dfDE['to_date_de'] = dfDE['to_date_de'].replace('9999-01-01', pd.NaT)
dfDE['to_date_de'] = pd.to_datetime(dfDE['to_date_de'])

# Today's date for tenure calculation
today = pd.Timestamp(datetime.today())


emp_dept = pd.merge(dfDE, dfE, left_on='employee_id', right_on='employee_id', how='left')

# Step 2: Calculate employee total tenure (years)
emp_dept['tenure_years'] = (today - emp_dept['hire_date']).dt.days // 365

# Step 3: Use most recent assignment only (to avoid duplication)
emp_dept_latest = emp_dept.sort_values('from_date_de').drop_duplicates('employee_id', keep='last')

# Step 4: Merge with department names
emp_dept_named = pd.merge(emp_dept_latest, dfD, left_on='department_id', right_on='department_id', how='left')

# Step 5 (Updated): Group by department and compute both average tenure and employee count
dept_summary = emp_dept_named.groupby('dept_name').agg(
    average_tenure_years=('tenure_years', 'mean'),
    employee_count=('employee_id', 'nunique')
).sort_values(by='average_tenure_years', ascending=False)


# Optional: Bar Plot

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=dept_summary, y="employee_count", x="dept_name", ax=ax)

ax.set_title("Average Tenure by Department")
ax.set_xlabel("Employee Count")
ax.grid(axis='x', linestyle='--', alpha=0.7)

st.pyplot(fig)


emp_salary = pd.merge(dfS, dfE, left_on='employee_id', right_on='employee_id', how='left')

# Step 2: Filter most recent salary entry for each employee
emp_salary_latest = emp_salary.sort_values('to_date_s').drop_duplicates('employee_id', keep='last')

# Step 3: Merge with latest department assignment
emp_dept_salary = pd.merge(emp_salary_latest, emp_dept_latest, on='employee_id', how='left')

# Step 4: Merge with department names
emp_dept_salary_named = pd.merge(emp_dept_salary, dfD, left_on='department_id', right_on='department_id', how='left')

# Step 5: Group by department name and sum salaries
dept_salary_total = emp_dept_salary_named.groupby('dept_name')['amount'].sum().sort_values(ascending=False)

# Step 6: Display result
print("\n💰 Total Salary Paid per Department:")
print(dept_salary_total)

# Step 7: Get top department
top_salary_dept = dept_salary_total.idxmax()
top_salary_value = dept_salary_total.max()

# Step 1: Reset index to convert Series to DataFrame
dept_salary_total_df = dept_salary_total.reset_index()
dept_salary_total_df.columns = ['Department', 'Total Salary']

# Step 2: Set visual style

# Step 1: Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
sns.set(style="whitegrid")

# Step 2: Bar plot
sns.barplot(
    data=dept_salary_total_df,
    x='Total Salary',
    y='Department',
    palette='viridis',
    ax=ax
)

# Step 3: Add value labels
for index, value in enumerate(dept_salary_total_df['Total Salary']):
    ax.text(value + 1000, index, f"${value:,.0f}", va='center')

# Step 4: Titles and labels
ax.set_title("💰 Total Salary Paid per Department", fontsize=16)
ax.set_xlabel("Total Salary (USD)")
ax.set_ylabel("Department")

# Step 5: Layout and display in Streamlit
fig.tight_layout()
st.pyplot(fig)

df_main['full_name'] = df_main['first_name'] + ' ' + df_main['last_name']

df_salary = df_main[['employee_id', 'full_name', 'amount', 'department_id', 'dept_name_x']].dropna(subset=['amount'])

df_sorted = df_salary.sort_values(by=['department_id', 'amount'], ascending=[True, False])

top10_per_dept = df_sorted.groupby('department_id').head(20)

top10_per_dept = top10_per_dept.sort_values(by=['department_id', 'amount'], ascending=[True, False])

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 13))

# Create barplot
sns.barplot(
    data=top10_per_dept,
    x='amount',
    y='full_name',
    hue='dept_name_x',
    dodge=False,
    ax=ax
)

# Customize plot
ax.set_title('Top 10 Highest Paid Employees per Department')
ax.set_xlabel('Salary')
ax.set_ylabel('Employee')
ax.legend(title='Department')

# Adjust layout
fig.tight_layout()

# Show in Streamlit
st.pyplot(fig)

df_main['from_date_s'] = pd.to_datetime(df_main['from_date_s'])
df_main['year'] = df_main['from_date_s'].dt.year
avg_salary_by_year = df_main.groupby('year')['amount'].mean().reset_index()

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create line plot
sns.lineplot(data=avg_salary_by_year, x='year', y='amount', marker='o', ax=ax)

# Customize plot
ax.set_title('Average Salary Growth Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Average Salary')
ax.grid(True)

# Adjust layout and show in Streamlit
fig.tight_layout()
st.pyplot(fig)

salary_growth_dept = df_main.groupby(['dept_name_x', 'year'])['amount'].mean().reset_index()
salary_growth_dept = salary_growth_dept.sort_values(by=['dept_name_x', 'year'])


# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 7))

# Line plot by department
sns.lineplot(data=salary_growth_dept, x='year', y='amount', hue='dept_name_x', marker='o', ax=ax)

# Customize plot
ax.set_title('Salary Growth Over Time by Department')
ax.set_xlabel('Year')
ax.set_ylabel('Average Salary')
ax.grid(True)
ax.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')

# Layout and show
fig.tight_layout()
st.pyplot(fig)

df_main['from_date_s'] = pd.to_datetime(df_main['from_date_s'])
df_main['year'] = df_main['from_date_s'].dt.year

avg_salary_by_year = df_main.groupby('year')['amount'].mean().reset_index()

avg_salary_by_year['growth_%'] = avg_salary_by_year['amount'].pct_change() * 100


# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the salary growth
ax.plot(avg_salary_by_year['year'], avg_salary_by_year['growth_%'], marker='o', color='green')

# Customize the chart
ax.set_title('Year-over-Year Salary Growth (%)')
ax.set_xlabel('Year')
ax.set_ylabel('Growth Percentage')
ax.grid(True)

# Layout and display in Streamlit
fig.tight_layout()
st.pyplot(fig)




df_main['tenure_years'] = (pd.to_datetime(df_main['to_date_de']) - pd.to_datetime(df_main['hire_date'])).dt.days / 365

avg_tenure_per_dept = df_main.groupby(['department_id', 'dept_name_x'])['tenure_years'].mean().reset_index()

avg_tenure_per_dept = avg_tenure_per_dept.sort_values(by='tenure_years')

short_tenure_df = df_main[df_main['tenure_years'] < 2]

short_counts = short_tenure_df.groupby(['department_id', 'dept_name_x'])['employee_id'].count().reset_index(name='short_term_count')

total_counts = df_main.groupby(['department_id', 'dept_name_x'])['employee_id'].count().reset_index(name='total_count')

turnover_df = short_counts.merge(total_counts, on=['department_id', 'dept_name_x'])
turnover_df['turnover_rate_%'] = (turnover_df['short_term_count'] / turnover_df['total_count']) * 100

turnover_df = turnover_df.sort_values(by='turnover_rate_%', ascending=False)


fig, ax = plt.subplots(figsize=(12, 6))

# Barplot
sns.barplot(
    data=turnover_df,
    x='dept_name_x',
    y='turnover_rate_%',
    palette='Reds_r',
    ax=ax
)

# Customize
ax.set_title('Departments with High Turnover Rate')
ax.set_ylabel('Turnover Rate (%)')
ax.set_xlabel('Department')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Layout and display
fig.tight_layout()
st.pyplot(fig)

df_main['to_date_dm'] = pd.to_datetime(df_main['to_date_dm'])
df_main['exit_year'] = df_main['to_date_dm'].dt.year
turnover_per_year = df_main.groupby(['dept_name_x', 'exit_year'])['employee_id'].nunique().reset_index()
turnover_per_year = turnover_per_year.rename(columns={'employee_id': 'num_employees_left'})








# High Turnover Departments
dfDE['to_date_de'] = pd.to_datetime(dfDE['to_date_de'], errors='coerce')
dfD['department_id'] = dfD['department_id'].astype(str)

dfDE['exit_year'] = dfDE['to_date_de'].dt.year

turnover = dfDE.merge(dfD, on='department_id', how='left')

turnover_count = turnover.groupby(['dept_name', 'exit_year'])['employee_id'].count().reset_index()
turnover_count.rename(columns={'employee_id': 'num_employees_left'}, inplace=True)

pivot_table = turnover_count.pivot(index='dept_name', columns='exit_year', values='num_employees_left').fillna(0)

fig, ax = plt.subplots(figsize=(14, 7))

# Plot heatmap
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="Reds", ax=ax)

# Customize
ax.set_title("🔴 Turnover per Department per Year")
ax.set_xlabel("Exit Year")
ax.set_ylabel("Department")

# Show in Streamlit
fig.tight_layout()
st.pyplot(fig)



gender_salary = df_main.groupby(['dept_name_x', 'gender_x'])['amount'].mean().reset_index()



plt.figure(figsize=(14, 6))

sns.barplot(
    data=gender_salary,
    x='dept_name_x',
    y='amount',
    hue='gender_x',
    palette='Set2'
)
plt.title('🔵 Average Salary by Department and Gender', fontsize=14)
plt.xlabel('Department')
plt.ylabel('Average Salary')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Gender')
plt.tight_layout()

plt.show()



title_counts_by_dept = df_main.groupby(['dept_name_x', 'title_x'])['employee_id'].count().reset_index()
title_counts_by_dept.rename(columns={'employee_id': 'count'}, inplace=True)

sns.set(style="whitegrid")

# Create the catplot
g = sns.catplot(
    data=title_counts_by_dept,
    x='title_x',
    y='count',
    hue='dept_name_x',
    kind='bar',
    height=6,
    aspect=2,
    palette='tab10'
)

# Customize titles and labels
g.fig.suptitle('Employee Count per Job Title in Each Department', fontsize=16, weight='bold')
g.set_axis_labels("Job Title", "Number of Employees")
g.set_xticklabels(rotation=30, ha='right', fontsize=10)
g._legend.set_title('Department')
g.set(ylim=(0, title_counts_by_dept['count'].max() * 1.1))

# Adjust layout
g.fig.tight_layout()
g.fig.subplots_adjust(top=0.92)

# Show in Streamlit
st.pyplot(g.fig)

dfDE_full = dfDE.merge(dfD, on='department_id', how='left')

df_main = dfDE_full.merge(dfE, on='employee_id', how='left')

df_main = df_main.merge(dfS, on='employee_id', how='left')

df_main = df_main.merge(dfT, on='employee_id', how='left')

df_main = df_main.merge(dfDM, on=['employee_id', 'department_id'], how='left')

df_main = df_main.merge(dfCE, on='employee_id', how='left')


for col in ['to_date_de', 'to_date_s', 'to_date_t', 'to_date_dm']:
    df_main[col] = pd.to_datetime(df_main[col], errors='coerce')

df_main['hire_date'] = pd.to_datetime(df_main['hire_date'], errors='coerce')


today = pd.to_datetime('today')

df_main['is_current'] = df_main['to_date_de'].isna() | (df_main['to_date_de'] > today)
df_main['exited'] = ~df_main['is_current']




# Is there a noticeable difference in hiring between men and women?
df_main['hire_date'] = pd.to_datetime(df_main['hire_date'], errors='coerce')
df_main = df_main[df_main['gender_x'].notna()]

gender_counts = df_main['gender_x'].value_counts()
gender_percent = df_main['gender_x'].value_counts(normalize=True) * 100

print("🔢 Number of employees by gender:\n", gender_counts)
print("\n📊 Percentage:\n", gender_percent.round(2))

fig, ax = plt.subplots(figsize=(6, 6))
gender_counts.plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=140,
    colors=["skyblue", "lightcoral"],
    ax=ax
)

ax.set_title("Gender Distribution of Hires")
ax.set_ylabel("")  # إزالة تسمية المحور
plt.tight_layout()

# عرض الشكل في Streamlit
st.pyplot(fig)



# Are certain departments hiring more than others? Why?

hires_by_dept = df_main.groupby('dept_name_x')['employee_id'].nunique().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=hires_by_dept.values, y=hires_by_dept.index, palette='viridis', ax=ax)

ax.set_title("Number of Hires by Department")
ax.set_xlabel("Number of Hires")
ax.set_ylabel("Department")
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

# عرض الشكل في Streamlit
st.pyplot(fig)


df_main['to_date_de'] = pd.to_datetime(df_main['to_date_de'], errors='coerce')

end_date = pd.to_datetime('2002-08-01')

df_main['end_date_fixed'] = df_main['to_date_de'].fillna(end_date)

df_main['status'] = df_main['end_date_fixed'].apply(lambda x: 'Current' if x >= end_date else 'Left')

status_by_dept = df_main.groupby(['dept_name_x', 'status']).size().unstack(fill_value=0)

status_by_dept['Total'] = status_by_dept.sum(axis=1)
status_by_dept = status_by_dept.sort_values(by='Total', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))

status_by_dept[['Current', 'Left']].plot(
    kind='barh',
    stacked=False,
    color=['#2ecc71', '#e74c3c'],
    ax=ax
)

ax.set_xlabel('Number of Employees')
ax.set_ylabel('Department')
ax.set_title('Current vs Left Employees by Department')
ax.invert_yaxis()
ax.legend(title='Status')
plt.tight_layout()

# عرض الشكل في Streamlit
st.pyplot(fig)



# What are the average salaries in each department? Is there a balance?
dept_salary_avg = df_main.groupby('dept_name_x')['salary_amount'].mean().sort_values(ascending=False)

fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(x=dept_salary_avg.values, y=dept_salary_avg.index, palette='viridis', ax=ax1)
ax1.set_title("Average Salary per Department")
ax1.set_xlabel("Average Salary")
ax1.set_ylabel("Department")
plt.tight_layout()
st.pyplot(fig1)

# الشكل الثاني: Salary Distribution per Department
fig2, ax2 = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df_main, x='dept_name_x', y='salary_amount', ax=ax2)
ax2.set_title("Salary Distribution per Department")
ax2.set_xlabel("Department")
ax2.set_ylabel("Salary Amount")
plt.setp(ax2.get_xticklabels(), rotation=45)
plt.tight_layout()
st.pyplot(fig2)


# Is there a significant salary difference between different job titles?
title_salary_avg = df_main.groupby('title_x')['salary_amount'].mean().sort_values(ascending=False)

# الشكل الأول: Average Salary by Job Title
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(x=title_salary_avg.values, y=title_salary_avg.index, palette='magma', ax=ax1)
ax1.set_title("Average Salary by Job Title")
ax1.set_xlabel("Average Salary")
ax1.set_ylabel("Job Title")
plt.tight_layout()
st.pyplot(fig1)

# الشكل الثاني: Salary Distribution by Job Title
fig2, ax2 = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df_main, x='title_x', y='salary_amount', ax=ax2)
ax2.set_title("Salary Distribution by Job Title")
ax2.set_xlabel("Job Title")
ax2.set_ylabel("Salary Amount")
plt.setp(ax2.get_xticklabels(), rotation=45)
plt.tight_layout()
st.pyplot(fig2)


# Is there a salary gap between men and women in the same job title?
salary_by_gender_title = df_main.groupby(['title_x', 'gender_x'])['salary_amount'].mean().unstack()

print(salary_by_gender_title.round(2))


# الرسم: Average Salary by Title and Gender
fig, ax = plt.subplots(figsize=(12, 6))
salary_by_gender_title.plot(kind='bar', ax=ax)

ax.set_title("Average Salary by Title and Gender")
ax.set_ylabel("Average Salary")
ax.set_xlabel("Job Title")
plt.setp(ax.get_xticklabels(), rotation=45)
ax.legend(title="Gender")
plt.tight_layout()

st.pyplot(fig)

salary_by_dept_gender = (
    df_main
    .groupby(['dept_name_x', 'gender_x'])['salary_amount']
    .mean()
    .unstack()
    .round(2)
)



fig, ax = plt.subplots(figsize=(12, 6))
salary_by_dept_gender.plot(kind='bar', ax=ax)

ax.set_title("Average Salary by Department and Gender")
ax.set_ylabel("Average Salary")
ax.set_xlabel("Department")
plt.setp(ax.get_xticklabels(), rotation=45)
ax.legend(title="Gender")
plt.tight_layout()

st.pyplot(fig)



# Are there employees who have been in the job for many years and haven't received a raise? Why?
df_main['full_name'] = df_main['first_name'] + ' ' + df_main['last_name']
long_term_no_raise = df_main[
    (df_main['company_tenure'] >= 5) &
    (df_main['salary_percentage_change'] == 0)
]

long_term_no_raise[['employee_id', 'full_name', 'title_x', 'dept_name_x', 'company_tenure', 'salary_amount']]

long_term_no_raise['dept_name_x'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=long_term_no_raise, x='company_tenure', bins=20, ax=ax)

ax.set_title("Distribution of Tenure (No Raise Given)")
ax.set_xlabel("Years in Company")
ax.set_ylabel("Number of Employees")
plt.tight_layout()

st.pyplot(fig)


df_titles = dfT.copy()
df_dept_movements = dfDM.copy()


max_valid_date = pd.to_datetime('2002-12-31')

df_titles['to_date_t'] = df_titles['to_date_t'].replace('9999-01-01', max_valid_date)
df_dept_movements['to_date_dm'] = df_dept_movements['to_date_dm'].replace('9999-01-01', max_valid_date)

df_titles['from_date_t'] = pd.to_datetime(df_titles['from_date_t'])
df_titles['to_date_t'] = pd.to_datetime(df_titles['to_date_t'])
df_dept_movements['from_date_dm'] = pd.to_datetime(df_dept_movements['from_date_dm'])
df_dept_movements['to_date_dm'] = pd.to_datetime(df_dept_movements['to_date_dm'])


df_title_dept = pd.merge(df_titles, df_dept_movements, on='employee_id', how='inner')

df_title_dept = df_title_dept[
    (df_title_dept['from_date_t'] <= df_title_dept['to_date_dm']) &
    (df_title_dept['to_date_t'] >= df_title_dept['from_date_dm'])
]

df_title_dept['period'] = df_title_dept[['from_date_t', 'to_date_t']].astype(str).agg(' ➝ '.join, axis=1)

df_title_dept = df_title_dept.sort_values(by=['employee_id', 'from_date_t'])

df_title_counts = df_title_dept.groupby(['employee_id', 'department_id'])['title'].nunique().reset_index()
df_title_counts.rename(columns={'title': 'num_titles_in_dept'}, inplace=True)

promoted = df_title_counts[df_title_counts['num_titles_in_dept'] > 1]

if 'dept_name' not in df_title_counts.columns:
    df_title_counts = df_title_counts.merge(dfD[['department_id', 'dept_name']], on='department_id', how='left')


dfE['hire_date'] = pd.to_datetime(dfE['hire_date'])  
dfE['hire_month'] = dfE['hire_date'].dt.month
dfE['hire_year'] = dfE['hire_date'].dt.year
dfE['hire_season'] = dfE['hire_date'].dt.month % 12 // 3 + 1

# خريطة للفصول
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
dfE['hire_season'] = dfE['hire_season'].map(season_map)
# استخدام FacetGrid لعرض الهستوجرام لكل قسم
st.subheader("📊 Number of Different Job Titles per Employee within Department")

fig1 = plt.figure(figsize=(16, 10))
g = sns.FacetGrid(df_title_counts, col="dept_name", col_wrap=4, sharex=False, sharey=False, height=4)
g.map_dataframe(sns.histplot, x='num_titles_in_dept', 
                bins=range(1, df_title_counts['num_titles_in_dept'].max() + 2), kde=False)
g.set_titles(col_template="{col_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("📊 Number of Different Job Titles per Employee within Department", fontsize=16)
st.pyplot(g.fig)

# 2️⃣ - عدد التعيينات الجديدة لكل شهر
st.subheader("📅 Number of New Employees by Month")

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.countplot(x='hire_month', data=dfE, palette='viridis', ax=ax2)
ax2.set_title('📅 Number of New Employees by Month')
ax2.set_xlabel('Month')
ax2.set_ylabel("Number of Appointments")
ax2.set_xticks(ticks=range(0, 12))
ax2.set_xticklabels(
    ['January', 'February', 'March', 'April', 'May', 'June',
     'July', 'August', 'September', 'October', 'November', 'December'],
    rotation=45
)
ax2.grid(True)
st.pyplot(fig2)
