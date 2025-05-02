import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# ============================ Project Title Page ================================================================================================================
print("="*70)
print("📊 COVID-19 Data Analysis & Visualization Project")
print("🎓 Course: CA2 Data Analysis Project")
print("🗂 Dataset Source: Official Government COVID Dataset")
print("="*70)

# ============================ Load and Prepare Dataset ================================================================================================================
df = pd.read_csv(r"C:\Users\paras\Downloads\COVID-19.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')  # Sorting based on Date

# ============================ Dataset Overview ================================================================================================================
print("\n🔍 Dataset Overview")
print("Shape of the matrix:", df.shape)
print("\n=== Column Names ===")
print(df.columns)
print("\n=== Data Types ===")
print(df.dtypes)
print("\n=== Missing Values ===")
print(df.isnull().sum())

# ============================ Descriptive Stats for Key Rates ===========================================================================================
print("\n📈 Descriptive Statistics")
print("\n=== Cases Rate - Total ===")
print("Mean:", df["Cases Rate - Total"].mean())
print("Median:", df["Cases Rate - Total"].median())
print("Mode:", df["Cases Rate - Total"].mode()[0])
print("Count:", df["Cases Rate - Total"].count())
print("Max:", df["Cases Rate - Total"].max())
print("Min:", df["Cases Rate - Total"].min())

print("\n=== Deaths Rate - Total ===")
print("Mean:", df["Deaths Rate - Total"].mean())
print("Median:", df["Deaths Rate - Total"].median())
print("Mode:", df["Deaths Rate - Total"].mode()[0])
print("Count:", df["Deaths Rate - Total"].count())
print("Max:", df["Deaths Rate - Total"].max())
print("Min:", df["Deaths Rate - Total"].min())

print("\n=== Hospitalizations Rate - Total ===")
print("Mean:", df["Hospitalizations Rate - Total"].mean())
print("Median:", df["Hospitalizations Rate - Total"].median())
print("Mode:", df["Hospitalizations Rate - Total"].mode()[0])
print("Count:", df["Hospitalizations Rate - Total"].count())
print("Max:", df["Hospitalizations Rate - Total"].max())
print("Min:", df["Hospitalizations Rate - Total"].min())

# ============================ Z-Test for Deaths vs Hospitalization ================================================================================================================
print("\n📊 Z-Test Between Deaths and Hospitalization Rates")
deaths_rate = df['Deaths Rate - Total']
hospitalizations_rate = df['Hospitalizations Rate - Total']
z_stat, p_value = stats.ttest_ind(deaths_rate.dropna(), hospitalizations_rate.dropna())
print(f"Z-statistic: {z_stat}, P-value: {p_value}")

# ============================ Line Plot - Trend Over Time ================================================================================================================
print("\n📉 Plotting COVID Trends Over Time")
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Cases Rate - Total'], label='Cases Rate', color='blue')
plt.plot(df['Date'], df['Deaths Rate - Total'], label='Deaths Rate', color='red')
plt.plot(df['Date'], df['Hospitalizations Rate - Total'], label='Hospitalizations Rate', color='green')
plt.title('COVID-19 Daily Rolling Rates Over Time')
plt.xlabel('Date')
plt.ylabel('Rate per 100K')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================ Objective 1 ============================================================================================================================================
print("\n📌 Objective 1: Monthly Average Cases Rate (Bar Chart)")
df['Month'] = df['Date'].dt.strftime('%b')
monthly_cases = df.groupby('Month')[["Cases Rate - Total"]].mean().reindex([
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Month', y='Cases Rate - Total', data=monthly_cases, hue='Month')
plt.title('Monthly Average COVID-19 Case Rate')
plt.xlabel('Month')
plt.ylabel('Average Case Rate')
plt.tight_layout()
plt.show()

# ============================ Objective 2 ============================================================================================================================================
print("\n📌 Objective 2: Age-wise Comparison (Pie Chart)")
avg_18_29 = df["Cases Rate - Age 18-29"].mean()
avg_60_plus = df["Cases Rate - Age 60-69"].mean()

labels = ['Age 18-29 (Younger Adults)', 'Age 60+ (Older Adults)']
sizes = [avg_18_29, avg_60_plus]
colors = ['#ff9999', '#66b3ff']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, startangle=140, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
plt.title('Proportion of Average Case Rates: Age 18-29 vs Age 60+')
plt.tight_layout()
plt.show()

# ============================ Objective 3 ============================================================================================================================================
print("\n📌 Objective 3: Race-wise Distribution (Donut Chart)")
race_cols = [
    'Cases Rate - Latinx', 'Cases Rate - Asian Non-Latinx',
    'Cases Rate - Black Non-Latinx', 'Cases Rate - White Non-Latinx',
    'Cases Rate - Other Race Non-Latinx'
]
race_means = [df[col].mean() for col in race_cols]

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(race_means, labels=race_cols, autopct='%1.1f%%', startangle=140)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')  # donut effect
plt.gca().add_artist(centre_circle)
plt.title('COVID-19 Case Rate Distribution by Race')
plt.tight_layout()
plt.show()

# ============================ Objective 4 ============================================================================================================================================
print("\n📌 Objective 4: Hospitalization vs Death (Scatter Plot)")
plt.figure(figsize=(10, 6))
plt.scatter(df["Hospitalizations Rate - Total"], df["Deaths Rate - Total"], color='teal', alpha=0.6)
plt.title("Hospitalizations vs. Death Rates")
plt.xlabel("Hospitalizations Rate - Total")
plt.ylabel("Deaths Rate - Total")
plt.tight_layout()
plt.show()

# ============================ Objective 5 ============================================================================================================================================
print("\n📌 Objective 5: Distribution of Case Rates (Histogram)")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Cases Rate - Total", hue="Month", bins=20, edgecolor="black")
plt.title("Distribution of COVID-19 Case Rates by Month")
plt.xlabel("Case Rate")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================ Objective 6 ============================================================================================================================================
print("\n📌 Objective 6: Correlation Analysis (Heatmap)")
correlation_matrix = df[['Cases Rate - Total', 'Deaths Rate - Total', 'Hospitalizations Rate - Total']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix Between Case, Death, and Hospitalization Rates")
plt.tight_layout()
plt.show()

# ============================ Project Summary ============================================================================================================================================
print("\n📋 Conclusion:")
print("✔ This project presented a clear visual and statistical overview of COVID-19 case dynamics.")
print("✔ It highlighted trends by month, age, and race, and explored relationships between hospitalizations and deaths.")
print("✔ Correlation analysis and distribution insights added depth to understanding pandemic impact.")

# ============================ Analysis Complete ============================================================================================================================================
print("\n✅ All analysis steps are successfully completed.")
