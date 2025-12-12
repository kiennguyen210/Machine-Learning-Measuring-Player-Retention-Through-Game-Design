# Measuring Player Retention Through Game Design | Python, Machine Learning
<img width="1024" height="500" alt="image" src="https://github.com/user-attachments/assets/78fa1552-9a54-4d19-9e43-f10728934a3d" />

*Evaluating whether changing the position of the first progression gate from level 30 to level 40 affects player retention in the mobile game Cookie Cats.*

**Author:** Nguyễn Duy Kiên

**Date:** October 2025

**Tools Used:** Python, Machine Learning

---

## 📑 Table of Contents  
1. [📌 Background & Overview](#1--background--overview)    
2. [📂 Dataset Description & Data Structure](#2--dataset-description--data-structure)    
3. [🔍 Exploratory Data Analysis (EDA)](#3--exploratory-data-analysis-eda)    
4. [🧑‍💻 Hypothesis Testing](#4-%E2%80%8D-hypothesis-testing)    
5. [🤖 Predictive Modeling: Logistic Regression](5--predictive-modeling-logistic-regression)    
6. [💡 Insights & Recommendations](6--insights--recommendations)    

---

## 1. 📌 Background & Overview 

### 1.1. Objective:
### 📖 What is this project about? What Business Question will it solve?

🎯 Main Business Question

**Objective:** Does moving the first gate from level 30 to level 40 affect player retention?

📘 Project Overview

The project performs in-depth data analysis (including exploratory analysis, Chi-square statistical testing, Bootstrapped Confidence Intervals, and logistic regression modeling) on the data from an A/B test.

Regarding the A/B Test: It tests the impact of a design change: moving the first progression gate from Level 30 (Group A: gate_30) to Level 40 (Group B: gate_40).Each group contains over 45,000 randomly assigned players.The key metrics analyzed are: Day 1 Retention Rate and Day 7 Retention Rate.

💡 Business Questions this project answers

✔️ Does moving the first gate from level 30 to level 40 affect player retention (Day 1 and Day 7)?

✔️ Which change (gate_30 or gate_40) results in better player retention?

✔️ How can behavioral data guide game design choices and support evidence-based iteration in free-to-play game development?

### 1.2. Who is this project for?  

👤 Mobile Game Designers & Product Development Teams: To inform design decisions on features (like progression gate mechanics) aimed at optimizing player engagement and retention.

👤 Data Analysts & Data Scientists: To review the statistical analysis methods ($\chi^2$ test, Bootstrap, Logistic Regression) applied to an A/B test scenario.

👤 Decision-makers & Stakeholders: To understand the key drivers of player engagement and support data-driven business strategies in the gaming industry.

---

## 2. 📂 Dataset Description & Data Structure

### 2.1. Data Overview

The dataset used in this analysis was collected from an A/B test run within the mobile game Cookie Cats. It includes anonymized behavioral data from over 90,000 players who installed the game and were randomly assigned to one of two groups, each experiencing a different game design condition.

## 2.2. Table Schema

| Column Name    | Data Type | Description                                                               |
|----------------|-----------|---------------------------------------------------------------------------|
| userid         | int64     | A unique identifier for each player                                       |
| version        | object    | The A/B test group — either gate_30 (control) or gate_40 (treatment)      |
| sum_gamerounds | int64     | Total number of game rounds played in the first 7 days after install      |
| retention_1    | bool      | Boolean value indicating whether the player returned the next day (Day 1) |
| retention_7    | bool      | Boolean value indicating whether the player returned a week later (Day 7) |

---

## 3. 🔍 Exploratory Data Analysis (EDA)

### 3.1. Take a look at our data

<details>
  <summary>
    Click here to see detail
  </summary>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```

```python
df.head()
```
|   | userid | version | sum_gamerounds | retention_1 | retention_7 |
|---|--------|---------|----------------|-------------|-------------|
| 0 | 116    | gate_30 | 3              | False       | False       |
| 1 | 337    | gate_30 | 38             | True        | False       |
| 2 | 377    | gate_40 | 165            | True        | False       |
| 3 | 483    | gate_40 | 1              | False       | False       |
| 4 | 488    | gate_40 | 179            | True        | True        |

```python
df.describe()
```
|       | userid       | sum_gamerounds |
|-------|--------------|----------------|
| count | 9.018900e+04 | 90189.000000   |
| mean  | 4.998412e+06 | 51.872457      |
| std   | 2.883286e+06 | 195.050858     |
| min   | 1.160000e+02 | 0.000000       |
| 25%   | 2.512230e+06 | 5.000000       |
| 50%   | 4.995815e+06 | 16.000000      |
| 75%   | 7.496452e+06 | 51.000000      |
| max   | 9.999861e+06 | 49854.000000   |

```python
df.info()
```
| # | Column         | Non-Null Count | Dtype  |
|---|----------------|----------------|--------|
| 0 | userid         | 90189 non-null | int64  |
| 1 | version        | 90189 non-null | object |
| 2 | sum_gamerounds | 90189 non-null | int64  |
| 3 | retention_1    | 90189 non-null | bool   |
| 4 | retention_7    | 90189 non-null | bool   |

</details>

Overall dataset:

1. The dataset contains 90,189 total player records.
2. There are no missing values across the key columns.

### 3.2. EDA

#### 1. Number of Players in Each A/B Test Group

```python
plt.figure(figsize=(10, 6))
df['version'].value_counts().plot(kind='bar')
plt.title("Number of Players in Each A/B Test Group")
plt.xlabel("Test Group")
plt.ylabel("Player Count")
plt.xticks(rotation=0)
plt.show()
```

<img width="868" height="547" alt="image" src="https://github.com/user-attachments/assets/a73e6cd7-ce2a-4bb6-b330-c4c4befbd97e" />

**Insights:**

Both groups have roughly equal numbers of players (around 45,000 each).

This indicates a balanced random assignment, which is essential for ensuring that observed differences in retention or gameplay are attributable to the gate change, not to group size or selection bias.

Since the groups are balanced, we can confidently proceed with comparing behavioral outcomes like retention and game rounds played.

#### 2. Distribution of Game Round

```python
plt.figure(figsize=(10, 6))
sns.histplot(data = df, x = 'sum_gamerounds', bins=100, kde = True)
plt.title("Distribution of Game Rounds")
plt.xlabel("Game Rounds")
plt.ylabel("Frequency")
plt.xlim(0, 500)
plt.show()
```

<img width="890" height="547" alt="image" src="https://github.com/user-attachments/assets/c788e23f-6668-44cf-9688-ccbe0b88c247" />

**Insights:**

The distribution is heavily right-skewed, meaning:

- A large number of players played very few rounds.
- A small number of highly engaged players played hundreds (even thousands) of rounds.

This type of distribution is common in free-to-play mobile games, where player activity follows a power-law: most players churn early, and a few become very active.

#### 3. Distribution of Game Rounds by Group by Violin Plot (Log Scale)

```python
plt.figure(figsize=(10, 6))
sns.violinplot(data = df, x = 'version', y = 'sum_gamerounds')
plt.yscale('log')
plt.title("Distribution of Game Rounds by Group (Log Scale)")
plt.xlabel("Test Group")
plt.ylabel("Game Rounds")
plt.show()
```

<img width="849" height="547" alt="image" src="https://github.com/user-attachments/assets/9e7b4b56-4288-49ad-aafc-bfd824586de6" />

**Insights:**

Both groups have a large density of players around 5–20 rounds.

The gate_30 group has a longer upper tail, meaning a few players played much more extensively than in the gate_40 group.

The spread of engagement is wider in the gate_30 group, hinting that earlier gating may create a challenge that hooks some players more - or filters more casual ones early.

#### 4. Retention Rate by Test Group

```python
# Group-wise average retention rates
ret_1 = df.groupby("version")["retention_1"].mean()
ret_7 = df.groupby("version")["retention_7"].mean()
print(" Day 1 Retention:\n", ret_1)
print("\n Day 7 Retention:\n", ret_7)
```

| Day 1 Retention: |          |
|------------------|----------|
| gate_30          | 0.448188 |
| gate_40          | 0.442283 |
| Day 7 Retention: |          |
| gate_30          | 0.190201 |
| gate_40          | 0.182000 |

**Insights:**

Players in the gate_30 group (gate appears earlier) show slightly higher retention on both Day 1 and Day 7.

However, these differences are relatively small at face value:

- ~0.6% difference on Day 1

- ~0.8% difference on Day 7

These numbers hint that earlier gating might help retain users - possibly by providing a sense of progress or challenge early on.

#### 5. Outlier Detection with Boxplot

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data = df, x = 'sum_gamerounds')
plt.title("Outlier Detection with Boxplot")
plt.xlabel("Test Group")
plt.ylabel("Game Rounds")
plt.xlim(0, df["sum_gamerounds"].quantile(0.99))
plt.show()
```

<img width="824" height="547" alt="image" src="https://github.com/user-attachments/assets/7a05ccde-31d2-45ec-aef9-32ee9b442aea" />

**Insights:**

The boxplot above shows the distribution of game rounds played before applying any outlier treatment. To make the visualization clearer, we zoomed in to the 99th percentile. This reveals the main spread of the data without being distorted by extreme values on the far right.

The variable sum_gamerounds is highly skewed. While most players play only a small number of rounds, a few play thousands. These extreme values can distort statistical models and mislead visual interpretations.

To address this, we:

- Capped the values at the 99th percentile (winsorization).

- Created a log-transformed version for modeling and visualization.

#### 6. Outlier Handling and Feature Engineering

```python
# Cap the top 1% of values and log-transformed
cap = df['sum_gamerounds'].quantile(0.99)
df['gamerounds_capped'] = df['sum_gamerounds'].clip(upper=cap)
df['gamerounds_log'] = np.log1p(df['gamerounds_capped'])

#Effect of Outlier Capping on Game Rounds
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(x=df['sum_gamerounds'], ax=axes[0])
axes[0].set_title("Before Capping")
axes[0].set_xlim(0, 1000)

sns.boxplot(x=df['gamerounds_capped'], ax=axes[1])
axes[1].set_title("After Capping")
axes[1].set_xlim(0, 1000)

plt.suptitle("Effect of Outlier Capping on Game Rounds")
plt.tight_layout()
plt.show()
```

<img width="1189" height="495" alt="image" src="https://github.com/user-attachments/assets/650dc7dc-aee6-4c10-8b8a-4db35e2d7434" />

---

## 4. 🧑‍💻 Hypothesis Testing

---

## 5. 🤖 Predictive Modeling: Logistic Regression

---

## 6. 💡 Insights & Recommendations
