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
5. [🤖 Predictive Modeling: Logistic Regression](#5--predictive-modeling-logistic-regression)    
6. [💡 Insights & Recommendations](#6--insights--recommendations)    

---

## 1. 📌 Background & Overview 

### 1.1. Objective:
### 📖 What is this project about? What Business Question will it solve?

#### 🎯 Main Business Question

Does moving the first gate from level 30 to level 40 affect player retention?

#### 📘 Project Overview

The project performs in-depth data analysis (including exploratory analysis, Chi-square statistical testing, Bootstrapped Confidence Intervals, and logistic regression modeling) on the data from an A/B test.

Regarding the A/B Test: 

- It tests the impact of a design change: moving the first progression gate from Level 30 (Group A: gate_30) to Level 40 (Group B: gate_40). 

- Each group contains over 45,000 randomly assigned players. 

- The key metrics analyzed are: Day 1 Retention Rate and Day 7 Retention Rate.

#### 💡 Business Questions this project answers

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

### 4.1. Hypothesis Testing: Chi-Squared Test

To determine whether the difference in retention rates between the two groups is statistically significant, we use the Chi-Squared Test for Independence.

**Hypotheses:**

For each retention test (Day 1 and Day 7), we define:

- Null Hypothesis (H0): There is no difference in retention rates between gate_30 and gate_40

- Alternative Hypothesis (H1): There is a difference in retention rates between the groups

The Chi-Squared test calculates a test statistic and a p-value.

If the p-value is less than a chosen significance level (commonly alpha = 0.05), we reject the null hypothesis and conclude that the difference is statistically significant.

#### 1. Data Preparation

```python
day1_table = pd.crosstab(df['version'], df['retention_1'])
day7_table = pd.crosstab(df['version'], df['retention_7'])
```

#### 2. Perform chi-squared test

```python
chi2_day1, p1, _, _ = chi2_contingency(day1_table)
chi2_day7, p7, _, _ = chi2_contingency(day7_table)
print("Chi-Squared Test Results")
print("-------------------------")
print(f"Day 1 retention: p-value = {p1:.5f}")
print(f"Day 7 retention: p-value = {p7:.5f}")
```

Chi-Squared Test Results:

| Day 1 retention | p-value = 0.07550 |
|-----------------|-------------------|
| Day 7 retention | p-value = 0.00160 |

#### 3. Interpretation of Chi-Squared Test Results:

After running the Chi-Squared test for both Day 1 and Day 7 retention, we obtained the following p-values:

- Day 1 Retention: p-value = 0.0755

- Day 7 Retention: p-value = 0.0016

**How to interpret these values:** We use a significance threshold of alpha = 0.05.

- Day 1 (p = 0.0755):
Since the p-value is greater than 0.05, we fail to reject the null hypothesis.
This means there is no statistically significant difference in Day 1 retention between the two groups. The slight difference we observed might be due to random chance.

- Day 7 (p = 0.0016):
Since the p-value is less than 0.05, we reject the null hypothesis.
This indicates a statistically significant difference in Day 7 retention between gate_30 and gate_40.

**Conclusion:**
The placement of the progression gate does not appear to impact short-term (Day 1) retention, but it does have a meaningful impact on long-term (Day 7) retention. Players who encountered the gate earlier (at level 30) were more likely to return after one week.

### 4.2. Bootstrapped Confidence Intervals

We apply this to Day 7 retention rates for each group. For each resample, we compute the mean retention, then calculate the 2.5th and 97.5th percentiles as our 95% confidence interval.

#### 1. Define Bootstrap

```python
def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    boot_means = []
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper
```

#### 2. Apply on our Dataset

```python
boot_30 = bootstrap_ci(df[df.version == 'gate_30']['retention_7'])
boot_40 = bootstrap_ci(df[df.version == 'gate_40']['retention_7'])
print("Day 7 Retention - gate_30 CI:", boot_30)
print("Day 7 Retention - gate_40 CI:", boot_40)
```

| Day 7 Retention - gate_30 CI | 0.18659899328859061 | 0.19391498881431768 |
|------------------------------|---------------------|---------------------|
| Day 7 Retention - gate_40 CI | 0.178460726769109   | 0.18549539449097585 |

#### 3. Interpretation of Bootstrapping Results

- 95% Confidence Interval for gate_30: (e.g., 18.65%, 19.37%)

- 95% Confidence Interval for gate_40: (e.g., 17.84%, 18.56%)

Since these intervals do not overlap, we gain further evidence that the difference in Day 7 retention is real and consistent, not just a result of one sample.

This supports the conclusion that placing the progression gate earlier (level 30) improves long-term engagement.

---

## 5. 🤖 Predictive Modeling: Logistic Regression

Now that we’ve confirmed there is a significant difference in Day 7 retention between the two groups, we can take it one step further:
Can we predict which players are more likely to return after one week?

To answer this, we use logistic regression, a statistical model commonly used for binary classification problems.

### 5.1. Model Training & Evaluation

#### 1. Data Preparation

```python
# Encode version as binary
df['version_bin'] = df['version'].map({'gate_30': 0, 'gate_40': 1})
# Select features and target
X = df[['version_bin', 'gamerounds_log', 'retention_1']]
y = df['retention_7']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 2. Apply Model

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

#### 3. Model Evaluation and Interpretation

```python
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

| Metric    | False (Did Not Return) | True (Returned) |
|-----------|------------------------|-----------------|
| Precision | 0.90                   | 0.72            |
| Recall    | 0.95                   | 0.52            |
| F1-score  | 0.92                   | 0.60            |
| Accuracy  | 87% overall            |                 |

The logistic regression model was trained to predict whether a player would return on Day 7 based on their behavior in the first few days.

- Accuracy (87%): The model correctly classified 87% of players.

- Precision (Returned = 0.72): When the model predicts that a player will return, it's right 72% of the time.

- Recall (Returned = 0.52): The model catches 52% of the actual returners — meaning it misses nearly half.

- F1-score balances both — showing that while the model performs well overall, it has room for improvement when it comes to detecting who will return.

#### 4. Insights and Recommendations

**Insights:**
The model is very good at identifying who won't return (high recall & precision for the "False" class). It is moderately effective at detecting players who will return, which is typical for imbalanced data (more non-returners than returners).

**Recommendations:** We may improve this further by:

- Adding more behavioral features (e.g., session duration, time between installs and first play)

- Handling class imbalance

### 5.2. Feature Importance Analysis

In logistic regression, feature importance is reflected in the magnitude and sign of the model coefficients:

- A positive coefficient means the feature increases the likelihood of the player returning on Day 7

- A negative coefficient means the feature decreases that likelihood

By plotting these coefficients, we can visually assess which behaviors or group assignments are most predictive.

#### 1. Create DataFrame of coefficients

```python
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)
```

#### 2. Plot Feature Importance

```python
plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficient", y="Feature", data=coef_df)
plt.axvline(0, color='gray', linestyle='--')
plt.title("Logistic Regression Feature Importance")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
```

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/8dfc951b-a508-4e60-b5d2-c7b2b0b445cb" />

#### 3. Interpretation of Feature Importance

The Logistic Regression model was trained to predict the likelihood of a player returning on Day 7 based on their initial behavior. The coefficients determine the influence of each feature on the prediction.

- gamerounds_log (Log-transformed total game rounds played): This feature displays the largest positive coefficient (approx. +1.45). This strong positive value indicates that the volume of gameplay (number of rounds completed) is the most powerful factor increasing the likelihood of a player returning on Day 7.

- retention_1 (Day 1 Retention): This feature has a negligible positive coefficient (approx. +0.05). Returning on Day 1 does positively influence Day 7 retention, but its effect is minimal compared to other factors.

- version_bin (Test Group Assignment): This feature shows a small negative coefficient (approx. -0.05). This confirms that the test group associated with this variable (likely the gate_40 group) slightly decreases the retention likelihood compared to the other group, but the impact is minimal.

#### 4. Insights and Recommendations

**Insights**

- Engagement Intensity Overcomes Group Assignment: The sheer magnitude of the gamerounds_log coefficient (approx. 1.45) dwarfs the influence of the test group assignment (version_bin, approx. -0.05). This reveals that player behavior and engagement intensity are overwhelmingly more important in predicting Day 7 retention than the specific in-game change being tested (the position of the gate).

- Quality of Play Matters More Than Quantity of Visits: The low coefficient for retention_1 suggests that simply opening the app on Day 1 is insufficient. True stickiness is driven by deep engagement, measured by the accumulated volume of gameplay.

**Recommendations**

1. Strategic Priority Shift: The team should shift its primary focus from marginally optimizing the gate position to maximizing early-game round completion.

2. Actionable Goal: Implement new features, missions, or reward loops that incentivize players to achieve higher game round counts within the first 1-3 days, as this is the biggest driver of long-term retention.

3. Model Validation: The model confirms that the gate_30 version is slightly superior (since the version_bin for gate_40 is negative), but this decision is minor compared to addressing the core engagement driver.

---

## 6. 💡 Insights & Recommendations

This analysis explored whether changing the placement of the first progression gate in the game Cookie Cats - from level 30 to level 40 - affects player retention.

We combined descriptive statistics, hypothesis testing, bootstrapping, and predictive modeling to draw the following conclusions:

### 6.1. Key findings

- Day 1 Retention: No statistically significant difference between groups (p = 0.0755).
  
- Day 7 Retention: Statistically significant improvement for players who saw the gate at level 30 (p = 0.0016).
  
- Bootstrapped confidence intervals confirmed that the retention improvement at Day 7 was robust and unlikely to be due to chance.
  
- Logistic regression modeling showed that early gameplay activity (gamerounds_log) and Day 1 retention were strong predictors of Day 7 retention.
  
- Being in the gate_40 group had a slight negative impact on long-term retention, suggesting that delaying the gate reduces player commitment.

### 6.2. Recommendations

Retain the gate at level 30, or experiment with even earlier placement (e.g., level 25), as it appears to increase long-term engagement without negatively impacting short-term retention.

**Additional opportunities:**

- Design early-game missions that boost game rounds and Day 1 retention
- Use these behavioral signals in a churn prediction model to guide LiveOps actions
- Extend A/B tests to explore other game mechanics (e.g., rewards, tutorials, pacing)
