import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, SelectKBest
import matplotlib.pyplot as plt
import seaborn as sns

####################
# Data collection
####################
temp_anom_df = pd.read_csv("GLB.Ts+dSST.csv", skiprows=1)
ch4_df = pd.read_csv("ch4_annmean_gl.csv", skiprows=43)
co2_df = pd.read_csv("co2_annmean_mlo.csv", skiprows=43)
n2o_df = pd.read_csv("n2o_annmean_gl.csv", skiprows=43)
sf6_df = pd.read_csv("sf6_annmean_gl.csv", skiprows=43)
owid_df = pd.read_csv("owid-co2-data.csv")
volcanoes_df = pd.read_csv("volcanoes.csv", delimiter="	")

gases_df = pd.merge(co2_df, ch4_df, how='outer', on='year')
gases_df = pd.merge(gases_df, n2o_df, how='outer', on='year')
gases_df.columns = ['year', 'co2_mean', 'co2_unc',
                    'ch4_mean', 'ch4_unc', 'n2o_mean', 'n2o_unc']
gases_df = pd.merge(gases_df, sf6_df, how='outer', on='year')
gases_df.columns = ['year', 'co2_mean', 'co2_unc',
                    'ch4_mean', 'ch4_unc', 'n2o_mean', 'n2o_unc',
                    'sf6_mean', 'sf6_unc']
gases_df = gases_df.drop(columns=['co2_unc', 'ch4_unc', 'n2o_unc', 'sf6_unc'])

owid_df_pruned = owid_df[owid_df['country'] == 'World']
owid_df_pruned = owid_df_pruned.drop(columns=['country', 'iso_code'])

inputs_df = pd.merge(gases_df, owid_df_pruned, how='outer', on='year')

temp_anom_df_pruned = temp_anom_df.drop(columns=[
    'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct',
    'Nov','Dec','D-N','DJF','MAM','JJA','SON'])

temp_anom_df_pruned.rename(columns={'Year':'year'}, inplace=True)

full_df = pd.merge(inputs_df, temp_anom_df_pruned, how='outer', on='year')
full_df = pd.merge(full_df, volcanoes_df, how='outer', on='year')
full_df.rename(columns={'J-D':'Temperature Anomaly'}, inplace=True)

#######################
# Data preprocessing
#######################
selected_columns_df = full_df[['year', 'co2_mean', 'ch4_mean', 'n2o_mean',
                               'sf6_mean', 'co2', 'co2_growth_prct', 'methane',
                               'nitrous_oxide', 'primary_energy_consumption',
                               'Eruptions_Total', 'Volcanoes_Active',
                               'Temperature Anomaly']]

print(f"Shape of selected_columns_df before dropping NaNs: {selected_columns_df.shape}")

# Drop rows with at least 3 NaN values
threshold = selected_columns_df.shape[1] - 3

selected_columns_df_cleaned = selected_columns_df.dropna(thresh=threshold)

print(f"Shape of selected_columns_df after dropping rows with at least 3 NaNs: {selected_columns_df_cleaned.shape}")
selected_columns_df_cleaned_copy = selected_columns_df_cleaned.copy()
selected_columns_df_cleaned_copy.rename(columns={'co2_mean':'atmospheric co2 mean',
                                    'ch4_mean':'atmospheric ch4 mean',
                                    'n2o_mean':'atmospheric n2o mean',
                                    'sf6_mean':'atmospheric sf6 mean',
                                    'co2':'co2 emissions',
                                    'co2_growth_prct':'co2 emissions growth%',
                                    'nitrous_oxide':'nitrous oxide emissions',
                                    'methane':'methane emissions'},
                                   inplace=True)

# Apply linear interpolation/extrapolation to specific columns
selected_columns_df_cleaned_copy2 = selected_columns_df_cleaned_copy.copy()
selected_columns_df_cleaned_copy2['atmospheric ch4 mean'] = selected_columns_df_cleaned_copy['atmospheric ch4 mean'].interpolate(method='linear', limit_direction='both')
selected_columns_df_cleaned_copy2['atmospheric n2o mean'] = selected_columns_df_cleaned_copy['atmospheric n2o mean'].interpolate(method='linear', limit_direction='both')
selected_columns_df_cleaned_copy2['atmospheric sf6 mean'] = selected_columns_df_cleaned_copy['atmospheric sf6 mean'].interpolate(method='linear', limit_direction='both')

# Engineer percentage change columns
selected_columns_df_cleaned_copy2['ch4 emissions growth%'] = selected_columns_df_cleaned_copy2['methane emissions'].pct_change() * 100
selected_columns_df_cleaned_copy2['atmospheric co2 growth%'] = selected_columns_df_cleaned_copy2['atmospheric co2 mean'].pct_change() * 100
selected_columns_df_cleaned_copy2['atmospheric ch4 growth%'] = selected_columns_df_cleaned_copy2['atmospheric ch4 mean'].pct_change() * 100

# Rearranging the columns to put the dependent variable last
columns = selected_columns_df_cleaned_copy2.columns.tolist()
columns.remove('Temperature Anomaly') # Remove 'Temperature Anomaly' from its current position
columns.append('Temperature Anomaly') # Add 'Temperature Anomaly' to the end of the list

# Reindex the DataFrame with the new column order
selected_columns_df_cleaned_copy2 = selected_columns_df_cleaned_copy2[columns]

selected_columns_df_cleaned_copy2.fillna(0, inplace=True) # Fill in NaN %change values

######################
# Model Development
######################
independent_variables = selected_columns_df_cleaned_copy2.drop(columns=['year', 'Temperature Anomaly']).astype(float)
independent_variables = independent_variables.to_numpy()
dependent_variable = selected_columns_df_cleaned_copy2['Temperature Anomaly'].astype(float).to_numpy()

independent_variables = np.array(independent_variables)
dependent_variable = np.array(dependent_variable)

# Splitting into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    independent_variables, dependent_variable, test_size=0.3, random_state=2026)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Accessing model information
score = regressor.score(X_test, y_test)

# Print the information
print("R-squared score:", score)

##############################
# Root cause identification
##############################
feat_select = SelectKBest(score_func=f_regression, k=7)
pearson_columns = feat_select.fit_transform(selected_columns_df_cleaned_copy2.drop(columns=['year', 'Temperature Anomaly']).astype(float), dependent_variable)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(pearson_columns,
                                                    dependent_variable,
                                                    test_size=0.3,
                                                    random_state=2026)

fs_model = LinearRegression()
fs_model.fit(X_train, y_train)

print(feat_select.get_feature_names_out())  # displaying the selected features

## Investigating human-driven factors
human_driven_factors = selected_columns_df_cleaned_copy2[['co2 emissions',
                                                         'co2 emissions growth%',
                                                         'methane emissions',
                                                         'nitrous oxide emissions',
                                                         'primary_energy_consumption',
                                                         'atmospheric co2 growth%',
                                                         'atmospheric ch4 growth%',
                                                         'ch4 emissions growth%'
                                                         ]]
human_ind_vars = np.array(human_driven_factors)
X_train_human, X_test_human, y_train_human, y_test_human = train_test_split(human_ind_vars,
                                                                            dependent_variable,
                                                                            test_size=0.3,
                                                                            random_state=2026)

regressor_human = LinearRegression()
regressor_human.fit(X_train_human, y_train_human)

# Accessing model information
human_score = regressor_human.score(X_test_human, y_test_human)

# Print the information
print("Human-driven R-squared score:", human_score)

## Investigating natural factors
natural_factors = selected_columns_df_cleaned_copy2[['Eruptions_Total',
                                                     'Volcanoes_Active']]
nat_ind_vars = np.array(natural_factors)
X_train_nat, X_test_nat, y_train_nat, y_test_nat = train_test_split(nat_ind_vars,
                                                                    dependent_variable,
                                                                    test_size=0.3,
                                                                    random_state=2026)

regressor_nat = LinearRegression()
regressor_nat.fit(X_train_nat, y_train_nat)

# Accessing model information
nat_score = regressor_nat.score(X_test_nat, y_test_nat)

# Print the information
print("Natural factors R-squared score:", nat_score)

#######################
# Data visualization
#######################
df_for_plot = selected_columns_df_cleaned_copy2.copy()

# Explicitly replace '***' with NaN, then convert to numeric, and finally fill any NaNs with 0.
df_for_plot['Temperature Anomaly'] = pd.to_numeric(
    df_for_plot['Temperature Anomaly'].replace('***', np.nan), errors='coerce'
).fillna(0)

# Ensure the 'atmospheric co2 mean' column is also numeric in case of any subtle issues
df_for_plot['atmospheric co2 mean'] = pd.to_numeric(
    df_for_plot['atmospheric co2 mean'], errors='coerce'
).fillna(0)

## Regression scatter plots (features vs. temperature anomaly)
plt.figure(figsize=(10, 6))
sns.regplot(x='atmospheric co2 mean', y='Temperature Anomaly', data=df_for_plot, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Atmospheric CO2 Mean vs. Temperature Anomaly')
plt.xlabel('Atmospheric CO2 Mean (ppm)')
plt.ylabel('Temperature Anomaly')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='primary_energy_consumption', y='Temperature Anomaly', data=df_for_plot, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Primary Energy Consumption vs. Temperature Anomaly')
plt.xlabel('Primary Energy Consumption (terawatt-hours)')
plt.ylabel('Temperature Anomaly')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='methane emissions', y='Temperature Anomaly', data=df_for_plot, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Methane Emissions vs. Temperature Anomaly')
plt.xlabel('Methane Emissions (million tonnes)')
plt.ylabel('Temperature Anomaly')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='co2 emissions', y='Temperature Anomaly', data=df_for_plot, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('CO2 Emissions vs. Temperature Anomaly')
plt.xlabel('CO2 Emissions (million tonnes)')
plt.ylabel('Temperature Anomaly')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='nitrous oxide emissions', y='Temperature Anomaly', data=df_for_plot, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Nitrous Oxide Emissions vs. Temperature Anomaly')
plt.xlabel('Nitrous Oxide Emissions (million tonnes)')
plt.ylabel('Temperature Anomaly')
plt.grid(True)
plt.show()

## Time-series trends of gases vs. temperature
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot 'atmospheric co2 mean' on the first y-axis
color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Atmospheric CO2 Mean (ppm)', color=color)
ax1.plot(selected_columns_df_cleaned_copy2['year'], df_for_plot['atmospheric co2 mean'], color=color, label='Atmospheric CO2 Mean (ppm)')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Temperature Anomaly', color=color)
ax2.plot(selected_columns_df_cleaned_copy2['year'], df_for_plot['Temperature Anomaly'], color=color, label='Temperature Anomaly')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and legend
plt.title('Atmospheric CO2 Mean vs. Temperature Anomaly Over Time')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

fig.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot 'atmospheric ch4 mean' on the first y-axis
color = 'tab:green'
ax1.set_xlabel('Year')
ax1.set_ylabel('Atmospheric CH4 Mean (ppm)', color=color)
ax1.plot(selected_columns_df_cleaned_copy2['year'], selected_columns_df_cleaned_copy['atmospheric ch4 mean'], color=color, label='Atmospheric CH4 Mean (ppm)')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Temperature Anomaly', color=color)
ax2.plot(selected_columns_df_cleaned_copy2['year'], df_for_plot['Temperature Anomaly'], color=color, label='Temperature Anomaly')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and legend
plt.title('Atmospheric CH4 Mean vs. Temperature Anomaly Over Time')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

fig.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot 'atmospheric n2o mean' on the first y-axis
color = 'tab:purple'
ax1.set_xlabel('Year')
ax1.set_ylabel('Atmospheric N2O Mean (ppm)', color=color)
ax1.plot(selected_columns_df_cleaned_copy2['year'], selected_columns_df_cleaned_copy['atmospheric n2o mean'], color=color, label='Atmospheric N2O Mean (ppm)')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Temperature Anomaly', color=color)
ax2.plot(selected_columns_df_cleaned_copy2['year'], df_for_plot['Temperature Anomaly'], color=color, label='Temperature Anomaly')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and legend
plt.title('Atmospheric N2O Mean vs. Temperature Anomaly Over Time')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

fig.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot 'atmospheric sf6 mean' on the first y-axis
color = 'tab:orange'
ax1.set_xlabel('Year')
ax1.set_ylabel('Atmospheric SF6 Mean (ppm)', color=color)
ax1.plot(selected_columns_df_cleaned_copy2['year'], selected_columns_df_cleaned_copy['atmospheric sf6 mean'], color=color, label='Atmospheric SF6 Mean (ppm)')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Temperature Anomaly', color=color)
ax2.plot(selected_columns_df_cleaned_copy2['year'], df_for_plot['Temperature Anomaly'], color=color, label='Temperature Anomaly')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and legend
plt.title('Atmospheric SF6 Mean vs. Temperature Anomaly Over Time')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

fig.tight_layout()
plt.show()

## Feature Importance
independent_vars_df = selected_columns_df_cleaned_copy2.drop(columns=['year', 'Temperature Anomaly']).astype(float)
dependent_var = selected_columns_df_cleaned_copy2['Temperature Anomaly'].astype(float)

# Re-initialize SelectKBest with k='all' to get scores for all features
feat_select_rerun = SelectKBest(score_func=f_regression, k='all')

# Fit the selector to the independent and dependent variables
feat_select_rerun.fit(independent_vars_df, dependent_var)

# Extract the F-regression scores
f_regression_scores = feat_select_rerun.scores_

# Get the names of all independent features
all_feature_names = independent_vars_df.columns.tolist()

# Create a Series mapping all input feature names to their scores
all_feature_scores_series = pd.Series(f_regression_scores, index=all_feature_names)

# Convert to DataFrame and rename columns
feature_importance_df = all_feature_scores_series.reset_index()
feature_importance_df.columns = ['Feature', 'F_Score']

# Sort by F_Score in descending order
feature_importance_df = feature_importance_df.sort_values(by='F_Score', ascending=False)

# 1. Set the figure size to 14x7 inches for better readability.
plt.figure(figsize=(14, 7))

# 2. Create a bar plot using seaborn.barplot
sns.barplot(x='F_Score', y='Feature', data=feature_importance_df, palette='viridis', hue='Feature', legend=False)

# 3. Add a title to the plot
plt.title('F-Regression Scores of Selected Independent Features for Temperature Anomaly Prediction')

# 4. Label the x-axis and y-axis
plt.xlabel('F-Score')
plt.ylabel('Feature')

# 5. Adjust the plot layout to prevent labels from overlapping
plt.tight_layout()

# 6. Display the plot
plt.show()
