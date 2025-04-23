import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

st.title("IPL Data Analysis Dashboard")

# --- Data Loading and Cleaning ---
@st.cache_data
def load_data():
    df = pd.read_csv("matches.csv")
    df_del = pd.read_csv("deliveries.csv")
    # Fill missing cities
    df.loc[(df['city'].isna()) & (df['venue'] == 'Sharjah Cricket Stadium'), 'city'] = 'Sharjah'
    df.loc[(df['city'].isna()) & (df['venue'] == 'Dubai International Cricket Stadium'), 'city'] = 'Dubai'
    df.replace({'season': {"2020/21": "2020", "2009/10": "2010", "2007/08": "2008"}}, inplace=True)
    team_map = {
        "Mumbai Indians": "Mumbai Indians",
        "Chennai Super Kings": "Chennai Super Kings",
        "Kolkata Knight Riders": "Kolkata Knight Riders",
        "Royal Challengers Bangalore": "Royal Challengers Bangalore",
        "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
        "Rajasthan Royals": "Rajasthan Royals",
        "Kings XI Punjab": "Kings XI Punjab",
        "Punjab Kings": "Kings XI Punjab",
        "Sunrisers Hyderabad": "Sunrisers Hyderabad",
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Delhi Capitals": "Delhi Capitals",
        "Delhi Daredevils": "Delhi Capitals",
        "Gujarat Titans": "Gujarat Titans",
        "Gujarat Lions": "Gujarat Titans",
        "Lucknow Super Giants": "Lucknow Super Giants",
        "Pune Warriors": "Pune Warriors",
        "Rising Pune Supergiant": "Pune Warriors",
        "Rising Pune Supergiants": "Pune Warriors",
        "Kochi Tuskers Kerala": "Kochi Tuskers Kerala"
    }
    for col in ['team1', 'team2', 'winner', 'toss_winner']:
        df[col] = df[col].map(team_map)
    for col in ['batting_team', 'bowling_team']:
        df_del[col] = df_del[col].map(team_map)
    return df, df_del

df, df_del = load_data()

# --- 1. Trend of Total Matches Over Seasons ---
st.header("Trend of Total Matches Over Seasons")
season_counts = df['season'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(season_counts.index, season_counts.values, marker='o', linestyle='-', color='blue')
ax.set_title('Trend of Total Matches Over Seasons', fontsize=16)
ax.set_xlabel('Season', fontsize=12)
ax.set_ylabel('Number of Matches', fontsize=12)
for i, value in enumerate(season_counts.values):
    ax.text(season_counts.index[i], value, str(value), fontsize=10, ha='center', va='bottom')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(fig)
plt.close(fig)

# --- 2. Number of Matches Hosted by Each City ---
st.header("Number of Matches Hosted by Each City")
city_counts = df['city'].replace(np.nan, 'Unknown').value_counts()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=city_counts.index, y=city_counts.values, palette='viridis', ax=ax)
ax.set_title('Number of Matches Hosted by Each City', fontsize=16)
ax.set_xlabel('City', fontsize=12)
ax.set_ylabel('Number of Matches', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 3. Distribution of Target Runs ---
st.header("Distribution of Target Runs")
if 'target_runs' in df.columns:
    mean_target_runs = df['target_runs'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['target_runs'].dropna(), bins=30, kde=True, color='green', ax=ax)
    ax.axvline(mean_target_runs, color='red', linestyle='--', label=f'Mean: {mean_target_runs:.2f}')
    ax.set_title('Distribution of Target Runs', fontsize=16)
    ax.set_xlabel('Target Runs', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- 4. Trend of Average Target Runs by Season ---
    st.header("Trend of Average Target Runs by Season")
    avg_run_byseason = pd.pivot_table(data=df, index='season', values='target_runs', aggfunc='mean').reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=avg_run_byseason, x='season', y='target_runs', marker='o', color='green', ax=ax)
    ax.set_title('Trend of Average Target Runs by Season', fontsize=16)
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Average Target Runs', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- 5. Distribution of Match Results ---
st.header("Distribution of Match Results")
result_distribution = df['result'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=result_distribution.index, y=result_distribution.values, palette='viridis', ax=ax)
ax.set_title('Distribution of Match Results', fontsize=16)
ax.set_xlabel('Result', fontsize=12)
ax.set_ylabel('Number of Matches', fontsize=12)
for index, value in enumerate(result_distribution):
    ax.text(index, value, f'{value}', ha='center', va='bottom')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 6. Distribution of Result Margin in Matches Won by Runs ---
st.header("Distribution of Result Margin in Matches Won by Runs")
runs_margin = df[df['result'] == 'runs']['result_margin']
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(runs_margin, bins=20, kde=True, color='blue', ax=ax)
ax.set_title('Distribution of Result Margin in Matches Won by Runs', fontsize=16)
ax.set_xlabel('Result Margin Runs', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
plt.grid()
st.pyplot(fig)
plt.close(fig)

# --- 7. Distribution of Result Margin in Matches Won by Wickets ---
st.header("Distribution of Result Margin in Matches Won by Wickets")
wickets_margin = df[df['result'] == 'wickets']['result_margin']
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(wickets_margin, bins=20, kde=True, color='skyblue', ax=ax)
ax.set_title('Distribution of Result Margin in Matches Won by Wickets', fontsize=16)
ax.set_xlabel('Result Margin Wickets', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
plt.grid()
st.pyplot(fig)
plt.close(fig)

# --- 8. Average Result Margin by Season (Runs vs Wickets) ---
st.header("Average Result Margin by Season (Runs vs Wickets)")
result_margins = df[(df['result'] == 'runs') | (df['result'] == 'wickets')]
avg_result = pd.pivot_table(
    data=result_margins,
    index='season',
    columns='result',
    values='result_margin',
    aggfunc='mean'
).reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=avg_result, x='season', y='runs', marker='o', label='Average Margin Runs', color='green', ax=ax)
sns.lineplot(data=avg_result, x='season', y='wickets', marker='o', label='Average Margin Wickets', color='red', ax=ax)
ax.set_title('Average Result Margin by Season (Runs vs Wickets)', fontsize=16)
ax.set_xlabel('Season', fontsize=14)
ax.set_ylabel('Average Result Margin', fontsize=14)
ax.legend(title='Result Type')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 9. Percentage of Matches Won by Toss Decision ---
st.header("Percentage of Matches Won by Toss Decision")
df_clean = df.dropna(subset=['winner'])
wins_by_decision = df_clean.groupby('toss_decision').size().reset_index(name='wins')
total_matches = df_clean.shape[0]
wins_by_decision['percentage'] = (wins_by_decision['wins'] / total_matches) * 100
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=wins_by_decision, x='toss_decision', y='percentage', palette='Set1', ax=ax)
ax.set_title('Percentage of Matches Won by Toss Decision', fontsize=16)
ax.set_xlabel('Toss Decision (Bat First vs Field First)', fontsize=14)
ax.set_ylabel('Percentage of Wins (%)', fontsize=14)
for index, row in wins_by_decision.iterrows():
    ax.text(index, row['percentage'] + 1, f'{row["percentage"]:.1f}%', ha='center', fontsize=12)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 10. Number of Super Overs Over Time ---
st.header("Number of Super Overs Over Time")
super_over_matches = df[df['super_over'] == 'Y']
super_over_by_season = super_over_matches.groupby('season').size().reset_index(name='super_over_count')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(super_over_by_season['season'], super_over_by_season['super_over_count'], marker='o', color='g')
ax.set_title('Number of Super Overs Over Time')
ax.set_xlabel('Season')
ax.set_ylabel('Number of Super Overs')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(fig)
plt.close(fig)

# --- 11. Top 5 Umpires by Match Counts ---
st.header("Top 5 Umpires by Match Counts")
umpire_counts = df['umpire1'].value_counts().add(df['umpire2'].value_counts(), fill_value=0).sort_values(ascending=False)
top_umpires = umpire_counts.head(5)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_umpires.index, y=top_umpires.values, palette='viridis', ax=ax)
ax.set_title('Top 5 Umpires by Match Counts', fontsize=16)
ax.set_xlabel('Umpires', fontsize=14)
ax.set_ylabel('Number of Matches', fontsize=14)
plt.xticks(rotation=45)
for index, value in enumerate(top_umpires.values):
    ax.text(index, value + 1, int(value), ha='center', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 12. Winning Contribution of Teams in IPL Matches (Pie Chart) ---
st.header("Winning Contribution of Teams in IPL Matches")
total_matches = (df['team1'].value_counts() + df['team2'].value_counts()).rename_axis('Team').reset_index(name="Total_Match_Played")
total_wins = df['winner'].value_counts().rename_axis('Team').reset_index(name='Total_Wins')
team_chasing = df[df['toss_decision'] == 'field']['winner'].value_counts().rename_axis('Team').reset_index(name='Chasing_Wins')
team_batting_first = df[df['toss_decision'] == 'bat']['winner'].value_counts().rename_axis('Team').reset_index(name='Batting_First_Wins')
merged_df = total_matches.merge(total_wins,on='Team',how='outer') \
    .merge(team_chasing, on='Team', how='outer') \
    .merge(team_batting_first, on='Team', how='outer')
merged_df= merged_df.fillna(0)
merged_df.sort_values(by='Total_Match_Played', ascending=False, inplace=True)
fig, ax = plt.subplots(figsize=(8, 8))
explode = [0.1] * len(merged_df)
ax.pie(
    merged_df['Total_Wins'],
    labels=merged_df['Team'],
    autopct='%1.1f%%',
    startangle=50,
    colors=plt.cm.Blues(np.linspace(0.3, 0.9, len(merged_df))),
    wedgeprops={'edgecolor': 'black'},
    explode=explode,
    pctdistance=0.85,
    labeldistance=1.05
)
ax.set_title('Winning Contribution of Teams in IPL Matches')
st.pyplot(fig)
plt.close(fig)

# --- 13. Percentage of Wins by Team ---
st.header("Percentage of Wins by Team")
percentage_df = merged_df.copy()
percentage_df['Total_Wins'] = (percentage_df['Total_Wins'] / percentage_df['Total_Match_Played']) * 100
percentage_df['Chasing_Wins'] = (percentage_df['Chasing_Wins'] / merged_df['Total_Wins']) * 100
percentage_df['Batting_First_Wins'] = (percentage_df['Batting_First_Wins'] / merged_df['Total_Wins']) * 100
melted_df = percentage_df.sort_values(by='Total_Wins', ascending=False).melt(id_vars='Team',
                                value_vars=['Total_Wins', 'Chasing_Wins', 'Batting_First_Wins'],
                                var_name='Win_Type',
                                value_name='Percentage')
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=melted_df, x='Team', y='Percentage', hue='Win_Type', palette='muted', ax=ax)
ax.set_xlabel('Teams', fontsize=14)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Percentage of Wins by Team', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 14. Toss and Match Outcomes by Team ---
st.header("Toss and Match Outcomes by Team")
total_matches = (df['team1'].value_counts() + df['team2'].value_counts()).rename_axis('Team').reset_index(name="Total_Match_Played")
toss_winners = df['toss_winner'].value_counts().rename_axis('Team').reset_index(name='Toss_Win')
team_wins_pivot = pd.pivot_table(
    data=df[df['toss_winner'] == df['winner']],
    index='winner',
    columns='toss_decision',
    aggfunc='size',
    fill_value=0
).rename_axis('Team').reset_index().rename(
    columns={'bat': 'Bat_first_win_After_Toss_Win', 'field': 'Chasing_Win_After_Toss_Win'}
)
toss_winner_matches = df[df['toss_winner'] == df['winner']]['winner'].value_counts().rename_axis('Team').reset_index(name='Toss_Win_Match_Win')
merged_df2 = (
    total_matches
    .merge(toss_winners, on='Team', how='outer')
    .merge(toss_winner_matches, on='Team', how='outer')
    .merge(team_wins_pivot, on='Team', how='outer')
)
merged_df2 = merged_df2.fillna(0)
sorted_df = merged_df2.set_index('Team').sort_values(by='Toss_Win_Match_Win', ascending=False)
melted_df2 = pd.melt(
    sorted_df.reset_index(),
    id_vars=['Team'],
    value_vars=['Toss_Win', 'Toss_Win_Match_Win', 'Bat_first_win_After_Toss_Win', 'Chasing_Win_After_Toss_Win'],
    var_name='Outcome',
    value_name='Count'
)
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(data=melted_df2, x='Team', y='Count', hue='Outcome', palette='Set1', ax=ax)
plt.xticks(rotation=45)
plt.title("Toss and Match Outcomes by Team", fontsize=14)
st.pyplot(fig)
plt.close(fig)

# --- 15. Toss and Match Outcomes by Team (Percentage) ---
st.header("Toss and Match Outcomes by Team (Percentage)")
percentage_df2 = merged_df2.copy()
percentage_df2['Toss_Win'] = (percentage_df2['Toss_Win'] / merged_df2['Total_Match_Played']) * 100
percentage_df2['Toss_Win_Match_Win'] = (percentage_df2['Toss_Win_Match_Win'] / merged_df2['Toss_Win']) * 100
percentage_df2['Bat_first_win_After_Toss_Win'] = (percentage_df2['Bat_first_win_After_Toss_Win'] / merged_df2['Toss_Win_Match_Win']) * 100
percentage_df2['Chasing_Win_After_Toss_Win'] = (percentage_df2['Chasing_Win_After_Toss_Win'] / merged_df2['Toss_Win_Match_Win']) * 100
columns_to_plot = ['Toss_Win', 'Toss_Win_Match_Win', 'Bat_first_win_After_Toss_Win', 'Chasing_Win_After_Toss_Win']
n_rows = len(columns_to_plot)
fig, axes = plt.subplots(n_rows, 1, figsize=(12, n_rows * 4))
for i, col in enumerate(columns_to_plot):
    sorted_df = percentage_df2.sort_values(by=col, ascending=True)
    sns.barplot(x='Team', y=col, data=sorted_df, palette='viridis', ax=axes[i])
    axes[i].set_title(f'{col} by Team', fontsize=14)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90, fontsize=10)
    axes[i].set_ylabel(f"{col} (%)", fontsize=12)
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 16. Highest Score by Each Team in IPL ---
st.header("Highest Score by Each Team in IPL")
pivot_table = pd.pivot_table(
    data=df_del,
    index='match_id',
    columns='batting_team',
    values='total_runs',
    aggfunc='sum',
    fill_value=0
)
melted_df = pivot_table.reset_index().melt(id_vars='match_id', var_name='batting_team', value_name='total_runs')
filtered_df = melted_df[melted_df['total_runs'] > 0]
highest_scores = filtered_df.groupby('batting_team')['total_runs'].max().reset_index()
highest_scores_sorted = highest_scores.sort_values(by='total_runs', ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='batting_team', y='total_runs', data=highest_scores_sorted, palette='viridis', ax=ax)
for index, value in enumerate(highest_scores_sorted['total_runs']):
    ax.text(index, value + 2, f'{value}', ha='center', fontsize=12)
ax.set_xlabel('Team', fontsize=14)
ax.set_ylabel('Highest Score (Runs)', fontsize=14)
ax.set_title('Highest Score by Each Team in IPL', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 17. Number of 200+ Runs Innings by Team in IPL ---
st.header("Number of 200+ Runs Innings by Team in IPL")
high_scores_df = filtered_df[filtered_df['total_runs'] > 200]
team_200_plus_count = high_scores_df.groupby('batting_team')['total_runs'].count().reset_index()
team_200_plus_count.columns = ['Team', '200+ Runs Count']
team_200_plus_count_sorted = team_200_plus_count.sort_values(by='200+ Runs Count', ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Team', y='200+ Runs Count', data=team_200_plus_count_sorted, palette='coolwarm', ax=ax)
for index, value in enumerate(team_200_plus_count_sorted['200+ Runs Count']):
    ax.text(index, value + 0.2, str(value), ha='center', fontsize=12)
ax.set_xlabel('Team', fontsize=14)
ax.set_ylabel('Number of 200+ Scores', fontsize=14)
ax.set_title('Number of 200+ Runs Innings by Team in IPL', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 18. Trend of 200+ Runs Innings in IPL (2008-2024) ---
st.header("Trend of 200+ Runs Innings in IPL (2008-2024)")
season_df = df[['id', 'season']]
season_merged = pd.merge(season_df, filtered_df, left_on='id', right_on='match_id', how='inner')
season_merged_filter = season_merged[season_merged['total_runs'] > 200]
seasonwise200 = season_merged_filter.groupby('season')['total_runs'].count()
seasonwise200_sorted = seasonwise200.sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=seasonwise200_sorted.index, y=seasonwise200_sorted.values, marker='o', color='b', ax=ax)
for index, value in enumerate(seasonwise200_sorted.values):
    ax.text(index, value + 0.2, str(value), ha='center', fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
ax.set_xlabel('Season', fontsize=14)
ax.set_ylabel('Number of 200+ Runs Matches', fontsize=14)
ax.set_title('Trend of 200+ Runs Innings in IPL (2008-2024)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- 19. Trend of Highest Scores in Each IPL Season (2008-2024) ---
st.header("Trend of Highest Scores in Each IPL Season (2008-2024)")
seasonwise_maxscore = season_merged_filter[['season', 'total_runs']].groupby('season')['total_runs'].max().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=seasonwise_maxscore, x='season', y='total_runs', marker='o', ax=ax)
for index, row in seasonwise_maxscore.iterrows():
    ax.text(row['season'], row['total_runs'], row['total_runs'],
             horizontalalignment='left', size='medium', color='black', weight='semibold')
ax.set_title('Trend of Highest Scores in Each IPL Season (2008-2024)')
ax.set_xlabel('Season')
ax.set_ylabel('Highest Score')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
