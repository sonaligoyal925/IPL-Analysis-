import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="IPL Dashboard", layout="wide")
st.title("🏏 IPL Data Analysis Dashboard")

# Load data
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

# Sidebar filters
season = st.sidebar.selectbox("Select Season", sorted(matches['season'].unique(), reverse=True))
team = st.sidebar.selectbox("Select Team", sorted(matches['team1'].unique()))

# Section 1: Matches per Season
st.subheader("📅 Matches Per Season")
match_count = matches['season'].value_counts().sort_index()
fig, ax = plt.subplots()
sns.lineplot(x=match_count.index, y=match_count.values, marker="o", ax=ax)
ax.set_xlabel("Season")
ax.set_ylabel("No. of Matches")
st.pyplot(fig)

# Section 2: Toss Decision Analysis
st.subheader("🧠 Toss Decision Analysis")
toss_decision = matches.groupby(['toss_decision', 'season']).size().unstack()
st.bar_chart(toss_decision)

# Section 3: Team Wins by Season
st.subheader(f"🏆 Wins by {team}")
team_data = matches[matches['winner'] == team]
wins_by_season = team_data['season'].value_counts().sort_index()
fig2, ax2 = plt.subplots()
sns.barplot(x=wins_by_season.index, y=wins_by_season.values, ax=ax2, palette="viridis")
ax2.set_ylabel("Wins")
ax2.set_xlabel("Season")
st.pyplot(fig2)

# Section 4: Top 10 Players (Runs)
st.subheader("🔥 Top 10 Run Scorers")
top_batsmen = deliveries.groupby("batsman")["batsman_runs"].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_batsmen)

# Section 5: Chasing vs Defending Win Count
st.subheader("⚔️ Chasing vs Defending")
matches["win_type"] = matches["win_by_runs"].apply(lambda x: "Defend" if x > 0 else "Chase")
win_type_counts = matches["win_type"].value_counts()
st.pie_chart(win_type_counts)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Author: [Your Name](https://linkedin.com/in/yourname)")

