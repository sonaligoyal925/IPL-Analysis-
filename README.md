# IPL Analysis Project 

This project is an in-depth analysis of the Indian Premier League matches using Python, Pandas, Matplotlib, Seaborn, and Streamlit for interactive visualizations. It includes multiple visualizations and statistics, such as match trends, city hosting statistics, target runs distribution, result margins, toss-related insights, and more.

## Project Description

The objective of this project is to analyze and visualize key statistics from the IPL dataset, providing insights into various aspects of IPL matches, including:
- Trends over seasons
- Number of matches hosted by each city
- Distribution of target runs and match results
- Toss decision outcomes
- Super over analysis
- Umpire statistics
- Team performance analysis (matches played, wins, etc.)

Additionally, the project is deployed using **Streamlit**, providing a web interface for users to interact with the visualizations.

## Datasets

The project uses the following two datasets:
- **matches.csv**: Contains details about the matches, including teams, venues, results, and seasons.
- **deliveries.csv**: Contains information about individual deliveries in each match.

## Key Features

- **Season-wise Match Trend**: Analysis of the number of matches played per season.
- **City-wise Match Distribution**: Number of matches hosted by each city.
- **Target Runs Distribution**: Visualization of target runs for each match.
- **Result Distribution**: Distribution of match results (e.g., "runs", "wickets").
- **Match Margin Analysis**: Distribution of result margins in matches won by runs and wickets.
- **Toss Decision Analysis**: Insights into how toss decisions affect match outcomes.
- **Super Over Analysis**: Number of super overs in each season.
- **Top Umpires**: Statistics on the top umpires based on match counts.
- **Team Performance Analysis**: Insights into match wins by teams, including wins when chasing or batting first.

## Streamlit Deployment

This project is deployed using **Streamlit**, allowing you to interact with the IPL data through a web interface. The following steps guide you on how to run the Streamlit app locally:

### Prerequisites

To run the Streamlit app locally, you need to have Python 3.x installed, along with the following dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- streamlit

```bash
pip install pandas numpy matplotlib seaborn streamlit
