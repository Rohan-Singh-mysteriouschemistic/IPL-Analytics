import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64

from app.analysis import (
    get_teams, team_vs_team, team_record, get_batsmen, get_bowlers,
    batsman_record_api, bowler_record_api, get_seasons, get_season_stats,
    compare_players_batting, compare_players_bowling
)
import json

# ---------------------- CONFIG ------------------------
st.set_page_config(page_title="IPL Dashboard", layout="wide")

team_logos = {
    "Mumbai Indians": "logos/MI.jpg",
    "Chennai Super Kings": "logos/CSK.jpg",
    "Royal Challengers Bangalore": "logos/RCB.jpg",
    "Kolkata Knight Riders": "logos/KKR.jpg",
    "Delhi Capitals": "logos/DC.jpg",
    "Sunrisers Hyderabad": "logos/SRH.jpg",
    "Rajasthan Royals": "logos/RR.jpg",
    "Gujarat Titans": "logos/GT.jpg",
    "Lucknow Super Giants": "logos/LSG.jpg",
    "Deccan Chargers": "logos/dchargers.png",
    "Delhi Daredevils": "logos/dd.webp",
    "Gujarat Lions": "logos/gl.png",
    "Kings XI Punjab": "logos/kxip.jpg",
    "Kochi Tuskers Kerala": "logos/ktk.png",
    "Pune Warriors": "logos/pw.png",
    "Rising Pune Supergiants": "logos/rps.webp",
    "Punjab Kings": "logos/PBKS.jpg"
}

# ---------------------- UTILS ------------------------
def get_logo_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def show_team_logo(name, path, wins):
    b64 = get_logo_base64(path)
    st.markdown(f"""
        <div style='text-align:center'>
            <h4>{name}</h4>
            <img src='data:image/png;base64,{b64}' style='height:120px;width:120px;object-fit:contain;border-radius:10px;box-shadow:0 0 10px rgba(0,0,0,0.1);padding:8px;background:white;'/>
            <div style='margin-top:8px;font-size:16px;'>🏆 <strong>Wins:</strong> {wins}</div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------- LOADED DATA ------------------------
teams = json.loads(get_teams())["teams"]
batsmen = json.loads(get_batsmen())["batsmen"]
bowlers = json.loads(get_bowlers())["bowlers"]
seasons = json.loads(get_seasons())["seasons"]

# ---------------------- SIDEBAR ------------------------
st.sidebar.image("logos/ipl.jpeg", use_container_width=True)
st.sidebar.title("🏏 IPL Dashboard")
st.sidebar.caption("2008 - 2022 • Stats & Visuals")
option = st.sidebar.selectbox("📂 Select Section", [
    "Home", "Season Overview", "Team Record", "Team vs Team",
    "Batsman Stats", "Bowler Stats", "Player Comparison"
])

# ---------------------- HOME ------------------------
if option == "Home":
    st.markdown("<h1 style='text-align:center; color:#ff4b4b;'>🏏 IPL Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:gray;'>Explore every run, wicket, rivalry, and title from 2008 to 2022</h4>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-style:italic;'>\"Behind every six, wicket, and victory lies a stat that tells the story.\"</p>", unsafe_allow_html=True)

    st.markdown("## 🚀 Welcome to the Ultimate IPL Stats Platform")
    st.write("""
    Dive deep into the numbers that shaped the Indian Premier League. Whether you're a data enthusiast, a cricket fan, or a curious analyst —
    this dashboard gives you the tools to explore IPL's rich history like never before.
    """)

    st.markdown("## 🔍 Key Features")
    st.markdown("""
    - 📅 **Season Overview:** Points tables, Orange & Purple Cap winners, total sixes & fours  
    - ⚔️ **Team vs Team Comparison:** Head-to-head stats, win ratios, and match summaries  
    - 📘 **Team Records:** Match count, wins, losses, no-results, and titles won  
    - 🧒 **Batsman Stats:** Runs, average, strike rate, boundaries, fifties, centuries, and highest scores  
    - 🎯 **Bowler Stats:** Wickets, economy, average, best figures, and strike rate  
    - 👥 **Player Comparison:** Side-by-side comparison of any two players (Batting or Bowling), including season-wise analysis  
    """)

    st.markdown("## 📌 How to Use")
    st.write("""
    - Use the sidebar to navigate different sections  
    - Select players or teams from dropdowns for detailed analysis  
    - Hover over plots and tables to explore insights interactively  
    """)

    st.markdown("---")
    st.markdown("<p style='text-align:center;'>Made with ❤️ for IPL fans and data explorers</p>", unsafe_allow_html=True)


# ---------------------- TEAM VS TEAM ------------------------
elif option == "Team vs Team":
    st.markdown("<h2 style='text-align:center;'>⚔️ Team vs Team Comparison</h2>", unsafe_allow_html=True)
    team1 = st.selectbox("🔹 Select Team 1", teams)
    team2 = st.selectbox("🔸 Select Team 2", teams, index=1)

    if team1 != team2:
        data = json.loads(team_vs_team(team1, team2))

        col1, col2 = st.columns(2)
        with col1:
            show_team_logo(team1, team_logos.get(team1, ""), data.get(team1, 0))
        with col2:
            show_team_logo(team2, team_logos.get(team2, ""), data.get(team2, 0))

        st.markdown("### 🧾 Match Summary")
        st.metric("🎮 Matches Played", data.get("total"))
        st.metric("➖ No Results", data.get("draws"))

        fig = px.pie(
            names=[team1, team2, "Draws"],
            values=[data.get(team1, 0), data.get(team2, 0), data.get("draws", 0)],
            title="Win Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- TEAM RECORD ------------------------
elif option == "Team Record":
    st.markdown("<h2 style='text-align:center;'>📘 Team Performance Summary</h2>", unsafe_allow_html=True)
    team = st.selectbox("Select a Team", teams)
    record = json.loads(team_record(team))[team]['overall']

    st.image(team_logos.get(team, ""), width=120)
    st.markdown("### 📊 Match Stats")
    cols = st.columns(5)
    cols[0].metric("📅 Matches", record['matches'])
    cols[1].metric("✅ Wins", record['won'])
    cols[2].metric("❌ Losses", record['loss'])
    cols[3].metric("➖ No Results", record['noResult'])
    cols[4].metric("🏆 Titles", record['titles'])

    fig = px.bar(
        x=["Wins", "Losses", "No Result"],
        y=[record['won'], record['loss'], record['noResult']],
        labels={'x': 'Result', 'y': 'Matches'},
        title="Match Result Breakdown",
        color_discrete_sequence=["green", "red", "gray"]
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- BATSMAN STATS ------------------------
elif option == "Batsman Stats":
    batsman = st.selectbox("Select Batsman", batsmen)
    record = json.loads(batsman_record_api(batsman))[batsman]['all']

    st.markdown(f"<h3 style='text-align:center;'>🏏 {batsman}</h3>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Innings", record['innings'])
    cols[1].metric("Runs", record['runs'])
    cols[2].metric("Avg", f"{record['average']:.2f}")
    cols[3].metric("SR", f"{record['strikeRate']:.2f}")
    st.metric("HS", record['highestScore'])
    st.metric("MoM", record['mom'])

    df = pd.DataFrame({
        "Metric": ["4s", "6s", "50s", "100s"],
        "Count": [record['fours'], record['sixes'], record['50s'], record['100s']]
    })
    fig = px.bar(df, x="Metric", y="Count", title="Boundary & Milestone Stats", color="Metric")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- BOWLER STATS ------------------------
elif option == "Bowler Stats":
    bowler = st.selectbox("Select Bowler", bowlers)
    record = json.loads(bowler_record_api(bowler))[bowler]['all']

    st.markdown(f"<h3 style='text-align:center;'>🎳 {bowler}</h3>", unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Wickets", record['wicket'])
    cols[1].metric("Best", record['best_figure'])
    cols[2].metric("Innings", record['innings'])

    cols = st.columns(3)
    cols[0].metric("Economy", f"{record['economy']:.2f}")
    cols[1].metric("Avg", f"{record['average']:.2f}")
    cols[2].metric("SR", f"{record['strikeRate']:.2f}")
    st.metric("MoM", record['mom'])

    df = pd.DataFrame({
        "Type": ["4s Given", "6s Given", "3+ Wicket Hauls"],
        "Count": [record['fours'], record['sixes'], record['3+W']]
    })
    fig = px.bar(df, x="Type", y="Count", title="Bowling Highlights", color="Type")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- SEASON OVERVIEW ------------------------
elif option == "Season Overview":
    season = st.selectbox("Select a Season", seasons)
    stats = json.loads(get_season_stats(season))

    st.subheader("📊 Points Table")
    df = pd.DataFrame(stats["pointsTable"], columns=["Team", "Points"])
    st.dataframe(df, use_container_width=True)

    st.subheader("🏆 Orange Cap")
    st.markdown(f"**{stats['orangeCap']['batter']}** scored **{stats['orangeCap']['batsman_run']}** runs.")

    st.subheader("🎯 Purple Cap")
    st.markdown(f"**{stats['purpleCap']['bowler']}** took **{stats['purpleCap']['isBowlerWicket']}** wickets.")

    col1, col2 = st.columns(2)
    col1.metric("💥 Total Sixes", stats["totalSixes"])
    col2.metric("🏏 Total Fours", stats["totalFours"])

# ---------------------- PLAYER COMPARISON ------------------------
elif option == "Player Comparison":
    view_mode = st.radio("View Mode", ["Batting", "Bowling"])
    season_filter = st.checkbox("Compare Season-wise")

    p1 = st.selectbox("Select Player 1", batsmen)
    p2 = st.selectbox("Select Player 2", batsmen, index=1)

    if p1 != p2:
        if view_mode == "Batting":
            result = compare_players_batting(p1, p2, season_wise=season_filter)
        else:
            result = compare_players_bowling(p1, p2, season_wise=season_filter)

        d1 = result[p1]
        d2 = result[p2]

        st.subheader(f"📈 {view_mode} Stats Comparison")
        if not season_filter:
            col1, col2 = st.columns(2)
            for c, player, data in zip([col1, col2], [p1, p2], [d1, d2]):
                c = col1 if player == p1 else col2
                c.markdown(f"### {player}")
                for k, v in data.items():
                    if isinstance(v, dict):
                        c.metric(str(k).capitalize(), v.get('average', 0))
                    else:
                        c.metric(str(k).capitalize(), v)


        if season_filter:
            seasons = sorted(set(d1.keys()).intersection(set(d2.keys())))

            if view_mode == "Bowling":
                # We'll use 'wicket' as default for season-wise bowling comparison
                df_season = pd.DataFrame({
                    "Season": seasons,
                    p1: [d1[s].get("wicket", 0) for s in seasons],
                    p2: [d2[s].get("wicket", 0) for s in seasons]
                })

                st.dataframe(df_season.set_index("Season"))

                df_melted = df_season.melt(id_vars="Season", var_name="Player", value_name="Wickets")
                fig = px.line(df_melted, x="Season", y="Wickets", color="Player",
                            markers=True, title="Season-wise Wicket Comparison")
                st.plotly_chart(fig, use_container_width=True)

            else:
                # Batting: use average
                df_season = pd.DataFrame({
                    "Season": seasons,
                    p1: [d1[s].get("average", 0) for s in seasons],
                    p2: [d2[s].get("average", 0) for s in seasons]
                })

                st.dataframe(df_season.set_index("Season"))

                df_melted = df_season.melt(id_vars="Season", var_name="Player", value_name="Average")
                fig = px.line(df_melted, x="Season", y="Average", color="Player",
                            markers=True, title="Season-wise Batting Average")
                st.plotly_chart(fig, use_container_width=True)



        else:
            # Dot plot comparison
            metric_keys = [k for k in d1.keys() if not isinstance(d1[k], dict)]
            df_dot = pd.DataFrame({
                "Metric": metric_keys * 2,
                "Value": [d1[m] for m in metric_keys] + [d2[m] for m in metric_keys],
                "Player": [p1]*len(metric_keys) + [p2]*len(metric_keys)
            })

            fig = px.strip(df_dot, x="Metric", y="Value", color="Player", title="Dot Plot: Stat Comparison")
            st.plotly_chart(fig, use_container_width=True)

            # Bar Graphs for each metric
            for metric in metric_keys:
                fig = px.bar(pd.DataFrame({
                    "Player": [p1, p2],
                    "Value": [d1[metric], d2[metric]]
                }), x="Player", y="Value", title=f"{str(metric).capitalize()} Comparison", color="Player", text="Value")
                st.plotly_chart(fig, use_container_width=True)
