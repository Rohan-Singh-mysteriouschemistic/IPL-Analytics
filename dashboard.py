import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64

# ---------------------- CONFIG ------------------------
st.set_page_config(page_title="IPL Dashboard", layout="wide")
API_BASE = "https://ipl-2008-2022-api.onrender.com/api"

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

# ---------------------- API DATA ------------------------
@st.cache_data
def get_teams():
    return requests.get(f"{API_BASE}/teams").json()["teams"]

@st.cache_data
def get_batsmen():
    return requests.get(f"{API_BASE}/batsmen").json()["batsmen"]

@st.cache_data
def get_bowlers():
    return requests.get(f"{API_BASE}/bowlers").json()["bowlers"]

@st.cache_data
def get_season_options():
    return requests.get(f"{API_BASE}/seasons").json()["seasons"]

teams = get_teams()

# ---------------------- SIDEBAR ------------------------
st.sidebar.image("logos/ipl.jpeg", use_column_width=True)
st.sidebar.title("🏏 IPL Dashboard")
st.sidebar.caption("2008 - 2022 • Stats & Visuals")
option = st.sidebar.selectbox("📂 Select Section", [
    "Home", "Season Overview", "Team Record", "Team vs Team", "Batsman Stats", "Bowler Stats", "Player Comparison"
])

# ---------------------- HOME ------------------------
if option == "Home":
    st.markdown("<h1 style='text-align:center;'>🏏 IPL Analytics Dashboard (2008–2022)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-style:italic;'>\"Behind every six, wicket, and victory lies a stat that tells the story.\"</p>", unsafe_allow_html=True)

    st.markdown("### 🌟 Why this Dashboard?")
    st.write("This platform helps you uncover **patterns**, **rivalries**, and **standout performances** across IPL history.")

    st.markdown("### 🔍 Key Features")
    st.markdown("""
    - 📊 **Team Comparison:** Head-to-head records  
    - 📘 **Team Performance:** Match stats and title wins  
    - 🧢 **Batting Records:** Runs, SR, milestones  
    - 🎯 **Bowling Records:** Economy, wickets  
    - 📅 **Season Insights:** Points tables & caps  
    """)

# ---------------------- TEAM VS TEAM ------------------------
elif option == "Team vs Team":
    st.markdown("<h2 style='text-align:center;'>⚔️ Team vs Team Comparison</h2>", unsafe_allow_html=True)
    team1 = st.selectbox("🔹 Select Team 1", teams)
    team2 = st.selectbox("🔸 Select Team 2", teams, index=1)

    if team1 != team2:
        data = requests.get(f"{API_BASE}/team-vs-team?team1={team1}&team2={team2}").json()

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
    record = requests.get(f"{API_BASE}/team-record?team={team}").json()[team]['overall']

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
    st.markdown("<h2 style='text-align:center;'>🧢 Batsman Insights</h2>", unsafe_allow_html=True)
    batsman = st.selectbox("Select Batsman", get_batsmen())

    try:
        record = requests.get(f"{API_BASE}/batsman?name={batsman}").json()[batsman]['all']
        st.markdown(f"<h3 style='text-align:center;'>🏏 {batsman}</h3>", unsafe_allow_html=True)

        cols = st.columns(4)
        cols[0].metric("🧮 Innings", record['innings'])
        cols[1].metric("🪙 Runs", record['runs'])
        cols[2].metric("📈 Average", f"{record['avg']:.2f}")
        cols[3].metric("🚀 Strike Rate", f"{record['strikeRate']:.2f}")

        st.metric("🏅 Highest Score", record['highestScore'])
        st.metric("🎖️ MoM Awards", record['mom'])

        df = pd.DataFrame({
            "Metric": ["4s", "6s", "50s", "100s"],
            "Count": [record['fours'], record['sixes'], record['fifties'], record['hundreds']]
        })

        fig = px.bar(df, x="Metric", y="Count", title="Boundary & Milestone Stats", color="Metric")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("⚠️ Batsman data unavailable.")

# ---------------------- BOWLER STATS ------------------------
elif option == "Bowler Stats":
    st.markdown("<h2 style='text-align:center;'>🎯 Bowler Analytics</h2>", unsafe_allow_html=True)
    bowler = st.selectbox("Select Bowler", get_bowlers())

    try:
        record = requests.get(f"{API_BASE}/bowler?name={bowler}").json()[bowler]['all']
        st.markdown(f"<h3 style='text-align:center;'>🎳 {bowler}</h3>", unsafe_allow_html=True)

        cols = st.columns(3)
        cols[0].metric("🎯 Wickets", record['wicket'])
        cols[1].metric("🔥 Best", record['best_figure'])
        cols[2].metric("🧮 Innings", record['innings'])

        cols = st.columns(3)
        cols[0].metric("⏱️ Economy", f"{record['economy']:.2f}")
        cols[1].metric("📉 Avg", f"{record['average']:.2f}")
        cols[2].metric("⚡ SR", f"{record['strikeRate']:.2f}")

        st.metric("🎖️ MoM Awards", record['mom'])

        df = pd.DataFrame({
            "Type": ["4s Given", "6s Given", "3+ Wicket Hauls"],
            "Count": [record['fours'], record['sixes'], record['3+W']]
        })

        fig = px.bar(df, x="Type", y="Count", title="Bowling Highlights", color="Type")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("⚠️ Bowler data unavailable.")

# ---------------------- SEASON OVERVIEW ------------------------
elif option == "Season Overview":
    st.markdown("<h2 style='text-align:center;'>📅 Season Overview</h2>", unsafe_allow_html=True)
    season = st.selectbox("Select a Season", get_season_options())

    try:
        stats = requests.get(f"{API_BASE}/season-stats?season={season}").json()

        st.subheader("📊 Points Table")
        df = pd.DataFrame(stats["pointsTable"])
        df.index = df.index + 1
        df.rename(columns={0: "Teams", 1: "Points"}, inplace=True)
        st.dataframe(df, use_container_width=True)

        st.subheader("🏆 Orange Cap")
        oc = stats["orangeCap"]
        st.markdown(f"**{oc['batter']}** scored **{oc['batsman_run']}** runs.")

        st.subheader("🎯 Purple Cap")
        pc = stats["purpleCap"]
        st.markdown(f"**{pc['bowler']}** took **{pc['isBowlerWicket']}** wickets.")

        col1, col2 = st.columns(2)
        col1.metric("💥 Total Sixes", stats["totalSixes"])
        col2.metric("🏏 Total Fours", stats["totalFours"])

    except:
        st.error("❌ Failed to load data for the selected season.")

# ---------------------- SECTION: PLAYER COMPARISON ------------------------
elif option == "Player Comparison":
    st.markdown("<h2 style='text-align: center;'>👥 Player Comparison</h2>", unsafe_allow_html=True)
    player1 = st.selectbox("Select Player 1", get_batsmen())
    player2 = st.selectbox("Select Player 2", get_batsmen(), index=1)

    if player1 != player2:
        try:
            comparison = requests.get(
                f"{API_BASE}/player-comparison?player1={player1}&player2={player2}"
            ).json()

            p1 = comparison.get(player1, {})
            p2 = comparison.get(player2, {})

            if p1 and p2:
                st.subheader("🔢 Key Metrics")
                cols = st.columns(2)
                for idx, (player, stats) in enumerate([(player1, p1), (player2, p2)]):
                    with cols[idx]:
                        st.markdown(f"### {player}")
                        st.metric("Runs", stats['runs'])
                        st.metric("Balls", stats['balls'])
                        st.metric("Average", f"{stats['average']:.2f}")
                        st.metric("Strike Rate", f"{stats['strike_rate']:.2f}")
                        st.metric("4s", stats['4s'])
                        st.metric("6s", stats['6s'])

                st.markdown("---")
                st.subheader("📈 Comparative Graphs")

                df_compare = pd.DataFrame({
                    "Player": [player1, player2],
                    "Average": [p1['average'], p2['average']],
                    "Strike Rate": [p1['strike_rate'], p2['strike_rate']],
                    "4s": [p1['4s'], p2['4s']],
                    "6s": [p1['6s'], p2['6s']]
                })

                p1_values = [
                    p1['average'],
                    p1['strike_rate'],
                    p1['4s'],
                    p1['6s'],
                    p1.get('50s', 0),    # use .get to avoid KeyError if not present
                    p1.get('100s', 0)
                ]

                p2_values = [
                    p2['average'],
                    p2['strike_rate'],
                    p2['4s'],
                    p2['6s'],
                    p2.get('50s', 0),
                    p2.get('100s', 0)
                ]
                df_dot = pd.DataFrame({
                    "Metric": ["Average", "Strike Rate", "4s", "6s", "50s", "100s"] * 2,
                    "Value": p1_values + p2_values,
                    "Player": [player1]*6 + [player2]*6
                })

                fig = px.strip(df_dot, x="Metric", y="Value", color="Player", title="Dot Plot: Stat Comparison", stripmode='overlay')
                st.plotly_chart(fig, use_container_width=True)

                fig1 = px.bar(df_compare, x="Player", y="Average", title="Batting Average Comparison", color="Player", text_auto=True)
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = px.bar(df_compare, x="Player", y="Strike Rate", title="Strike Rate Comparison", color="Player", text_auto=True)
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.bar(df_compare.melt(id_vars="Player", value_vars=["4s", "6s"]),
                              x="Player", y="value", color="variable",
                              barmode="group", text_auto=True,
                              title="Boundary Count Comparison (4s vs 6s)")
                st.plotly_chart(fig3, use_container_width=True)

            else:
                st.error("⚠️ One or both player records could not be found in the response.")
        except:
            st.error("❌ Failed to fetch player comparison data.")
