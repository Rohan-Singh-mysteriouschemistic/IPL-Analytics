import pandas as pd
import numpy as np
import json

# Custom JSON encoder to handle NumPy data types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return super().default(obj)

# Reading datasets
ipl = pd.read_csv("IPL_2008_2022.csv")
balls = pd.read_csv("IPL_Balls.csv")

# Merging and computing BowlingTeam
# If BattingTeam is Team2, then BowlingTeam is Team1; else it's Team2
data = balls.merge(ipl, on='ID')
data['BowlingTeam'] = data.apply(lambda row: row['Team1'] if row['BattingTeam'] == row['Team2'] else row['Team2'], axis=1)

# Shared data for batting and bowling statistics
batter_data = data[balls.columns.tolist() + ['BowlingTeam', 'Player_of_Match']]
bowler_data = batter_data.copy()

# Calculating bowler runs excluding certain extras
bowler_data['bowler_run'] = bowler_data.apply(lambda row: 0 if row['extra_type'] in ['penalty', 'legbyes', 'byes'] else row['total_run'], axis=1)

# Declaring all the types of valid Dismissals
valid_dismissals = ['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket']
bowler_data['isBowlerWicket'] = bowler_data.apply(lambda row: row['isWicketDelivery'] if row['kind'] in valid_dismissals else 0, axis=1)

# Returns a JSON list of all unique IPL teams
def get_teams():
    teams = sorted(set(ipl['Team1']).union(ipl['Team2']))
    return json.dumps({"teams": teams}, cls=NpEncoder)

# Returns the head-to-head stats between two teams
def team_vs_team(t1, t2):
    if t1 not in ipl.values or t2 not in ipl.values:
        return json.dumps({'error': 'Invalid team name'}, cls=NpEncoder)

    df = ipl[((ipl.Team1 == t1) & (ipl.Team2 == t2)) | ((ipl.Team1 == t2) & (ipl.Team2 == t1))]
    total = df.shape[0]
    wins = df['WinningTeam'].value_counts().to_dict()

    return json.dumps({'total': total, t1: wins.get(t1, 0), t2: wins.get(t2, 0), 'draws': total - wins.get(t1, 0) - wins.get(t2, 0)}, cls=NpEncoder)

# Returns overall and vs-team stats for a team
def team_record(team):
    df = ipl[(ipl.Team1 == team) | (ipl.Team2 == team)]
    total = df.shape[0]

    # Wins
    won = df[df.WinningTeam == team].shape[0]
    nr = df['WinningTeam'].isnull().sum()

    # Losses
    loss = total - won - nr

    # Titles (finals won)
    titles = df[(df.MatchNumber == 'Final') & (df.WinningTeam == team)].shape[0]

    # Head-to-head stats with each opponent
    vs = {opp: json.loads(team_vs_team(team, opp)) for opp in ipl.Team1.unique()}

    return json.dumps({team: {'overall': {'matches': total, 'won': won, 'loss': loss, 'noResult': nr, 'titles': titles}, 'against': vs}}, cls=NpEncoder)

# Computes batting stats for a batsman
def batsman_stats(name, df):
    out = df[df.player_out == name].shape[0]
    df = df[df.batter == name]

    inngs = df['ID'].nunique()
    runs = df['batsman_run'].sum()

    fours = df[(df.batsman_run == 4) & (df.non_boundary == 0)].shape[0]
    sixes = df[(df.batsman_run == 6) & (df.non_boundary == 0)].shape[0]
    avg = runs / out if out else np.inf

    balls = df[~(df.extra_type == 'wides')].shape[0]
    sr = (runs / balls) * 100 if balls else 0

    match_scores = df.groupby('ID')['batsman_run'].sum()
    fifties = match_scores.between(50, 99).sum()
    hundreds = (match_scores >= 100).sum()

    top_id = match_scores.idxmax() if not match_scores.empty else None
    hs = f"{match_scores.max()}" if (df[df.ID == top_id].player_out == name).any() else f"{match_scores.max()}*"

    notouts = inngs - out
    moms = df[df.Player_of_Match == name]['ID'].nunique()

    return {'innings': inngs, 'runs': runs, 'fours': fours, 'sixes': sixes, 'avg': avg, 'strikeRate': sr, 'fifties': fifties, 'hundreds': hundreds,'highestScore': hs, 'notOut': notouts, 'mom': moms}

# Wrapper function to return JSON of batsman overall and vs-team stats
def batsman_record_api(name):
    df = batter_data[batter_data.innings.isin([1, 2])]
    record = batsman_stats(name, df)
    vs = {t: batsman_stats(name, df[df.BowlingTeam == t]) for t in ipl.Team1.unique()}

    return json.dumps({name: {'all': record, 'against': vs}}, cls=NpEncoder)

# Computes bowling stats for a bowler
def bowler_stats(name, df):
    df = df[df.bowler == name]
    inngs = df.ID.nunique()

    balls = df[~df.extra_type.isin(['wides', 'noballs'])].shape[0]
    runs = df['bowler_run'].sum()
    wkts = df['isBowlerWicket'].sum()

    eco = (runs / balls) * 6 if balls else 0
    avg = runs / wkts if wkts else np.inf
    sr = (balls / wkts) * 100 if wkts else np.nan

    fours = df[(df.batsman_run == 4) & (df.non_boundary == 0)].shape[0]
    sixes = df[(df.batsman_run == 6) & (df.non_boundary == 0)].shape[0]

    match_summary = df.groupby('ID').agg({'isBowlerWicket': 'sum', 'bowler_run': 'sum'})
    w3 = (match_summary['isBowlerWicket'] >= 3).sum()
    best = match_summary.sort_values(['isBowlerWicket', 'bowler_run'], ascending=[False, True]).head(1)

    best_fig = f"{int(best['isBowlerWicket'].values[0])}/{int(best['bowler_run'].values[0])}" if not best.empty else np.nan
    moms = df[df.Player_of_Match == name]['ID'].nunique()

    return { 'innings': inngs, 'wicket': wkts, 'economy': eco, 'average': avg, 'strikeRate': sr, 'fours': fours, 'sixes': sixes, 'best_figure': best_fig, '3+W': w3, 'mom': moms}

# Wrapper function to return JSON of bowler overall and vs-team stats
def bowler_record_api(name):
    df = bowler_data[bowler_data.innings.isin([1, 2])]
    record = bowler_stats(name, df)
    vs = {t: bowler_stats(name, df[df.BattingTeam == t]) for t in ipl.Team1.unique()}
    
    return json.dumps({name: {'all': record, 'against': vs}}, cls=NpEncoder)
