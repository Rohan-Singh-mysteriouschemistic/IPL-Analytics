import pandas as pd
import numpy as np
import json
from app.data_loader import load_data
from app.utils import NpEncoder

ipl, balls = load_data()

# Merging and computing BowlingTeam
data = balls.merge(ipl[['ID', 'Season', 'Team1', 'Team2', 'WinningTeam', 'Player_of_Match']], on='ID', how='left')
data['BowlingTeam'] = data.apply(lambda row: row['Team1'] if row['BattingTeam'] == row['Team2'] else row['Team2'], axis=1)

# After merging
required_cols = balls.columns.tolist() + ['BowlingTeam', 'Player_of_Match', 'Season']
batter_data = data[[col for col in required_cols if col in data.columns]]

bowler_data = batter_data.copy()
bowler_data['bowler_run'] = bowler_data.apply(
    lambda row: 0 if row['extra_type'] in ['penalty', 'legbyes', 'byes'] else row['total_run'], axis=1
)

# Compute bowler wicket flag
valid_dismissals = ['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket']
bowler_data['isBowlerWicket'] = bowler_data.apply(
    lambda row: row['isWicketDelivery'] if row['kind'] in valid_dismissals else 0, axis=1
)

def get_teams():
    teams = sorted(set(ipl['Team1']).union(ipl['Team2']))
    return json.dumps({"teams": teams}, cls=NpEncoder)

def get_batsmen():
    names = sorted(batter_data['batter'].unique())
    return json.dumps({"batsmen": names}, cls=NpEncoder)

def get_bowlers():
    names = sorted(bowler_data['bowler'].unique())
    return json.dumps({"bowlers": names}, cls=NpEncoder)

def get_seasons():
    seasons = sorted(ipl['Season'].unique())
    return json.dumps({"seasons": seasons}, cls=NpEncoder)

def team_vs_team(t1, t2):
    df = ipl[((ipl.Team1 == t1) & (ipl.Team2 == t2)) | ((ipl.Team1 == t2) & (ipl.Team2 == t1))]
    total = df.shape[0]
    wins = df['WinningTeam'].value_counts().to_dict()
    return json.dumps({'total': total, t1: wins.get(t1, 0), t2: wins.get(t2, 0), 'draws': total - wins.get(t1, 0) - wins.get(t2, 0)}, cls=NpEncoder)

def team_record(team):
    df = ipl[(ipl.Team1 == team) | (ipl.Team2 == team)]
    total = df.shape[0]
    won = df[df.WinningTeam == team].shape[0]
    nr = df['WinningTeam'].isnull().sum()
    loss = total - won - nr
    titles = df[(df.MatchNumber == 'Final') & (df.WinningTeam == team)].shape[0]
    vs = {opp: json.loads(team_vs_team(team, opp)) for opp in ipl.Team1.unique()}
    return json.dumps({team: {'overall': {'matches': total, 'won': won, 'loss': loss, 'noResult': nr, 'titles': titles}, 'against': vs}}, cls=NpEncoder)

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
    return {'innings': inngs, 'runs': runs, 'fours': fours, 'sixes': sixes, 'average': avg, 'strikeRate': sr, '50s': fifties, '100s': hundreds, 'highestScore': hs, 'notOuts': notouts, 'mom': moms}

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
    return {'innings': inngs, 'wicket': wkts, 'economy': eco, 'average': avg, 'strikeRate': sr, 'fours': fours, 'sixes': sixes, 'best_figure': best_fig, '3+W': w3, 'mom': moms}

def batsman_record_api(name):
    df = batter_data[batter_data.innings.isin([1, 2])]
    record = batsman_stats(name, df)
    vs = {t: batsman_stats(name, df[df.BowlingTeam == t]) for t in ipl.Team1.unique()}
    return json.dumps({name: {'all': record, 'against': vs}}, cls=NpEncoder)

def bowler_record_api(name):
    df = bowler_data[bowler_data.innings.isin([1, 2])]
    record = bowler_stats(name, df)
    vs = {t: bowler_stats(name, df[df.BattingTeam == t]) for t in ipl.Team1.unique()}
    return json.dumps({name: {'all': record, 'against': vs}}, cls=NpEncoder)

def compare_players_batting(p1, p2, season_wise=False):
    df = batter_data[batter_data.innings.isin([1, 2])]
    if season_wise:
        seasons = sorted(df['Season'].unique())
        return {
            p1: {s: batsman_stats(p1, df[df['Season'] == s]) for s in seasons},
            p2: {s: batsman_stats(p2, df[df['Season'] == s]) for s in seasons},
        }
    return {
        p1: batsman_stats(p1, df),
        p2: batsman_stats(p2, df),
    }

def compare_players_bowling(p1, p2, season_wise=False):
    df = bowler_data[bowler_data.innings.isin([1, 2])].copy()

    # Ensure required columns exist
    if 'Season' not in df.columns or 'bowler_run' not in df.columns:
        raise KeyError("Required columns missing in bowler_data")

    if season_wise:
        seasons = sorted(df['Season'].dropna().unique())
        return {
            p1: {s: bowler_stats(p1, df[df['Season'] == s]) for s in seasons},
            p2: {s: bowler_stats(p2, df[df['Season'] == s]) for s in seasons},
        }
    return {
        p1: bowler_stats(p1, df),
        p2: bowler_stats(p2, df),
    }


def get_season_stats(season):
    try:
        season = int(season)
    except:
        return json.dumps({'error': 'Invalid season format'}, cls=NpEncoder)

    season_matches = ipl[ipl['Season'] == season]
    if season_matches.empty:
        return json.dumps({'error': 'No data for this season'}, cls=NpEncoder)

    points = {}
    for _, row in season_matches.iterrows():
        team1, team2, winner = row['Team1'], row['Team2'], row['WinningTeam']
        points[team1] = points.get(team1, 0)
        points[team2] = points.get(team2, 0)
        if pd.notna(winner):
            points[winner] += 2
        else:
            points[team1] += 1
            points[team2] += 1
    points_table = sorted(points.items(), key=lambda x: -x[1])

    match_ids = season_matches['ID'].unique()
    season_balls = balls[balls['ID'].isin(match_ids)]

    orange_cap = (
        season_balls.groupby('batter', as_index=False)['batsman_run']
        .sum()
        .sort_values('batsman_run', ascending=False)
        .iloc[0]
        .to_dict()
    )

    valid_kinds = ['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket']
    season_balls['isBowlerWicket'] = (
        (season_balls['isWicketDelivery'] == 1) & season_balls['kind'].isin(valid_kinds)
    ).astype(int)
    purple_cap = (
        season_balls.groupby('bowler', as_index=False)['isBowlerWicket']
        .sum()
        .sort_values('isBowlerWicket', ascending=False)
        .iloc[0]
        .to_dict()
    )

    boundary_balls = season_balls[season_balls['non_boundary'] == 0]
    total_sixes = (boundary_balls['batsman_run'] == 6).sum()
    total_fours = (boundary_balls['batsman_run'] == 4).sum()

    return json.dumps({
        'pointsTable': points_table,
        'orangeCap': orange_cap,
        'purpleCap': purple_cap,
        'totalSixes': int(total_sixes),
        'totalFours': int(total_fours)
    }, cls=NpEncoder)
