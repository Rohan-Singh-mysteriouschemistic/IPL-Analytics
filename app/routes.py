from flask import request, jsonify
from app.analysis import *

def register_routes(app):
    @app.route('/')
    def home():
        return "Hello World"

    @app.route('/api/teams')
    def get_teams_route():
        return get_teams()

    @app.route('/api/team-vs-team')
    def team_vs_team_route():
        team1 = request.args.get('team1')
        team2 = request.args.get('team2')
        return team_vs_team(team1, team2)

    @app.route('/api/team-record')
    def team_record_route():
        team = request.args.get('team')
        return team_record(team)

    @app.route('/api/batsman')
    def batsman_route():
        name = request.args.get('name')
        return batsman_record_api(name)

    @app.route('/api/bowler')
    def bowler_route():
        name = request.args.get('name')
        return bowler_record_api(name)
    
    @app.route('/api/batsmen')
    def get_batsmen_route():
        return get_batsmen()

    @app.route('/api/bowlers')
    def get_bowlers_route():
        return get_bowlers()
    
    @app.route('/api/season-stats')
    def season_stats():
        season = request.args.get('season')
        return get_season_stats(season)

    @app.route('/api/seasons')
    def get_season_list():
        return get_seasons()
    
    @app.route('/api/player-comparison', methods=['GET'])
    def player_comparison():
        player1 = request.args.get("player1")
        player2 = request.args.get("player2")
        result = compare_players(player1, player2)
        return jsonify(result)

