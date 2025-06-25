from flask import Flask, request
import analysis

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/api/teams')
def get_teams():
    return analysis.get_teams()

@app.route('/api/team-vs-team')
def team_vs_team():
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')

    return analysis.team_vs_team(team1, team2)

@app.route('/api/team-record')
def team_record():
    team = request.args.get('team')

    return analysis.team_record(team)

@app.route('/api/batsman')
def batsman():
    name = request.args.get('name')

    return analysis.batsman_record_api(name)

@app.route('/api/bowler')
def bowler():
    name = request.args.get('name')
    
    return analysis.bowler_record_api(name)

app.run(debug=True)
