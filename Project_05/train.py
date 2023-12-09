import subprocess
import random

def generate_random_seed():
    return "RANDOM" + str(random.randint(1, 10000))

def run_capture_script(team_one, team_two, layout_argument):

    command = [
        "py",
        "-3.11",
        "capture.py",
        "-r", team_one,
        "-b", team_two,
        "-l", layout_argument
    ]

    subprocess.run(command)

# Example usage:
team_one = "secondTeam"
team_two_options = ['baselineTeam', 'testTeam1', 'testTeam2', 'testTeam3']

layout_options = [
    'alleyCapture.lay',
    'bloxCapture.lay',
    'crowdedCapture.lay',
    'defaultCapture.lay',
    'fastCapture.lay',
    'jumboCapture.lay',
    'mediumCapture.lay',
    'officeCapture.lay',
    'strategicCapture.lay'
]

# Run the command 20000 times
for _ in range(10000):
    team_two = random.choice(team_two_options)
    layout_argument = random.choice(layout_options)
    run_capture_script(team_one, team_two, layout_argument)

for _ in range(10000):
    team_two = random.choice(team_two_options)
    layout_argument = generate_random_seed()
    run_capture_script(team_one, team_two, layout_argument)
