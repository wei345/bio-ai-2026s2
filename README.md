# Badminton Smash Kinematic Analysis and Injury Risk Assessment

A prototype web application for extracting kinematic metrics from badminton smash videos and assessing potential shoulder injury-risk patterns using AI-based video analysis.

## Requirements

- Python 3.11

## Quick Start

1. Clone or download this repository.
2. Open a command-line terminal and enter the project directory.
3. Create a virtual environment and install the required dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

   pip install -e .
   ````

   The virtual environment only needs to be created once. When returning to the project directory later, simply activate the existing environment:

   ```bash
   source .venv/bin/activate
   ```

4. Start the web UI:

   ```bash
   python src/app.py
   ```

   If the application fails to start because the port is already in use, resolve the port conflict before continuing.

5. Open the web UI in a browser:

   <http://localhost:5001>

6. Upload one or more smash videos.

7. Click **Analyze Smashes**.

8. Click **Generate Assessment**.

9. Review the results, video, and kinematic plots in the web UI.
