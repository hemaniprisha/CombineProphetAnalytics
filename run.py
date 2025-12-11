import argparse
import subprocess
import sys
import os

def run_analysis():
    print("\nRunning combine_analysis.py...\n")
    subprocess.run([sys.executable, "combine_analysis.py"], check=True)

def run_app():
    print("\nLaunching app.py...\n")
    # If it's Streamlit:
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except FileNotFoundError:
        # If not Streamlit, fallback to normal Python execution
        subprocess.run([sys.executable, "app.py"], check=True)

def main():
    parser = argparse.ArgumentParser(description="Run CombineProphetAnalytics components.")
    parser.add_argument("--all", action="store_true", help="Run analysis and then launch the app.")
    parser.add_argument("--analysis", action="store_true", help="Only run combine_analysis.py.")
    parser.add_argument("--app", action="store_true", help="Only launch app.py.")

    args = parser.parse_args()

    if args.all:
        run_analysis()
        run_app()
    elif args.analysis:
        run_analysis()
    elif args.app:
        run_app()
    else:
        print("No arguments provided. Use --all, --analysis, or --app.")

if __name__ == "__main__":
    main()
