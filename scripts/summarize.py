"""Build results/summary.md and results/quality_vs_speed.png from results/*.json"""
from llmseries.report import summary

if __name__ == "__main__":
    summary()
