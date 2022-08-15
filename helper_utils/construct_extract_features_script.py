import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_lengths", default="4")
    parser.add_argument("--home_path", default=os.getcwd())
    args = parser.parse_args()