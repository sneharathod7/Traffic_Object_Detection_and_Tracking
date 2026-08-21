import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.safety.unsafe_overtaking_rule import detect_unsafe_overtaking

if __name__ == "__main__":
    csv_path = r"D:\btp\narain_data\long1_tracks_narain_cleaned_edited.csv"
    print("START", csv_path)
    out = detect_unsafe_overtaking(csv_path)
    print("OUT", out)
    print("DONE")
