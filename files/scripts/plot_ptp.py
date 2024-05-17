import os
import argparse
import matplotlib.pyplot as plt
import textwrap
import numpy as np
import re

def extract_offset(ptp_log):
    offsets = []
    pattern = r'offset\s+(-?\d+)\s+s2'
    with open(f"{ptp_log}", "r", encoding="utf-8") as log_file:
        ptp_content = log_file.read().splitlines()
    for line in ptp_content:
        match =  re.search(pattern, line)
        if match:
            offsets.append(int(match.group(1)))

    return offsets

def plot_offset(offsets, output, file_name):
    plt.plot(range(len(offsets)), offsets)
    plt.xlabel("Time (s)")
    plt.ylabel('Offset (ns)')
    plt.axhline(y=np.nanmean(offsets), color='red', linestyle='--', linewidth=3, label='Average')
    plt.legend()
    plt.title('File: {}'.format(file_name))
    plt.savefig(f"{output}/plot_{file_name}.png")
    print(f"Plot saved as 'plot_{file_name}.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PTP offset plots")
    parser.add_argument("--input", "-i", type=str, required=True, help="ptp4l or phc2sys output file")
    parser.add_argument("--output", "-o", default="../results/", type=str, help="Output directory for the generated files.")


    args = parser.parse_args()
    file_name = args.input.split("/")[-1]
    offsets = extract_offset(args.input)
    plot_offset(offsets, args.output, file_name)
