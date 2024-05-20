import os
import argparse
import matplotlib.pyplot as plt
import textwrap
import numpy as np

ADOC_FILE_PATH = f"latency-tests-report.adoc"

def compute_pacing(pub_sv, sub_sv):
    pub_pacing = []
    sub_pacing = []

    pub_timestamps = [int(item.split(":")[2]) for item in pub_sv]
    sub_timestamps = [int(item.split(":")[2]) for item in sub_sv]

    pub_pacing = np.diff(pub_timestamps)
    sub_pacing = np.diff(sub_timestamps)

    return pub_pacing, sub_pacing

def compute_latency(pub_sv, sub_sv, output):
    diff = []
    sv_drop = 0
    for index_sv in range(0, len(pub_sv)-1):

        pub_decomposed_sv = pub_sv[index_sv].split(":")
        pub_sv_id = pub_decomposed_sv[0]
        pub_sv_cnt = int(pub_decomposed_sv[1]) + sv_drop
        pub_sv_timestamp = pub_decomposed_sv[2]

        sub_decomposed_sv = sub_sv[index_sv].split(":")
        sub_sv_cnt = int(sub_decomposed_sv[1])
        sub_sv_timestamp = sub_decomposed_sv[2]

        if pub_sv_cnt != sub_sv_cnt:
            diff.append(f"SV DROPPED:{pub_sv_id}:{pub_sv_cnt}\n")
        else:
            latency = int(sub_sv_timestamp) - int(pub_sv_timestamp)
            diff.append(f"{pub_sv_id}:{pub_sv_cnt}:{latency}\n")

    with open(f"{output}/diff_results", "w", encoding="utf-8") as results_file:
        for line in diff:
            results_file.write(line)

    # Read the differences file
    filename = os.path.join(output, f"diff_results")
    stream_name = np.genfromtxt(filename, delimiter=":", usecols=[0], dtype=str)[0]
    latencies = np.genfromtxt(filename, delimiter=":", usecols=[2], dtype=int)
    return stream_name, latencies


def extract_sv(sv_file_path):
    with open(f"{sv_file_path}", "r", encoding="utf-8") as sv_file:
        sv_content = sv_file.read()
        sv = sv_content.split("\n")[:-1]
    return sv

def get_stream_count(output):
    filename = os.path.join(output, f"diff_results")
    data = np.genfromtxt(filename, delimiter=":", usecols=[0], dtype=str)
    return np.unique(data).size

def compute_min(values):
    return np.min(values) if values.size > 0 else None

def compute_max(values):
    return np.max(values) if values.size > 0 else None

def compute_average(values):
    return np.round(np.mean(values)) if values.size > 0 else None

def compute_neglat(values):
    return np.count_nonzero(values < 0)

def compute_size(values):
    return np.size(values)

def save_histogram(values, sub_name, output):
    # Plot latency histograms
    plt.hist(values, bins=20, alpha=0.7)

    # Add titles and legends
    plt.xlabel("Latency (us)")
    plt.ylabel("Frequency")
    plt.title(f"Latency Histogram for {sub_name}")

    # Save the plot
    if not os.path.exists(output):
        os.makedirs(output)
    filename = os.path.realpath(f"{output}/latency_histogram_{sub_name}.png")
    plt.savefig(filename)
    print(f"Histogram saved as 'latency_histogram_{sub_name}.png'.")
    plt.close()
    return filename

def plot_cdf(values, output):
    sorted_latency = np.sort(values)

    # Calculate the cumulative percentage for each latency value
    cumulative_percentage = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency) * 100

    # Plot the CDF
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_percentage, sorted_latency, linestyle='-', marker="x", linewidth=1)
    plt.ylabel('Latency (µs)')
    plt.xlabel('Cumulative Percentage (%)')
    plt.title('Cumulative Distribution Function (CDF) of Latency')


    plt.grid(True)
    plt.savefig(f"{output}/cdf.png")
    plt.close()

def plot_stream(stream_name, values, sub_name, output):
    plt.plot(range(len(values)), values)
    plt.xscale("log")
    plt.xlabel("Samples value")
    plt.ylabel('Latency (µs)')
    plt.title('Stream: {}'.format(stream_name))
    plt.savefig(f"{output}/plot_{sub_name}.png")
    print(f"Plot saved as 'plot_{sub_name}.png'.")

def generate_adoc(pub, sub, output):
    sub_name = sub.split("_")[4]
    with open(f"{output}/{ADOC_FILE_PATH}", "w", encoding="utf-8") as adoc_file:
        adoc_file.write("== Latency tests\n")
        latency_block = textwrap.dedent(
                """
                === Subscriber {_sub_name_} latency test on {_size_} samples value
                |===
                |Number of stream |Minimum latency |Maximum latency |Average latency
                |{_stream_} |{_minlat_} us |{_maxlat_} us |{_avglat_} us
                |Number of latencies < 0: {_neglat_} ({_neg_percentage_}%)
                |===
                image::{_output_}/latency_histogram_{_sub_name_}.png[]
                """
        )

        pacing_block = textwrap.dedent(
                """
                == Pacing tests
                |===
                |Publisher minimum pacing |Publisher maximum pacing |Publisher average pacing
                |{_pub_minpace_} us |{_pub_maxpace_} us |{_pub_avgpace_} us
                |Subscriber minimum pacing |Subscriber maximum pacing |Subscriber average pacing
                |{_sub_minpace_} us |{_sub_maxpace_} us |{_sub_avgpace_} us
                """
        )

        pub_sv = extract_sv(pub)
        sub_sv = extract_sv(sub)
        stream_name, latencies = compute_latency(pub_sv, sub_sv, output)
        filename = save_latency_histogram(latencies,sub_name,output)
        plot_stream(stream_name, latencies, sub_name, output)
        plot_cdf(latencies, output)

        adoc_file.write(
                latency_block.format(
                    _sub_name_=sub_name,
                    _stream_= get_stream_count(output),
                    _minlat_= compute_min(latencies),
                    _maxlat_= compute_max(latencies),
                    _avglat_= compute_average(latencies),
                    _neglat_ = compute_neglat(latencies),
                    _size_ = compute_size(latencies),
                    _neg_percentage_ = np.round(compute_neglat(latencies) / compute_size(latencies),5) *100,
                    _output_= filename
                )
        )

        adoc_file.write(
                pacing_block.format(
                    _pub_minpace_= compute_min(pub_pacing),
                    _pub_maxpace_= compute_max(pub_pacing),
                    _pub_avgpace_= compute_average(pub_pacing),
                    _sub_minpace_= compute_min(sub_pacing),
                    _sub_maxpace_= compute_max(sub_pacing),
                    _sub_avgpace_= compute_average(sub_pacing),
                    _output_= filename
                )
        )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Latency tests report in AsciiDoc format.")
    parser.add_argument("--pub", "-p", type=str, required=True, help="SV publisher file")
    parser.add_argument("--sub", "-s", type=str, required=True, help="SV subscriber file")
    parser.add_argument("--output", "-o", default="../results/", type=str, help="Output directory for the generated files.")

    args = parser.parse_args()
    generate_adoc(args.pub, args.sub, args.output)
