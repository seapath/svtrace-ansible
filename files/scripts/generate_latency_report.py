import os
import argparse
import matplotlib.pyplot as plt
import textwrap
import numpy as np

ADOC_FILE_PATH = f"latency-tests-report.adoc"

def compute_pacing(pub_sv, sub_sv):
    pub_pacing = np.diff(pub_sv[3])
    sub_pacing = np.diff(sub_sv[3])

    return pub_pacing, sub_pacing

def detect_sv_drop(sv_counter):
    diffs = np.diff(sv_counter)
    diffs = diffs - 1
    discontinuities = np.where(diffs > 0)

    if discontinuities[0].size > 0:
        discontinuities = np.stack((sv_counter[:-1],diffs))
    return discontinuities

def compute_latency(pub_sv, sub_sv):
    # Detect discontinuities
    pub_discontinuities = detect_sv_drop(pub_sv[2])
    sub_discontinuities = detect_sv_drop(sub_sv[2])

    if sub_discontinuities[0].size > 0:
        sv_discontinuities = np.where(sub_discontinuities[2] > 0)

        for sv_discontinuity in sv_discontinuities[0]:
            sv_dropped = sub_discontinuities[1][sv_discontinuity]

            for sv in range(0, sv_dropped):
                print(f"Warning: SV {pub_sv_cnt[sv_discontinuity+1]} dropped in subscriber data")
                pub_timestamps = np.delete(pub_timestamps,sv_discontinuity+1)
                pub_sv_cnt = np.delete(pub_sv_cnt,sv_discontinuity+1)

    latencies = sub_sv[3] - pub_sv[3]

    stream_name = pub_sv[0][0]

    return stream_name, latencies


def extract_sv(sv_file_path):

    with open(f"{sv_file_path}", "r", encoding="utf-8") as sv_file:
        sv_content = sv_file.read().splitlines()

    sv_it = np.array([str(item.split(":")[0]) for item in sv_content])
    sv_id = np.array([str(item.split(":")[1]) for item in sv_content])
    sv_cnt = np.array([int(item.split(":")[2]) for item in sv_content])
    sv_timestamps = np.array([int(item.split(":")[3]) for item in sv_content])

    sv = [sv_it, sv_id, sv_cnt, sv_timestamps]

    return sv

def get_stream_count(pub_sv):
    return np.unique(pub_sv).size

def compute_min(values):
    return np.min(values) if values.size > 0 else None

def compute_max(values):
    return np.max(values) if values.size > 0 else None

def compute_average(values):
    return np.round(np.mean(values)) if values.size > 0 else None

def compute_neglat(values):
    return np.count_nonzero(values < 0)

def compute_lat_threshold(values, threshold):
    indices_exceeding_threshold = np.where(values > threshold)[0]
    return indices_exceeding_threshold

def save_sv_lat_threshold(data_type, latencies, SVs, indices_exceeding_threshold, output):
    file_name = f"{output}/sv_{data_type}_exceed"
    with open(file_name, "w", encoding="utf-8") as sv_lat_exceed_file:
        for exceeding_lat in indices_exceeding_threshold:
            iteration = SVs[0][exceeding_lat]
            sv_cnt = SVs[2][exceeding_lat]
            latency = latencies[exceeding_lat]
            sv_lat_exceed_file.write(f"SV {iteration}-{sv_cnt} {latency}us\n")

def compute_size(values):
    return np.size(values)

def save_histogram(plot_type, values, sub_name, output):
    # Plot latency histograms
    plt.hist(values, bins=20, alpha=0.7)

    # Add titles and legends
    plt.xlabel(f"{plot_type} (us)")
    plt.ylabel("Occurrences")
    plt.yscale('log')
    plt.title(f"{plot_type} Histogram for {sub_name}")

    # Save the plot
    if not os.path.exists(output):
        os.makedirs(output)
    filename = os.path.realpath(f"{output}/histogram_{plot_type}_{sub_name}.png")
    plt.savefig(filename)
    print(f"Histogram saved as 'histogram_{plot_type}_{sub_name}.png'.")
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
    plt.title('Cumulative Distribution Function (CDF) of total latency')


    plt.grid(True)
    plt.savefig(f"{output}/cdf.png")
    plt.close()

def plot_stream(stream_name, plot_type, values, lat_name, output):
    plt.plot(range(len(values)), values)
    plt.xlabel("Samples value")
    plt.ylabel(f'{plot_type} (µs)')
    plt.title('latency over time, Stream: {}'.format(stream_name))

    lat_name = lat_name.replace(" ", "_")
    plt.savefig(f"{output}/plot_{plot_type}_{lat_name}.png")
    print(f"Plot saved as 'plot_{plot_type}_{lat_name}.png'.")
    plt.close()

def percentage_hist(values, lat_name, output):
    max_value = max(values)
    bin_width = 100
    bins = np.arange(0, max_value + bin_width, bin_width)

    hist, bin_edges = np.histogram(values, bins=bins)
    percentages = (hist / len(values)) * 100

    bin_labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins)-1)]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_labels, percentages, color='skyblue')

    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{percentage:.2f}%', ha='center', va='bottom')

    plt.xlabel('Latency (µs)')
    plt.ylabel('Percentage')
    plt.title(f'Percentage distribution of {lat_name} latencies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output}/percentage_hist_{lat_name}.png")
    plt.close()

def generate_adoc(pub, hyp, sub, output, ttot, tnet):
    sub_name = sub.split("_")[4]
    with open(f"{output}/{ADOC_FILE_PATH}", "w", encoding="utf-8") as adoc_file:
        adoc_file.write("== Latency tests\n")
        sub_latency_block = textwrap.dedent(
                """
                === Subscriber {_sub_name_} latency test on {_size_} samples value
                |===
                |Number of stream |Minimum latency |Maximum latency |Average latency
                |{_stream_} |{_minlat_} us |{_maxlat_} us |{_avglat_} us
                |Number of latencies < 0: {_neglat_} ({_neg_percentage_}%)
                |Number of latencies > {_Ttot_}us {_lat_Ttot_}
                |===
                image::{_output_}/latency_histogram_{_sub_name_}.png[]
                """
        )

        network_latency_block = textwrap.dedent(
                """
                === Network latency
                |===
                |Minimum latency |Maximum latency |Average latency
                |{_minnetlat_} us |{_maxnetlat_} us |{_avgnetlat_} us
                |Number of latencies > {_Tnet_}us {_lat_Tnet_}
                |===
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
        hyp_sv = extract_sv(hyp)
        sub_sv = extract_sv(sub)
        stream_name, total_latencies = compute_latency(pub_sv, sub_sv)
        _, network_latencies = compute_latency(pub_sv, hyp_sv)
        pub_pacing, sub_pacing = compute_pacing(pub_sv, sub_sv)
        #TODO : hyp pacing
        total_lat_exceeding_threshold = compute_lat_threshold(total_latencies, ttot)
        network_lat_exceeding_threshold = compute_lat_threshold(network_latencies, tnet)
        pub_pacing_exceeding_threshold = compute_lat_threshold(pub_pacing, 280)
        sub_pacing_exceeding_threshold = compute_lat_threshold(sub_pacing, 280)

        save_sv_lat_threshold("total latency", total_latencies, pub_sv,  total_lat_exceeding_threshold, output)
        save_sv_lat_threshold("network latency", network_latencies, pub_sv,  network_lat_exceeding_threshold, output)
        save_sv_lat_threshold("publisher pacing", pub_pacing, pub_sv, pub_pacing_exceeding_threshold, output)
        save_sv_lat_threshold("subscriber pacing", sub_pacing, sub_sv, sub_pacing_exceeding_threshold, output)

        filename = save_histogram("latency", total_latencies,"total latency",output)
        filename = save_histogram("latency", network_latencies,"network latency",output)
        plot_stream(stream_name,"latency", total_latencies, "total latency", output)
        plot_stream(stream_name,"latency", network_latencies, "network latency", output)
        plot_cdf(total_latencies, output)

        save_histogram("Pacing", pub_pacing,"publisher",output)
        plot_stream(stream_name,"Pacing", pub_pacing, "publisher", output)

        save_histogram("Pacing", sub_pacing,"subscriber",output)
        plot_stream(stream_name,"Pacing", sub_pacing, "subscriber", output)
        percentage_hist(sub_pacing,"subscriber",output)
        percentage_hist(pub_pacing,"publisher",output)

        adoc_file.write(
                sub_latency_block.format(
                    _sub_name_=sub_name,
                    _stream_= get_stream_count(output),
                    _minlat_= compute_min(total_latencies),
                    _maxlat_= compute_max(total_latencies),
                    _avglat_= compute_average(total_latencies),
                    _neglat_ = compute_neglat(total_latencies),
                    _size_ = compute_size(total_latencies),
                    _neg_percentage_ = np.round(compute_neglat(total_latencies) / compute_size(total_latencies),5) *100,
                    _output_= filename,
                    _Ttot_ = ttot,
                    _lat_Ttot_ = len(total_lat_exceeding_threshold)
                )
        )

        adoc_file.write(
                network_latency_block.format(
                    _minnetlat_= compute_min(network_latencies),
                    _maxnetlat_= compute_max(network_latencies),
                    _avgnetlat_= compute_average(network_latencies),
                    _Tnet_ = tnet,
                    _lat_Tnet_ = len(network_lat_exceeding_threshold)
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
    parser.add_argument("--hyp", "-y", type=str, required=True, help="SV hypervisor file")
    parser.add_argument("--sub", "-s", type=str, required=True, help="SV subscriber file")
    parser.add_argument("--output", "-o", default="../results/", type=str, help="Output directory for the generated files.")
    parser.add_argument("--ttot", default=100, type=int, help="Total latency threshold.")
    parser.add_argument("--tnet", default=100, type=int, help="Network latency threshold.")

    args = parser.parse_args()
    generate_adoc(args.pub, args.hyp, args.sub, args.output, args.ttot, args.tnet)
