import os
import argparse
import matplotlib.pyplot as plt
import textwrap
import numpy as np

ADOC_FILE_PATH = f"latency-tests-report.adoc"

def compute_pacing(sv):
    return np.diff(sv[3])

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
    plt.title(f"{sub_name} {plot_type} Histogram")

    # Save the plot
    if not os.path.exists(output):
        os.makedirs(output)
    filename = os.path.realpath(f"{output}/histogram_{plot_type}_{sub_name}.png")
    plt.savefig(filename)
    print(f"Histogram saved as 'histogram_{plot_type}_{sub_name}.png'.")
    plt.close()
    return filename

def plot_stream(stream_name, plot_type, values, lat_name, output):
    plt.plot(range(len(values)), values)
    plt.xlabel("Samples value")
    plt.ylabel(f'{plot_type} (µs)')
    plt.title(f'{lat_name} {plot_type} over time, Stream: {stream_name}')

    lat_name = lat_name.replace(" ", "_")
    plt.savefig(f"{output}/plot_{plot_type}_{lat_name}.png")
    print(f"Plot saved as 'plot_{plot_type}_{lat_name}.png'.")
    plt.close()

def generate_adoc(pub, hyp, sub, output, ttot):
    sub_name = sub.split("_")[2]
    with open(f"{output}/{ADOC_FILE_PATH}", "w", encoding="utf-8") as adoc_file:

        total_latency_block = textwrap.dedent(
                """
                === Total latency on {_sub_name_}
                |===
                |Number of stream |Minimum latency |Maximum latency |Average latency
                |{_stream_} |{_minlat_} us |{_maxlat_} us |{_avglat_} us
                |Number of latencies < 0us: {_neglat_} ({_neg_percentage_}%)
                |Number of latencies > {_Ttot_}us: {_lat_Ttot_}
                |===
                image::{_image_path_}[]
                """
        )

        seapath_latency_block = textwrap.dedent(
                """
                === Seapath latency on {_sub_name_}
                |===
                |Minimum latency |Maximum latency |Average latency
                |{_minnetlat_} us |{_maxnetlat_} us |{_avgnetlat_} us
                |Number of latencies > {_Ttot_}us: {_lat_Tseap_}
                |===
                image::{_image_path_}[]
                """
        )

        pacing_block = textwrap.dedent(
                """
                == Pacing tests
                === Publisher
                |===
                |Minimun pacing |Maximum pacing |Average pacing
                |{_pub_minpace_} us |{_pub_maxpace_} us |{_pub_avgpace_} us
                |===
                === Hypervisor
                |===
                |Minimun pacing |Maximum pacing |Average pacing
                |{_hyp_minpace_} us |{_hyp_maxpace_} us |{_hyp_avgpace_} us
                |===
                === Subscriber {_sub_name_}
                |===
                |Minimun pacing |Maximum pacing |Average pacing
                |{_sub_minpace_} us |{_sub_maxpace_} us |{_sub_avgpace_} us
                |===
                """
        )

        pub_sv = extract_sv(pub)
        hyp_sv = extract_sv(hyp)
        sub_sv = extract_sv(sub)
        stream_name, total_latencies = compute_latency(pub_sv, sub_sv)
        _, seapath_latencies = compute_latency(hyp_sv, sub_sv)
        pub_pacing = compute_pacing(pub_sv)
        hyp_pacing = compute_pacing(hyp_sv)
        sub_pacing = compute_pacing(sub_sv)
        total_lat_exceeding_threshold = compute_lat_threshold(total_latencies, ttot)
        seapath_lat_exceeding_threshold = compute_lat_threshold(seapath_latencies, ttot)
        pub_pacing_exceeding_threshold = compute_lat_threshold(pub_pacing, 280)
        hyp_pacing_exceeding_threshold = compute_lat_threshold(hyp_pacing, 280)
        sub_pacing_exceeding_threshold = compute_lat_threshold(sub_pacing, 280)

        save_sv_lat_threshold("total latency", total_latencies, pub_sv,  total_lat_exceeding_threshold, output)
        save_sv_lat_threshold("seapath latency", seapath_latencies, pub_sv,  seapath_lat_exceeding_threshold, output)
        save_sv_lat_threshold("publisher pacing", pub_pacing, pub_sv, pub_pacing_exceeding_threshold, output)
        save_sv_lat_threshold("hypervisor pacing", hyp_pacing, hyp_sv, hyp_pacing_exceeding_threshold, output)
        save_sv_lat_threshold("subscriber pacing", sub_pacing, sub_sv, sub_pacing_exceeding_threshold, output)

        total_lat_filename = save_histogram("latency", total_latencies,"total",output)
        plot_stream(stream_name,"latency", total_latencies, "total", output)

        seap_lat_filename = save_histogram("latency", seapath_latencies,"seapath",output)
        plot_stream(stream_name,"latency", seapath_latencies, "seapath", output)

        save_histogram("pacing", pub_pacing,"publisher",output)
        plot_stream(stream_name,"pacing", pub_pacing, "publisher", output)

        save_histogram("pacing", hyp_pacing,"hypervisor",output)
        plot_stream(stream_name,"pacing", hyp_pacing, "hypervisor", output)

        save_histogram("pacing", sub_pacing,"subscriber",output)
        plot_stream(stream_name,"pacing", sub_pacing, "subscriber", output)

        adoc_file.write("== Latency tests on {_size_} samples value\n".format(
            _size_=compute_size(total_latencies)))

        adoc_file.write(
                total_latency_block.format(
                    _sub_name_=sub_name,
                    _stream_= get_stream_count(output),
                    _minlat_= compute_min(total_latencies),
                    _maxlat_= compute_max(total_latencies),
                    _avglat_= compute_average(total_latencies),
                    _neglat_ = compute_neglat(total_latencies),
                    _neg_percentage_ = np.round(compute_neglat(total_latencies) / compute_size(total_latencies),5) *100,
                    _image_path_= total_lat_filename,
                    _Ttot_ = ttot,
                    _lat_Ttot_ = len(total_lat_exceeding_threshold)
                )
        )

        adoc_file.write(
                seapath_latency_block.format(
                    _sub_name_=sub_name,
                    _minnetlat_= compute_min(seapath_latencies),
                    _maxnetlat_= compute_max(seapath_latencies),
                    _avgnetlat_= compute_average(seapath_latencies),
                    _image_path_= seap_lat_filename,
                    _Ttot_ = ttot,
                    _lat_Tseap_ = len(seapath_lat_exceeding_threshold)
                )
        )

        adoc_file.write(
                pacing_block.format(
                    _sub_name_=sub_name,
                    _pub_minpace_= compute_min(pub_pacing),
                    _pub_maxpace_= compute_max(pub_pacing),
                    _pub_avgpace_= compute_average(pub_pacing),
                    _hyp_minpace_= compute_min(hyp_pacing),
                    _hyp_maxpace_= compute_max(hyp_pacing),
                    _hyp_avgpace_= compute_average(hyp_pacing),
                    _sub_minpace_= compute_min(sub_pacing),
                    _sub_maxpace_= compute_max(sub_pacing),
                    _sub_avgpace_= compute_average(sub_pacing),
                )
        )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Latency tests report in AsciiDoc format.")
    parser.add_argument("--pub", "-p", type=str, required=True, help="SV publisher file")
    parser.add_argument("--hyp", "-y", type=str, required=True, help="SV hypervisor file")
    parser.add_argument("--sub", "-s", type=str, required=True, help="SV subscriber file")
    parser.add_argument("--output", "-o", default="../results/", type=str, help="Output directory for the generated files.")
    parser.add_argument("--ttot", default=100, type=int, help="Total latency threshold.")

    args = parser.parse_args()
    generate_adoc(args.pub, args.hyp, args.sub, args.output, args.ttot)
