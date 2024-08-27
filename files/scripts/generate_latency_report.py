import os
import argparse
import matplotlib.pyplot as plt
import textwrap
import numpy as np

ADOC_FILE_PATH = f"latency-tests-report.adoc"

def compute_pacing(sv):
    streams = len(sv)
    pacing = [[0]] * len(sv)
    for stream in range(0, streams):
        pacing[stream] = np.diff(sv[stream][2])
    return pacing

def detect_sv_drop(pub_sv, sub_sv, iteration_size=4000):
# This function is used to detect if there are any missed SV's in
# subscriber data, by testing the continuity of the SV counter of
# subscriber data.

    total_sv_drops = 0
    pub_sv_iter = np.sort(np.unique(pub_sv[0].astype(int)))

    for iteration in range(len(pub_sv_iter)):
        sub_sv_current_iter = np.where(sub_sv[0].astype(int) == iteration)[0]
        sub_sv_start_index = sub_sv_current_iter[0]
        sub_sv_end_index = sub_sv_current_iter[-1]+1
        sub_sv_cnt = sub_sv[1][sub_sv_start_index:sub_sv_end_index]

        diffs = np.diff(sub_sv_cnt) - 1
        neg_diffs = np.where(diffs < 0)[0]

        if neg_diffs.size > 0:
            print("Fatal: SV disordered detected")
            exit(1)

        if iteration_size-sub_sv_cnt[-1] > 0:
            diffs[-1] = iteration_size-sub_sv_cnt[-1] - 1
        if sub_sv_cnt[0] > 0:
            diffs[0] = sub_sv_cnt[0] - 1

        discontinuities = np.where(diffs > 0)[0]

        for disc in discontinuities:
            num_lost_values = diffs[disc]

            if num_lost_values == diffs[-1]:
                disc += 1
            if disc == 0:
                disc += -1
                num_lost_values += 1
            for _ in range(num_lost_values):
                pub_sv[0] = np.delete(pub_sv[0], sub_sv_start_index + disc + 1 )
                pub_sv[1] = np.delete(pub_sv[1], sub_sv_start_index + disc + 1 )
                pub_sv[2] = np.delete(pub_sv[2], sub_sv_start_index + disc + 1 )
            total_sv_drops += num_lost_values

    return total_sv_drops

def investigate_array_differences(array1, array2):
    # This function checks if pub and sub counter are well aligned.
    len1 = len(array1)
    len2 = len(array2)

    min_len = min(len1, len2)
    max_len = max(len1, len2)

    diff_indices = np.where(array1[:min_len] != array2[:min_len])[0]
    diffs = [(i, array1[i], array2[i]) for i in diff_indices]

    if len1 > len2:
        extra_elements = array1[len2:max_len]
        extra_info = {'array': 'array1', 'indices': np.arange(len2, max_len), 'values': extra_elements}
    elif len2 > len1:
        extra_elements = array2[len1:max_len]
        extra_info = {'array': 'array2', 'indices': np.arange(len1, max_len), 'values': extra_elements}
    else:
        extra_info = None

    return diffs, extra_info

def compute_latency(pub_sv, sub_sv):
    latencies = [[0]] * len(pub_sv)
    sv_drop = 0
    for stream in range(0, len(pub_sv)):
        if len(pub_sv[stream][1]) != len(sub_sv[stream][1]):
            sv_drop = detect_sv_drop(pub_sv[stream], sub_sv[stream])
            diffs, extra_info = investigate_array_differences(pub_sv[stream][1], sub_sv[stream][1])

            if diffs:
                print("Warning: SV counter misalignment between pub and sub")
            if extra_info:
                print(f"Warning: Extra elements in {extra_info['array']} at indices {extra_info['indices']}: {extra_info['values']}")
        latencies[stream] = sub_sv[stream][2] - pub_sv[stream][2]

        stream_name = stream

    return stream_name, latencies, sv_drop


def extract_sv(sv_file_path):
    stream_number = 0
    with open(f"{sv_file_path}", "r", encoding="utf-8") as sv_file:
        sv_content = sv_file.read().splitlines()

    sv_id = np.array([str(item.split(":")[1]) for item in sv_content])
    stream_names = np.unique(sv_id)

    # Initialize sv as a list of empty lists
    sv = [[] * len(stream_names)]

    sv_it = np.array([str(item.split(":")[0]) for item in sv_content])
    sv_cnt = np.array([int(item.split(":")[2]) for item in sv_content])
    sv_timestamps = np.array([int(item.split(":")[3]) for item in sv_content])

    for items in stream_names:
        id_occurrences = np.where(sv_id == items)

        sv_it_occurrences = sv_it[id_occurrences]
        sv_cnt_occurrences = sv_cnt[id_occurrences]
        sv_timestamps_occurrences = sv_timestamps[id_occurrences]

        # Append a new sublist containing the three arrays
        sv[stream_number] = [sv_it_occurrences, sv_cnt_occurrences, sv_timestamps_occurrences]

        stream_number += 1

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
    streams = len(values)
    indices_exceeding_threshold = [[0]] * len(values)
    for stream in range(0, streams):
        indices_exceeding_threshold[stream] = np.where(values[stream] > threshold)[0]
    return indices_exceeding_threshold

def save_sv_lat_threshold(data_type, latencies, SVs, indices_exceeding_threshold, output):
    streams = len(SVs)

    for stream in range(0, streams):

        file_name = f"{output}/sv_{data_type}_exceed_stream_{stream}"

        with open(file_name, "w", encoding="utf-8") as sv_lat_exceed_file:
            for exceeding_lat in indices_exceeding_threshold[stream]:
                iteration = SVs[stream][0][exceeding_lat]
                sv_cnt = SVs[stream][1][exceeding_lat]
                latency = latencies[stream][exceeding_lat]
                sv_lat_exceed_file.write(f"SV {iteration}-{stream}-{sv_cnt} {latency}us\n")

def compute_size(values):
    return np.size(values)

def save_histogram(plot_type, values, sub_name, output):
    streams = len(values)

    for stream in range(0, streams):
        # Plot latency histograms
        plt.hist(values[stream], bins=20, alpha=0.7)

        # Add titles and legends
        plt.xlabel(f"{plot_type} (us)")
        plt.ylabel("Occurrences")
        plt.yscale('log')
        plt.title(f"{sub_name} {plot_type} Histogram")

        # Save the plot
        if not os.path.exists(output):
            os.makedirs(output)
        filename = f"histogram_{sub_name}_stream_{stream}_{plot_type}.png"
        filepath = os.path.realpath(f"{output}/{filename}")
        plt.savefig(filepath)
        print(f"Histogram saved as {filename}.")
        plt.close()

    return filepath

def plot_stream(stream_name, plot_type, values, lat_name, output):
    streams = len(values)

    for stream in range(0, streams):
        plt.plot(range(len(values[stream])), values[stream])
        plt.xlabel("Samples value")
        plt.ylabel(f'{plot_type} (µs)')
        plt.title(f'{lat_name} {plot_type} over time, Stream: {stream_name}')

        lat_name = lat_name.replace(" ", "_")

        filename = f"plot_{lat_name}_stream_{stream}_{plot_type}.png"
        filepath = os.path.realpath(f"{output}/{filename}")
        plt.savefig(filepath)
        print(f"Plot saved as {filename}.")
        plt.close()

def generate_adoc(pub, hyp, sub, output, ttot):
    sub_name = sub.split("_")[3]
    with open(f"{output}/{ADOC_FILE_PATH}", "w", encoding="utf-8") as adoc_file:

        total_latency_block = textwrap.dedent(
                """
                === Total latency on {_sub_name_}
                |===
                |Number of stream |Minimum latency |Maximum latency |Average latency
                |{_stream_} |{_minlat_} us |{_maxlat_} us |{_avglat_} us
                |Number of latencies < 0us: {_neglat_} ({_neg_percentage_}%)
                |Number of latencies > {_Ttot_}us: {_lat_Ttot_}
                |SV drop: {_total_sv_drop_}
                |===
                """
        )

        seapath_latency_block = textwrap.dedent(
                """
                === Seapath latency on {_sub_name_}
                |===
                |Minimum latency |Maximum latency |Average latency
                |{_minnetlat_} us |{_maxnetlat_} us |{_avgnetlat_} us
                |Number of latencies > {_Ttot_}us: {_lat_Tseap_}
                |SV drop: {_seapath_sv_drop_}
                |===
                """
        )

        pub_block = textwrap.dedent(
            """
            === Publisher
            |===
            |Minimun pacing |Maximum pacing |Average pacing
            |{_pub_minpace_} us |{_pub_maxpace_} us |{_pub_avgpace_} us
            |===
            """
        )
        hyp_block = textwrap.dedent(
                """
                === Hypervisor
                |===
                |Minimun pacing |Maximum pacing |Average pacing
                |{_hyp_minpace_} us |{_hyp_maxpace_} us |{_hyp_avgpace_} us
                |===
                """
        )
        sub_block = textwrap.dedent(
                """
                === Subscriber {_sub_name_}
                |===
                |Minimun pacing |Maximum pacing |Average pacing
                |{_sub_minpace_} us |{_sub_maxpace_} us |{_sub_avgpace_} us
                |===\n
                """
        )
        if pub is not None:
            pub_sv = extract_sv(pub)
            sub_sv = extract_sv(sub)
            stream_name, total_latencies, total_sv_drop = compute_latency(pub_sv, sub_sv)
            pub_pacing = compute_pacing(pub_sv)
            pub_pacing_exceeding_threshold = compute_lat_threshold(pub_pacing, 280)
            total_lat_exceeding_threshold = compute_lat_threshold(total_latencies, ttot)
            save_sv_lat_threshold("total_latency", total_latencies, pub_sv,  total_lat_exceeding_threshold, output)
            save_sv_lat_threshold("publisher_pacing", pub_pacing, pub_sv, pub_pacing_exceeding_threshold, output)
            total_lat_filename = save_histogram("latency", total_latencies,"total",output)
            plot_stream(stream_name,"latency", total_latencies, "total", output)
            save_histogram("pacing", pub_pacing,"publisher",output)
            plot_stream(stream_name,"pacing", pub_pacing, "publisher", output)


        sub_sv = extract_sv(sub)
        hyp_sv = extract_sv(hyp)
        stream_name, seapath_latencies, seapath_sv_drop = compute_latency(hyp_sv, sub_sv)
        hyp_pacing = compute_pacing(hyp_sv)
        sub_pacing = compute_pacing(sub_sv)
        seapath_lat_exceeding_threshold = compute_lat_threshold(seapath_latencies, ttot)
        hyp_pacing_exceeding_threshold = compute_lat_threshold(hyp_pacing, 280)
        sub_pacing_exceeding_threshold = compute_lat_threshold(sub_pacing, 280)

        save_sv_lat_threshold("seapath_latency", seapath_latencies, sub_sv,  seapath_lat_exceeding_threshold, output)
        save_sv_lat_threshold("hypervisor_pacing", hyp_pacing, hyp_sv, hyp_pacing_exceeding_threshold, output)
        save_sv_lat_threshold("subscriber_pacing", sub_pacing, sub_sv, sub_pacing_exceeding_threshold, output)


        seap_lat_filename = save_histogram("latency", seapath_latencies,"seapath",output)
        plot_stream(stream_name,"latency", seapath_latencies, "seapath", output)

        save_histogram("pacing", hyp_pacing,"hypervisor",output)
        plot_stream(stream_name,"pacing", hyp_pacing, "hypervisor", output)

        save_histogram("pacing", sub_pacing,"subscriber",output)
        plot_stream(stream_name,"pacing", sub_pacing, "subscriber", output)


        streams = len(sub_sv)

        for stream in range(0, streams):
            adoc_file.write("== Stream {_stream_} Latency tests on {_size_} samples value\n".format(
                _size_=compute_size(seapath_latencies[stream]),
                _stream_=stream))

            if pub is not None:
                adoc_file.write(
                        total_latency_block.format(
                            _sub_name_=sub_name,
                            _stream_= get_stream_count(output),
                            _minlat_= compute_min(total_latencies[stream]),
                            _maxlat_= compute_max(total_latencies[stream]),
                            _avglat_= compute_average(total_latencies[stream]),
                            _neglat_ = compute_neglat(total_latencies[stream]),
                            _neg_percentage_ = np.round(compute_neglat(total_latencies[stream]) / compute_size(total_latencies[stream]),5) *100,
                            _Ttot_ = ttot,
                            _lat_Ttot_ = len(total_lat_exceeding_threshold[0]),
                            _total_sv_drop_ = total_sv_drop
                        )
                )

            adoc_file.write(
                    seapath_latency_block.format(
                        _sub_name_=sub_name,
                        _minnetlat_= compute_min(seapath_latencies[stream]),
                        _maxnetlat_= compute_max(seapath_latencies[stream]),
                        _avgnetlat_= compute_average(seapath_latencies[stream]),
                        _Ttot_ = ttot,
                        _lat_Tseap_ = len(seapath_lat_exceeding_threshold[0]),
                        _seapath_sv_drop_ = seapath_sv_drop
                    )
            )
            adoc_file.write("== Pacing tests\n")
            if pub is not None:
                adoc_file.write(
                        pub_block.format(
                            _sub_name_=sub_name,
                            _pub_minpace_= compute_min(pub_pacing[stream]),
                            _pub_maxpace_= compute_max(pub_pacing[stream]),
                            _pub_avgpace_= compute_average(pub_pacing[stream]),
                        )
                )
            adoc_file.write(
                    hyp_block.format(
                        _hyp_minpace_= compute_min(hyp_pacing[stream]),
                        _hyp_maxpace_= compute_max(hyp_pacing[stream]),
                        _hyp_avgpace_= compute_average(hyp_pacing[stream]),
                    )
            )
            adoc_file.write(
                    sub_block.format(
                        _sub_name_=sub_name,
                        _sub_minpace_= compute_min(sub_pacing[stream]),
                        _sub_maxpace_= compute_max(sub_pacing[stream]),
                        _sub_avgpace_= compute_average(sub_pacing[stream]),
                    )
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Latency tests report in AsciiDoc format.")
    parser.add_argument("--pub", "-p", type=str, help="SV publisher file")
    parser.add_argument("--hyp", "-y", type=str, required=True, help="SV hypervisor file")
    parser.add_argument("--sub", "-s", type=str, required=True, help="SV subscriber file")
    parser.add_argument("--output", "-o", default="../results/", type=str, help="Output directory for the generated files.")
    parser.add_argument("--ttot", default=100, type=int, help="Total latency threshold.")

    args = parser.parse_args()
    generate_adoc(args.pub, args.hyp, args.sub, args.output, args.ttot)
