#!/bin/bash

set -e

usage() {
    cat <<EOF
    Latency tests script

    This script launches latency test on <it> iterations with <cycles> PCAP replay
    each.

Usage: multiple-tests.sh <it> <cycles> [target]

EOF
}

if [ -z "$2" ]; then
  usage
  exit 1
fi

SSH_OPTS="-i ~/.ssh/sfl-seapath -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"
VM_HOST="root@192.168.216.83"
it="$1"
pcap_cyles="$2"
dev="$3"

sed -i "s/^\(\s\{28\}pcap_cycles:\s*\)[0-9]\+/\1$pcap_cyles/" seapath-inventories/fromveur/common.yaml

for ((i=1; i<=it; i++)); do
    ssh $SSH_OPTS $VM_HOST reboot
    sleep 120

    if [ "$dev" == "vm" ]; then
        ansible-playbook -i seapath-inventories/fromveur/common.yaml -i seapath-inventories/fromveur/vms.yaml playbooks/configure_latency_tests.yaml
        ansible-playbook -i seapath-inventories/fromveur/common.yaml -i seapath-inventories/fromveur/vms.yaml playbooks/run_latency_tests.yaml
    else
        ansible-playbook -i seapath-inventories/fromveur/common.yaml -i seapath-inventories/fromveur/hyp.yaml playbooks/configure_latency_tests.yaml
        ansible-playbook -i seapath-inventories/fromveur/common.yaml -i seapath-inventories/fromveur/hyp.yaml playbooks/run_latency_tests.yaml
    fi
    mv tests_results tests_results-$(date +%F_%Hh%Mm%S)
done
