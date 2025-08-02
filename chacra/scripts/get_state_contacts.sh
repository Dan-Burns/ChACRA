#!/bin/bash
structure=$1
trajectory=$2
contacts_folder=$3
state=$4
n_jobs=$5




# can just have getcontacts in the same environment - only need vmdpython
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate getcontacts

get_dynamic_contacts.py \
--topology "$structure" \
--trajectory "$trajectory" \
--output "$contacts_folder/contacts/cont_state_$state.tsv" \
--cores "$n_jobs" \
--itypes "all" --distout \
--sele "protein" \
--sele2 "protein"

get_contact_frequencies.py \
--input_files "$contacts_folder/contacts/cont_state_$state.tsv" \
--output_file "$contacts_folder/freqs/freqs_state_$state.tsv"

