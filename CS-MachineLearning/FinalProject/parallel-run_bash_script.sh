while read -r method_name
do
    for i in 0 1 2 3 4
    do
        file_name="even-data/even_data_np/even-${i}-train_test_val.npz"

        sem -j 1 python3 ml_project.py -f $file_name -c $method_name
    done
done < method_names.txt

echo "---PARALLEL RUN COMPLETED---"
