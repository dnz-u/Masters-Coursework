# COMPILE filtering_seq and filtering_omp
rm -f filtering_seq filtering_omp res_denoised_image.txt res_blurred_image.txt
g++ -Wall filtering_seq.cpp util.cpp -o filtering_seq
g++ -Wall -fopenmp filtering_omp.cpp util.cpp -o filtering_omp


# RUN TESTS
filepath="example-image-kernel"
testimage="$filepath/image_large.txt"
testkernel="$filepath/tiny_kernel2.txt"
blurredOutput="$filepath/res_blurred_image.txt"
denoisedOutput="$filepath/res_denoised_image.txt"

echo " "
export OMP_NUM_THREADS=1


echo " "
echo "*** ReadMe: The blur kernel was also used in denoise for runtime tests. ***"
echo " "

echo "**** SEQUENTIAL Test****"
echo " "
export OMP_NUM_THREADS=1
echo "Number of Threads: 1"
echo ">>>"
echo "ReadMe: Only for the sequential case, the time taken to print the resulting images to file is included in the result of the time command."
echo "ReadMe: The process of writing the result images to a file can be commented out in the main function."

time ./filtering_seq $testimage $testkernel $denoisedOutput $blurredOutput

python3 text_to_jpg.py $testimage
python3 text_to_jpg.py $blurredOutput
echo "------------------------------------"


echo " "
echo "**** PARALLEL ****"

for num_threads in 1 2 4 6
do
  echo " "
  export OMP_NUM_THREADS=$num_threads
  echo "Number of Threads: $num_threads"
  echo ">>>"
  time ./filtering_omp $testimage $testkernel
done

echo " "
echo " "
echo "@ execution finished."