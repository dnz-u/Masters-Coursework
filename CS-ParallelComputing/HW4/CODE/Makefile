SRC_DIR = src
BIN_DIR = bin
RESULTS_DIR = results
INPUT_IMAGE = data/image_large.txt
INPUT_KERNEL = data/tiny_kernel2.txt

NVCC_FLAGS = -I $(SRC_DIR)

all: conv_cons conv_naive conv_tiled

conv_naive: $(SRC_DIR)/conv_naive.cu $(SRC_DIR)/util.cpp | $(BIN_DIR)
	nvcc $(NVCC_FLAGS) $(SRC_DIR)/conv_naive.cu $(SRC_DIR)/util.cpp -o $(BIN_DIR)/conv_naive

conv_cons: $(SRC_DIR)/conv_cons.cu $(SRC_DIR)/util.cpp | $(BIN_DIR)
	nvcc $(NVCC_FLAGS) $(SRC_DIR)/conv_cons.cu $(SRC_DIR)/util.cpp -o $(BIN_DIR)/conv_cons

conv_tiled: $(SRC_DIR)/conv_tiled.cu $(SRC_DIR)/util.cpp | $(BIN_DIR)
	nvcc $(NVCC_FLAGS) $(SRC_DIR)/conv_tiled.cu $(SRC_DIR)/util.cpp -o $(BIN_DIR)/conv_tiled

clean:
	rm -rf $(BIN_DIR)/* $(RESULTS_DIR)/*

run_naive: | $(RESULTS_DIR)
	./$(BIN_DIR)/conv_naive $(INPUT_IMAGE) $(INPUT_KERNEL) $(RESULTS_DIR)/naive_output.txt
	python3 text_to_jpg.py $(RESULTS_DIR)/naive_output.txt

run_cons: | $(RESULTS_DIR)
	./$(BIN_DIR)/conv_cons $(INPUT_IMAGE) $(INPUT_KERNEL) $(RESULTS_DIR)/cons_output.txt
	python3 text_to_jpg.py $(RESULTS_DIR)/cons_output.txt

run_tiled: | $(RESULTS_DIR)
	./$(BIN_DIR)/conv_tiled $(INPUT_IMAGE) $(INPUT_KERNEL) $(RESULTS_DIR)/tiled_output.txt
	python3 text_to_jpg.py $(RESULTS_DIR)/tiled_output.txt

run_default_image:
	python3 text_to_jpg.py $(INPUT_IMAGE)
