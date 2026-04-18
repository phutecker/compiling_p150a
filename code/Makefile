CXX ?= g++
BASE_CXXFLAGS := -std=c++11 -O3 -march=native -mtune=native -flto -funroll-loops
WARN_CXXFLAGS := -Wall -Wextra -Wpedantic
CXXFLAGS ?= $(BASE_CXXFLAGS) $(WARN_CXXFLAGS)
LDFLAGS ?= -flto
TARGET := bin/render_discrete
SRC := y_me_dijo_tu_mama.cpp
INPUT_DIR := input
OUTPUT_DIR := output

PKG_CONFIG := $(shell command -v pkg-config 2>/dev/null)
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null)

$(TARGET): $(SRC) | bin
	@test -n "$(PKG_CONFIG)" || (echo "pkg-config is missing. Install: sudo apt install g++ make pkgconf libopencv-dev ffmpeg" && exit 1)
	@test -n "$(strip $(OPENCV_LIBS))" || (echo "OpenCV 4 development files were not found. Install: sudo apt install libopencv-dev" && exit 1)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(OPENCV_CFLAGS) $(OPENCV_LIBS)

.PHONY: dirs
dirs:
	mkdir -p $(INPUT_DIR) $(OUTPUT_DIR) bin

bin:
	mkdir -p bin

.PHONY: prepare
prepare: dirs
	./scripts/prepare_inputs.sh $(INPUT_DIR) $(OUTPUT_DIR)

.PHONY: clear
clear: dirs
	./scripts/clear_outputs.sh $(OUTPUT_DIR)

.PHONY: run
run: dirs $(TARGET)
	./$(TARGET)

.PHONY: all
all: clear prepare run

.PHONY: debug
debug: CXXFLAGS = -std=c++11 -O0 -g3 $(WARN_CXXFLAGS)
debug: LDFLAGS =
debug: dirs $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)
