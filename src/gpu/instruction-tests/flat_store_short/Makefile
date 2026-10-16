CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3
AMDGPU_TARGET ?= gfx950
CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps
TARGET = flat_store_short
all: $(TARGET)
$(TARGET): flat_store_short.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
clean:
	rm -f $(TARGET)
.PHONY: all clean
