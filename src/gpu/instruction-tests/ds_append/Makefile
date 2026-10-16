CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# Target CDNA4 (MI350 series).
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = ds_append

all: $(TARGET)

$(TARGET): ds_append.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
