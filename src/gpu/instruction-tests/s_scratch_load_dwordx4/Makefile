CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# Target CDNA4 (MI350 series).
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = s_scratch_load_dwordx4

all: $(TARGET)

$(TARGET): s_scratch_load_dwordx4.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
