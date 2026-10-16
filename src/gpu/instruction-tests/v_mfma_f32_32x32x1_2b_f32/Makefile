CXX      = hipcc
CXXFLAGS = -O2 -std=c++17 -g3

# CDNA4 only. This variant is targeted at gfx950 (MI350 / CDNA4).
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = v_mfma_f32_32x32x1_2b_f32

all: $(TARGET)

$(TARGET): v_mfma_f32_32x32x1_2b_f32.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
