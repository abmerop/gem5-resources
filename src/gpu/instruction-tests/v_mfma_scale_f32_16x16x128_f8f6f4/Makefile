CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# Target CDNA4 (MI350 series).
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = v_mfma_scale_f32_16x16x128_f8f6f4

all: $(TARGET)

$(TARGET): v_mfma_scale_f32_16x16x128_f8f6f4.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
