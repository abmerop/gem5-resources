CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# Target GFX9 (Vega / MI-series).  Adjust for your hardware:
#   gfx906  = Radeon VII / MI50 / MI60
#   gfx908  = MI100
#   gfx90a  = MI200 series
#   gfx942  = MI300X
#   gfx950  = MI350 / CDNA4
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = v_mfma_f32_16x16x4_f32

all: $(TARGET)

$(TARGET): v_mfma_f32_16x16x4_f32.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
