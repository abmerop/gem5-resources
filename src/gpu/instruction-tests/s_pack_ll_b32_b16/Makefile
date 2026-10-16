CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# Target GFX9 (Vega / MI-series).  Adjust for your hardware:
#   gfx906  = Radeon VII / MI50 / MI60
#   gfx908  = MI100
#   gfx90a  = MI200 series
#   gfx942  = MI300X
AMDGPU_TARGET ?= gfx942,gfx90a,gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = s_pack_ll_b32_b16

all: $(TARGET)

$(TARGET): s_pack_ll_b32_b16.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
