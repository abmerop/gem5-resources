CXX     = hipcc
CXXFLAGS = -O2 -std=c++14

# Target GFX9 (Vega / MI-series).  Adjust for your hardware:
#   gfx906  = Radeon VII / MI50 / MI60
#   gfx908  = MI100
#   gfx90a  = MI200 series
AMDGPU_TARGET ?= gfx942,gfx90a

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = scalar_atomic_dec

all: $(TARGET)

$(TARGET): scalar_atomic_dec.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
