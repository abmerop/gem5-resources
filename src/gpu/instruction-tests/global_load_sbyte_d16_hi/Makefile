CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# Target GFX9 (Vega / MI-series).  These D16 GLOBAL instructions exist on
# both gfx90a (MI200) and gfx942 (MI300).
AMDGPU_TARGET ?= gfx942,gfx90a,gfx906

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = global_load_sbyte_d16_hi

all: $(TARGET)

$(TARGET): global_load_sbyte_d16_hi.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) *.s *.o *.bc *.hipi *.out *.d *-hip-* *.cui

.PHONY: all clean
