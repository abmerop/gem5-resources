CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# Target CDNA4 (MI350 series).
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = v_cvt_scalef32_sr_pk32_fp6_f16

all: $(TARGET)

$(TARGET): v_cvt_scalef32_sr_pk32_fp6_f16.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
