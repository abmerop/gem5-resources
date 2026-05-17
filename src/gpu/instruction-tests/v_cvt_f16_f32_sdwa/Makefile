CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

AMDGPU_TARGET ?= gfx942,gfx90a

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = v_cvt_f16_f32_sdwa

all: $(TARGET)

$(TARGET): v_cvt_f16_f32_sdwa.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
