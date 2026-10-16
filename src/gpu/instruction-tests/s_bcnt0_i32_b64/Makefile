CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# CDNA4 only.
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = s_bcnt0_i32_b64

all: $(TARGET)

$(TARGET): s_bcnt0_i32_b64.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
