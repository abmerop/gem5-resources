CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# CDNA4 only.
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = v_cmp_ne_i64

all: $(TARGET)

$(TARGET): v_cmp_ne_i64.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
