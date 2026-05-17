CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

AMDGPU_TARGET ?= gfx942,gfx90a

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = v_cmp_eq_f16

all: $(TARGET)

$(TARGET): v_cmp_eq_f16.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
