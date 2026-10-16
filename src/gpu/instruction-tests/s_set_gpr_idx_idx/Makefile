CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = s_set_gpr_idx_idx

all: $(TARGET)

$(TARGET): s_set_gpr_idx_idx.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
