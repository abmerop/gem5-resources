CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3
AMDGPU_TARGET ?= gfx950
CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps
TARGET = ds_wrxchg2st64_rtn_b64
all: $(TARGET)
$(TARGET): ds_wrxchg2st64_rtn_b64.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
clean:
	rm -f $(TARGET)
.PHONY: all clean
