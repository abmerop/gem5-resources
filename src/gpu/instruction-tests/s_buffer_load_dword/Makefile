CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# CDNA4 only.
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = s_buffer_load_dword

all: $(TARGET)

$(TARGET): s_buffer_load_dword.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
