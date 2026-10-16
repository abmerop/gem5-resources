CXX      = hipcc
CXXFLAGS = -O2 -std=c++14 -g3

# Target CDNA4 (MI350 series) only.
AMDGPU_TARGET ?= gfx950

CXXFLAGS += --offload-arch=$(AMDGPU_TARGET) -save-temps

TARGET = v_qsad_pk_u16_u8

all: $(TARGET)

$(TARGET): v_qsad_pk_u16_u8.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
