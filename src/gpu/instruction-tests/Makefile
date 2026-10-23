AMDGPU_TARGET ?= gfx950

SUBDIRS := $(shell find . -mindepth 1 -maxdepth 1 -type d | sort)

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ AMDGPU_TARGET=$(AMDGPU_TARGET)

clean:
	for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done

.PHONY: all clean
