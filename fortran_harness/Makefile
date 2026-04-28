# Standalone build of the ICON-port physics drivers used by the harness.
# Module compile order is mandatory (.mod files for dependencies must
# exist before each compile). Listed in topological order.

FC      ?= gfortran
FFLAGS  ?= -O2 -g -cpp -ffree-line-length-none -fimplicit-none -fbacktrace \
           -D__LP64__ -D__ICON__ \
           -Wall -Wno-conversion -Wno-unused-variable -Wno-unused-dummy-argument \
           -Wno-compare-reals -Wno-character-truncation
LDFLAGS ?=

SRCDIR  := src
BUILDDIR := build

# Shared modules used by both drivers
COMMON_SRCS := \
    $(SRCDIR)/mo_kind.f90 \
    $(SRCDIR)/mo_exception.f90 \
    $(SRCDIR)/mo_physical_constants.f90 \
    $(SRCDIR)/mo_math_constants.f90 \
    $(SRCDIR)/mo_echam_cld_config.f90 \
    $(SRCDIR)/mo_echam_convect_tables.f90

# cumastr-specific modules
CUMASTR_SRCS := \
    $(COMMON_SRCS) \
    $(SRCDIR)/mo_echam_cnv_config.f90 \
    $(SRCDIR)/mo_cuadjust.f90 \
    $(SRCDIR)/mo_cuinitialize.f90 \
    $(SRCDIR)/mo_cuascent.f90 \
    $(SRCDIR)/mo_cudescent.f90 \
    $(SRCDIR)/mo_cufluxdts.f90 \
    $(SRCDIR)/mo_cumastr.f90 \
    $(SRCDIR)/cumastr_driver.f90

# mo_cloud-specific modules
CLOUD_SRCS := \
    $(COMMON_SRCS) \
    $(SRCDIR)/mo_cloud.f90 \
    $(SRCDIR)/cloud_driver.f90

CUMASTR_OBJS := $(patsubst $(SRCDIR)/%.f90,$(BUILDDIR)/%.o,$(CUMASTR_SRCS))
CLOUD_OBJS   := $(patsubst $(SRCDIR)/%.f90,$(BUILDDIR)/%.o,$(CLOUD_SRCS))

CUMASTR_TARGET := $(BUILDDIR)/cumastr_driver
CLOUD_TARGET   := $(BUILDDIR)/cloud_driver

.PHONY: all cumastr cloud clean
all: cumastr cloud
cumastr: $(CUMASTR_TARGET)
cloud:   $(CLOUD_TARGET)

$(CUMASTR_TARGET): $(CUMASTR_OBJS)
	$(FC) $(FFLAGS) -J$(BUILDDIR) -o $@ $^ $(LDFLAGS)

$(CLOUD_TARGET): $(CLOUD_OBJS)
	$(FC) $(FFLAGS) -J$(BUILDDIR) -o $@ $^ $(LDFLAGS)

# Compile sources in declared order, putting .mod into BUILDDIR
$(BUILDDIR)/%.o: $(SRCDIR)/%.f90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -J$(BUILDDIR) -I$(BUILDDIR) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
