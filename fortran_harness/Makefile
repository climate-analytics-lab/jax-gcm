# Standalone build of the ICON-port Tiedtke-Nordeng cumastr driver.
# Module compile order is mandatory (mod files for dependencies must
# exist before each compile). Listing in topological order.

FC      ?= gfortran
FFLAGS  ?= -O2 -g -cpp -ffree-line-length-none -fimplicit-none -fbacktrace \
           -D__LP64__ -D__ICON__ \
           -Wall -Wno-conversion -Wno-unused-variable -Wno-unused-dummy-argument \
           -Wno-compare-reals -Wno-character-truncation
LDFLAGS ?=

SRCDIR  := src
BUILDDIR := build

# Topological order. Modules first, driver last.
SRCS := \
    $(SRCDIR)/mo_kind.f90 \
    $(SRCDIR)/mo_exception.f90 \
    $(SRCDIR)/mo_physical_constants.f90 \
    $(SRCDIR)/mo_echam_cld_config.f90 \
    $(SRCDIR)/mo_echam_convect_tables.f90 \
    $(SRCDIR)/mo_echam_cnv_config.f90 \
    $(SRCDIR)/mo_cuadjust.f90 \
    $(SRCDIR)/mo_cuinitialize.f90 \
    $(SRCDIR)/mo_cuascent.f90 \
    $(SRCDIR)/mo_cudescent.f90 \
    $(SRCDIR)/mo_cufluxdts.f90 \
    $(SRCDIR)/mo_cumastr.f90 \
    $(SRCDIR)/cumastr_driver.f90

OBJS := $(patsubst $(SRCDIR)/%.f90,$(BUILDDIR)/%.o,$(SRCS))
TARGET := $(BUILDDIR)/cumastr_driver

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -J$(BUILDDIR) -o $@ $^ $(LDFLAGS)

# Compile sources in declared order, putting .mod into BUILDDIR
$(BUILDDIR)/%.o: $(SRCDIR)/%.f90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -J$(BUILDDIR) -I$(BUILDDIR) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
