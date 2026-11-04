# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# The host tools, built on the F2 instance against AWS's SDK
# (SDK_DIR, set by aws-fpga's sdk_setup.sh). C++23 with <print>: GCC 14 or
# newer (CXX=g++-14 on Ubuntu 24.04).

CXX ?= g++
CXXFLAGS = -std=c++23 -O2 -Wall -Wextra -Werror -I$(SDK_DIR)/userspace/include
# The SDK's own build of the library, as its tools link it (sdk_setup.sh
# also installs it, in a folder the linker may not search).
LDFLAGS = -L$(SDK_DIR)/userspace/lib/so -Wl,-rpath,$(SDK_DIR)/userspace/lib/so
LDLIBS = -lfpga_mgmt

all: bringup core_driver

# M8's bring-up tool.
bringup: bringup.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# The runtime's driver (sw/f2_platform.py); sw/tests/test_f2_platform.py
# builds it against a fake card instead.
core_driver: core_driver.cpp card_f2.cpp card.h ../../../sw/driver_protocol.h
	$(CXX) $(CXXFLAGS) -o $@ core_driver.cpp card_f2.cpp $(LDFLAGS) $(LDLIBS)

clean:
	rm -f bringup core_driver

.PHONY: all clean
