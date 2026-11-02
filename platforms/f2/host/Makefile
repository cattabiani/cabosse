# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
#
# The bring-up host tool, built on the F2 instance against AWS's SDK
# (SDK_DIR, set by aws-fpga's sdk_setup.sh). C++23 with <print>: GCC 14 or
# newer (CXX=g++-14 on Ubuntu 24.04).

CXX ?= g++
CXXFLAGS = -std=c++23 -O2 -Wall -Wextra -Werror -I$(SDK_DIR)/userspace/include
LDLIBS = -lfpga_mgmt

bringup: bringup.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f bringup

.PHONY: clean
