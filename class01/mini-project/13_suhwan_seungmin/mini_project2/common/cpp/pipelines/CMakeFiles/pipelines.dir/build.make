# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/open_model_zoo/demos

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/open_model_zoo/demos

# Include any dependencies generated for this target.
include common/cpp/pipelines/CMakeFiles/pipelines.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include common/cpp/pipelines/CMakeFiles/pipelines.dir/compiler_depend.make

# Include the progress variables for this target.
include common/cpp/pipelines/CMakeFiles/pipelines.dir/progress.make

# Include the compile flags for this target's objects.
include common/cpp/pipelines/CMakeFiles/pipelines.dir/flags.make

common/cpp/pipelines/CMakeFiles/pipelines.dir/src/async_pipeline.cpp.o: common/cpp/pipelines/CMakeFiles/pipelines.dir/flags.make
common/cpp/pipelines/CMakeFiles/pipelines.dir/src/async_pipeline.cpp.o: common/cpp/pipelines/src/async_pipeline.cpp
common/cpp/pipelines/CMakeFiles/pipelines.dir/src/async_pipeline.cpp.o: common/cpp/pipelines/CMakeFiles/pipelines.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/open_model_zoo/demos/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object common/cpp/pipelines/CMakeFiles/pipelines.dir/src/async_pipeline.cpp.o"
	cd /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/cpp/pipelines/CMakeFiles/pipelines.dir/src/async_pipeline.cpp.o -MF CMakeFiles/pipelines.dir/src/async_pipeline.cpp.o.d -o CMakeFiles/pipelines.dir/src/async_pipeline.cpp.o -c /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines/src/async_pipeline.cpp

common/cpp/pipelines/CMakeFiles/pipelines.dir/src/async_pipeline.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pipelines.dir/src/async_pipeline.cpp.i"
	cd /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines/src/async_pipeline.cpp > CMakeFiles/pipelines.dir/src/async_pipeline.cpp.i

common/cpp/pipelines/CMakeFiles/pipelines.dir/src/async_pipeline.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pipelines.dir/src/async_pipeline.cpp.s"
	cd /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines/src/async_pipeline.cpp -o CMakeFiles/pipelines.dir/src/async_pipeline.cpp.s

common/cpp/pipelines/CMakeFiles/pipelines.dir/src/requests_pool.cpp.o: common/cpp/pipelines/CMakeFiles/pipelines.dir/flags.make
common/cpp/pipelines/CMakeFiles/pipelines.dir/src/requests_pool.cpp.o: common/cpp/pipelines/src/requests_pool.cpp
common/cpp/pipelines/CMakeFiles/pipelines.dir/src/requests_pool.cpp.o: common/cpp/pipelines/CMakeFiles/pipelines.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/open_model_zoo/demos/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object common/cpp/pipelines/CMakeFiles/pipelines.dir/src/requests_pool.cpp.o"
	cd /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/cpp/pipelines/CMakeFiles/pipelines.dir/src/requests_pool.cpp.o -MF CMakeFiles/pipelines.dir/src/requests_pool.cpp.o.d -o CMakeFiles/pipelines.dir/src/requests_pool.cpp.o -c /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines/src/requests_pool.cpp

common/cpp/pipelines/CMakeFiles/pipelines.dir/src/requests_pool.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pipelines.dir/src/requests_pool.cpp.i"
	cd /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines/src/requests_pool.cpp > CMakeFiles/pipelines.dir/src/requests_pool.cpp.i

common/cpp/pipelines/CMakeFiles/pipelines.dir/src/requests_pool.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pipelines.dir/src/requests_pool.cpp.s"
	cd /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines/src/requests_pool.cpp -o CMakeFiles/pipelines.dir/src/requests_pool.cpp.s

# Object files for target pipelines
pipelines_OBJECTS = \
"CMakeFiles/pipelines.dir/src/async_pipeline.cpp.o" \
"CMakeFiles/pipelines.dir/src/requests_pool.cpp.o"

# External object files for target pipelines
pipelines_EXTERNAL_OBJECTS =

intel64/Release/libpipelines.a: common/cpp/pipelines/CMakeFiles/pipelines.dir/src/async_pipeline.cpp.o
intel64/Release/libpipelines.a: common/cpp/pipelines/CMakeFiles/pipelines.dir/src/requests_pool.cpp.o
intel64/Release/libpipelines.a: common/cpp/pipelines/CMakeFiles/pipelines.dir/build.make
intel64/Release/libpipelines.a: common/cpp/pipelines/CMakeFiles/pipelines.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/open_model_zoo/demos/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library ../../../intel64/Release/libpipelines.a"
	cd /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines && $(CMAKE_COMMAND) -P CMakeFiles/pipelines.dir/cmake_clean_target.cmake
	cd /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pipelines.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
common/cpp/pipelines/CMakeFiles/pipelines.dir/build: intel64/Release/libpipelines.a
.PHONY : common/cpp/pipelines/CMakeFiles/pipelines.dir/build

common/cpp/pipelines/CMakeFiles/pipelines.dir/clean:
	cd /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines && $(CMAKE_COMMAND) -P CMakeFiles/pipelines.dir/cmake_clean.cmake
.PHONY : common/cpp/pipelines/CMakeFiles/pipelines.dir/clean

common/cpp/pipelines/CMakeFiles/pipelines.dir/depend:
	cd /home/ubuntu/open_model_zoo/demos && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/open_model_zoo/demos /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines /home/ubuntu/open_model_zoo/demos /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines /home/ubuntu/open_model_zoo/demos/common/cpp/pipelines/CMakeFiles/pipelines.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : common/cpp/pipelines/CMakeFiles/pipelines.dir/depend
