# CMake generated Testfile for 
# Source directory: /Users/piyushajitchavan/Desktop/tensor_alloc
# Build directory: /Users/piyushajitchavan/Desktop/tensor_alloc/build_release
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_size_class "/Users/piyushajitchavan/Desktop/tensor_alloc/build_release/test_size_class")
set_tests_properties(test_size_class PROPERTIES  _BACKTRACE_TRIPLES "/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;62;add_test;/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;0;")
add_test(test_validation "/Users/piyushajitchavan/Desktop/tensor_alloc/build_release/test_validation")
set_tests_properties(test_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;66;add_test;/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;0;")
add_test(test_api "/Users/piyushajitchavan/Desktop/tensor_alloc/build_release/test_api")
set_tests_properties(test_api PROPERTIES  _BACKTRACE_TRIPLES "/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;71;add_test;/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;0;")
add_test(test_alignment "/Users/piyushajitchavan/Desktop/tensor_alloc/build_release/test_alignment")
set_tests_properties(test_alignment PROPERTIES  _BACKTRACE_TRIPLES "/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;77;add_test;/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;0;")
add_test(test_reuse "/Users/piyushajitchavan/Desktop/tensor_alloc/build_release/test_reuse")
set_tests_properties(test_reuse PROPERTIES  _BACKTRACE_TRIPLES "/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;83;add_test;/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;0;")
add_test(test_invalid_inputs "/Users/piyushajitchavan/Desktop/tensor_alloc/build_release/test_invalid_inputs")
set_tests_properties(test_invalid_inputs PROPERTIES  _BACKTRACE_TRIPLES "/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;89;add_test;/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;0;")
add_test(test_multithreading "/Users/piyushajitchavan/Desktop/tensor_alloc/build_release/test_multithreading")
set_tests_properties(test_multithreading PROPERTIES  _BACKTRACE_TRIPLES "/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;95;add_test;/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;0;")
add_test(test_trace_recorder "/Users/piyushajitchavan/Desktop/tensor_alloc/build_release/test_trace_recorder")
set_tests_properties(test_trace_recorder PROPERTIES  _BACKTRACE_TRIPLES "/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;101;add_test;/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;0;")
add_test(test_trace_pipeline "/opt/homebrew/Frameworks/Python.framework/Versions/3.15/bin/python3.15" "/Users/piyushajitchavan/Desktop/tensor_alloc/tests/test_trace_pipeline.py")
set_tests_properties(test_trace_pipeline PROPERTIES  _BACKTRACE_TRIPLES "/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;106;add_test;/Users/piyushajitchavan/Desktop/tensor_alloc/CMakeLists.txt;0;")
