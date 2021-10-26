#!/bin/sh

echo "start report lcov coverage"
pushd ..
lcov -d build -b . --no-external -c -o gamma_test_coverage.info
genhtml -o gamma_test_coverage_report --prefix=`pwd` gamma_test_init_coverage.info gamma_test_coverage.info

popd
