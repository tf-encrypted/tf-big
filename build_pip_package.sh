TEMP=./pip-package
OUTPUT=./dist

set -e
set -x

rm -rf $TEMP
mkdir -p $TEMP
cp setup.py $TEMP
cp README.md $TEMP
# cp MAINFEST.in $TEMP

pip install tensorflow==1.13.1 && ./build_so_files.sh $TEMP
pip install tensorflow==1.13.2 && ./build_so_files.sh $TEMP
# pip install tensorflow==1.14.0 && ./build_so_files.sh $TEMP

mkdir -p $OUTPUT
./bundle_pip_package.sh $TEMP $OUTPUT

# rm -rf $TEMP
