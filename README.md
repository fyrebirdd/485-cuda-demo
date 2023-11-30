## Build Instructions   
```
$ mkdir cmake-build
$ cd cmake-build
$ cmake ..
```
This will generate the solution file for visual studio in which you can then build the solution.

## Generating Tests
```
$ cd testing
$ python testFileGen.py <characters to generate> <output file>
```
- **Characters to generate**: How many characters to generate in the file

- **Output File**: Name of the output file


## Running Tests
```
$ cd testing
$ python runTest.py <input file> <char>
```

- **Input file**: 
A file generated with the generateTestFile.py file


- **Char**: 
Character you want to search for in the file (should be a number between 0-9), this will only be used with the counting test.


## Running the binaries on their own
```
$cd testing
$ ./runCountTest.exe <file> <char> on windows
$ ./runEDistTest.exe <file> on windows
or
$ ./runCountTest <file> <char> on linux
$ ./runEDistTest <file> on windows
```
