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
$ python testFileGen.py <rows> <output file>
```
- **Rows**: Dictates the line amount and numbers per line (NxN file)

- **Output File**: Name of the output file


## Running Tests
```
$ cd testing
$ python runTest.py <input file> <rows> <char>
```

- **Input file**: 
A file generated with the generateTestFile.py file


- **Rows**:
 Rows in the input file (Same number used to generate the file)

- **Char**: 
Character you want to search for in the file (should be a number between 0-9)
