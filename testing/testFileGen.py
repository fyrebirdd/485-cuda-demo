import sys
import random


'''
    This script generates a NxM file with random numbers in it.
    EX. 
    
    python testFileGen.py 1000 1000 test.txt 
    
    will generate a file called test.txt with 1000 lines and 1000 random numbers per line.
'''
def main():
    if len(sys.argv) != 3:
        print("Usage: python "+ sys.argv[0] +" <number of rows> <output file name>")
        return
    
    numRows = int(sys.argv[1])
    outputFileName = sys.argv[2]

    totalNums = numRows * numRows
    totalNumsWritten = 0

    with open(outputFileName, "w") as f:
        for i in range(numRows):
            for j in range(numRows):
                f.write(str(random.randint(0, 9)))
                if j != numRows - 1:
                    f.write(" ")
                totalNumsWritten += 1
            if i != numRows - 1:
                f.write("\n")
            percentComplete = round((totalNumsWritten/totalNums) * 100, 3)
            print("", end=f"\r{percentComplete}% Complete")
    f.close()


if __name__ == "__main__":
    main()