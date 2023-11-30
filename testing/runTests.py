import subprocess
import sys
import os


def main():
    n = len(sys.argv)

    if (n != 3):
        print("Usage: python "+ sys.argv[0] +" <inputFile> <char to search for in char search test>")
        return
    inputFile = sys.argv[1]
    character = sys.argv[2]

    print("Running Count Test...")

    if (sys.platform.startswith('linux')):
        subprocess.run(["./runCountTest", inputFile, character])

    elif (sys.platform.startswith('win32')):
        subprocess.run(["runCountTest.exe", inputFile, character])

    print("Running Euclidean Distance Test...")

    if (sys.platform.startswith('linux')):
        subprocess.run(["./runEDistTest", inputFile])

    elif (sys.platform.startswith('win32')):
        subprocess.run(["runEDistTest.exe", inputFile])

if __name__ == "__main__":
    main()