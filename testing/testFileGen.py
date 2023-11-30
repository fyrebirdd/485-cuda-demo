import sys
import random
import time

'''
    This script generates a file with one line of N random numbers.
    EX. 
    
    python testFileGen.py 1000 test.txt 
    
    will generate a file called test.txt with 1000 random numbers on the first line.

    This script is used to generate test files for the Count and Euclidean Distance tests.
    It can generate about 2 million random numbers in 1 second.
'''
def estimate_file_size(num_digits):
    # Assuming ASCII encoding where each character is 1 byte
    digit_size = 1
    return num_digits * digit_size

def main():
    if len(sys.argv) != 3:
        print("Usage: python " + sys.argv[0] + " <# of numbers> <output file name>")
        return
    
    num = int(sys.argv[1])
    outputFileName = sys.argv[2]

    file_size_estimate = estimate_file_size(num)
    if (file_size_estimate >= 1000000000):
        file_size_estimate /= 1000000000
        print(f"Estimated file size: {file_size_estimate:.1f} GB")
    elif(file_size_estimate >= 1000000):
        file_size_estimate /= 1000000
        print(f"Estimated file size: {file_size_estimate:.1f} MB")
    elif(file_size_estimate >= 1000):
        file_size_estimate /= 1000
        print(f"Estimated file size: {file_size_estimate:.1f} KB")
    else:
        print(f"Estimated file size: {file_size_estimate:.1f} B")

    # Ask user for confirmation before generating the file
    user_input = input("Do you still want to generate this file? (y/n): ").lower()

    if user_input != "y":
        print("File generation aborted.")
        return

    start_time = time.time()

    # Write to the file in 2 million digit chunks
    chunks = num // 2000000

    if (chunks > 60):
        chunksMinutes = chunks / 60
        if (chunksMinutes == 1):
            print(f"This should take approximately {chunksMinutes:.1f} minute...")
        else:
            print(f"This should take approximately {chunksMinutes:.1f} minutes...")
    else:
        if (chunks == 1):
            print(f"This should take approximately {chunks:.0f} second...")
        else:
            print(f"This should take approximately {chunks:.0f} seconds...")

            

    for _ in range(chunks):
        random_digits = [str(random.randint(0, 9)) for _ in range(2000000)]
        with open(outputFileName, "a") as f:
            f.write(''.join(random_digits))

    # Write the remaining digits (if num wasnt perfectly divisible by 2 million)
    random_digits = [str(random.randint(0, 9)) for _ in range(num-chunks*2000000)]
    with open(outputFileName, "a") as f:
        f.write(''.join(random_digits))

    end_time = time.time()
    elapsed_time = end_time - start_time


    if (elapsed_time > 60):
        elapsed_time /= 60
        print(f"Generated {num} random digits in {elapsed_time:.1f} minutes.")
    else:
        print(f"Generated {num} random digits in {elapsed_time:.0f} seconds.")

if __name__ == "__main__":
    main()