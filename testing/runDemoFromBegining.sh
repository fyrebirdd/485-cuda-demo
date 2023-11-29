if [ "$#" -ne 2 ]
then
  echo "USEAGE: runDemoFromBegining.sh <numberRows> <targetChar>"
  exit 1
fi

echo "Generating $1 x $1 matrix..."
python3 testFileGen.py $1 "matrix.txt"

echo "Running Computaions..."
python3 runTest.py "matrix.txt" $1 $2