set -e

if [ "$#" != 2 ]; then
	echo "submit.sh expects two arguments-- an integer and a string"
else
	kaggle competitions submit -c jigsaw-toxic-comment-classification-challenge -f ./Results/"$1"/submission.csv -m "$2"
fi
