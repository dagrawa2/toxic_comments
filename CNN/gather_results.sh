set -e

if [ ! -d Results/ ]; then
	echo "Creating Results directory"
	mkdir Results
fi

python gather_results.py
