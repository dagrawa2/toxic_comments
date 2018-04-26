set -e

if [ ! -d Objects/ ]; then
	echo "Creating Objects directory"
	mkdir Objects
fi

if [ ! -d Out/ ]; then
	echo "Creating Out directory"
	mkdir Out
fi
