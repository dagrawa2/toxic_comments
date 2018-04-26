set -e

echo "Running create_dirs.sh . . . "
sh create_dirs.sh

python run_all.py

echo "Running gather_results.sh . . . "
sh gather_results.sh
