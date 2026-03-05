while true; do
    scp "$1/data/history.csv" data/history.csv
    sleep 30
done
