#!/bin/bash
echo "1 - adaboost\n2 - bagging\n3 - 100 bagged predictors\n4 - Ranfom Forest\n5 - 100 Random Forests"
read -p 'Choice: ' choice

if [ "$choice" == 1 ]; then
    python ID3_adaboost.py
fi

if [ "$choice" == 2 ]; then
    python ID3_Bagging.py
fi

if [ "$choice" == 3 ]; then
    python ID3_bagging_100.py
fi

if [ "$choice" == 4 ]; then
    python ID3_RandomForest.py
fi

if [ "$choice" == 5 ]; then
    python ID3_RandomForest_100.py
fi
