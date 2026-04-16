@echo off

REM usage: feature.py [-h] train_input validation_input test_input feature_dictionary_in train_out validation_out test_out
set TR_IN=smalldata\train_small.tsv
set V_IN=smalldata\val_small.tsv
set TE_IN=smalldata\test_small.tsv
set F_D=glove_embeddings.txt
set TR_OUT=my/train_formatted.tsv
set V_OUT=my/val_formatted.tsv
set TE_OUT=my/test_formatted.tsv

echo python feature.py %TR_IN% %V_IN% %TE_IN% %F_D% %TR_OUT% %V_OUT% %TE_OUT%
python feature.py %TR_IN% %V_IN% %TE_IN% %F_D% %TR_OUT% %V_OUT% %TE_OUT%

REM lr.py [-h] train_input validation_input test_input train_out test_out metrics_out num_epoch learning_rate
set LR_TR_IN=my/train_formatted.tsv
set LR_V_IN=my/val_formatted.tsv
set LR_TE_IN=my/test_formatted.tsv
set LR_TR_OUT=my/train_labels.txt
set LR_TE_OUT=my/test_labels.txt
set LR_METRICS_OUT=my/metrics.txt
set NUM_EPOCH=500
set LEARNING_RATE=0.1

echo python lr.py %LR_TR_IN% %LR_V_IN% %LR_TE_IN% %LR_TR_OUT% %LR_TE_OUT% %LR_METRICS_OUT% %NUM_EPOCH% %LEARNING_RATE%
python lr.py %LR_TR_IN% %LR_V_IN% %LR_TE_IN% %LR_TR_OUT% %LR_TE_OUT% %LR_METRICS_OUT% %NUM_EPOCH% %LEARNING_RATE%

echo.
echo Process finished.
pause
