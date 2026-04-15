@echo off
echo =============================================
echo Running feature.py with following arguments:
echo 
echo =============================================

REM usage: feature.py [-h] train_input validation_input test_input feature_dictionary_in train_out validation_out test_out
set TR_IN=smalldata\train_small.tsv
set V_IN=smalldata\val_small.tsv
set TE_IN=smalldata\test_small.tsv
set F_D=
set TR_OUT=
set V_OUT=
set TE_OUT=

python feature.py %TR_IN% %V_IN% %TE_IN% %F_D% %TR_OUT% %V_OUT% %TE_OUT%

echo.
echo Process finished.
pause
