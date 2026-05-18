@echo off
REM Bhairav overnight: manifest -> mine 500 -> retrieve 220
REM Usage: double-click or run from cmd:
REM   cd /d "d:\New folder (11)"
REM   eval\run_overnight.bat

cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set LOG=eval\results\overnight_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG=%LOG: =0%

echo ============================================================ >> "%LOG%"
echo BHAIRAV OVERNIGHT START %date% %time% >> "%LOG%"
echo ============================================================ >> "%LOG%"

echo [1/3] Building manifest (per-source 75)...
echo [1/3] Building manifest... >> "%LOG%"
python -m eval.build_manifest --per-source 75 >> "%LOG%" 2>&1
if errorlevel 1 (
    echo FAILED at build_manifest. See %LOG%
    exit /b 1
)

echo [2/3] Mining 500 triplets (Groq)...
echo [2/3] Mining... >> "%LOG%"
python -m eval.mine_triplets --max-chunks 500 --questions-per-chunk 1 --delay 2.5 >> "%LOG%" 2>&1
if errorlevel 1 (
    echo FAILED at mine_triplets. See %LOG%
    exit /b 1
)

echo [3/3] Retrieval eval 220 queries (CPU, server must be OFF)...
echo [3/3] Retrieval eval... >> "%LOG%"
python -m eval.bhairav_retrieval_eval --triplets eval\data\triplets.jsonl --max-queries 220 --stage after_rerank >> "%LOG%" 2>&1
if errorlevel 1 (
    echo FAILED at retrieval eval. See %LOG%
    exit /b 1
)

echo ============================================================ >> "%LOG%"
echo BHAIRAV OVERNIGHT DONE %date% %time% >> "%LOG%"
echo ============================================================ >> "%LOG%"

echo.
echo ALL DONE. Log: %LOG%
echo Check eval\results\retrieval_after_rerank_*.json
exit /b 0
