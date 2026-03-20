@echo off
echo ============================================================
echo Testing Grounding Improvements
echo ============================================================
echo.

echo Step 1: Quick test (5 questions)
echo ----------------------------------------
python test_grounding.py
echo.

echo Step 2: Generate full submission
echo ----------------------------------------
python hack\generate_submission.py
echo.

echo Step 3: Check format
echo ----------------------------------------
python check_single_doc.py
echo.

echo Step 4: Run diagnostics
echo ----------------------------------------
python hack\test_diagnostic.py
echo.

echo ============================================================
echo Testing complete!
echo ============================================================
pause
