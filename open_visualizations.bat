@echo off
echo ========================================
echo Opening Embedding Visualizations
echo ========================================
echo.

echo Opening 3D UMAP visualization...
start "" "visualizations\3d_umap.html"

timeout /t 2 /nobreak >nul

echo Opening Distribution Analysis...
start "" "visualizations\distribution_analysis.html"

timeout /t 2 /nobreak >nul

echo Opening Dimension Analysis...
start "" "visualizations\dimension_analysis.html"

echo.
echo ========================================
echo All visualizations opened in browser!
echo ========================================
echo.
echo Available files:
echo   - 3d_umap.html (opened)
echo   - 3d_pca.html
echo   - 2d_umap.html
echo   - distribution_analysis.html (opened)
echo   - dimension_analysis.html (opened)
echo.
pause
