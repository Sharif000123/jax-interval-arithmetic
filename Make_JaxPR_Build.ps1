Remove-Item -Recurse -Force build

# create fresh build dir and configure using the venv python
New-Item -ItemType Directory -Path build
Set-Location build

cmake -G "Ninja" -DPython_EXECUTABLE="D:\Work\ML_Stuff\formalax\.venv\Scripts\python.exe" ..
cmake --build . --config Release

pause