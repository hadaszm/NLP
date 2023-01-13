Steps to reproduce the results:

1. Download files from https://drive.google.com/drive/folders/1-lAjO3VyEMt0KWbpxhLRIx_L7h_SIuaZ?usp=sharing
2. `cd data_for_script`
3. Run script with: 
	`python script.py prepare_data=True` <br />
	It takes long time so you can instead use:  <br />
	`python script.py prepare_data=False` <br />
	It will skip initial step.  <br />
	If you want to stop the program and run it later you can use timestamp from filenames of results for example:  <br />
	`python script.py prepare_data=False timestamp=2023-01-13_16-21-53`