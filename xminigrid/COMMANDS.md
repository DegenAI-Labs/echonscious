
1. source activate py_model_welfare_echo [or whatever your environment is named]. 
   source ~/.llmapikeys [source your llm api keys from wherever, can be global or a local .env you have made here.] 


2. Run the run script that for environments 6 to 16, visits all seeds 0 to 16:

bash scripts/run.sh echo gpt-4o small-1m


3. Collate the results generated for various models, methods, envnumbers [to be figured out if these map one one to seeds, or all past seeds are visited in each env] 

python scripts/collate.py --results-dir results/small-1m --output sampleoutput
