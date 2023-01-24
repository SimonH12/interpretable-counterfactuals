# interpretable-counterfactuals

Bachelor Thesis - Assessing scores for improving the human-friendliness and interpretability of counterfactual explanations


## Description
In this thesis, the problem of selecting counterfactuals from a list of possible CFs
based on how interpretable they are for humans will be tackled. Most research
focuses on simple technical properties of the CFs. In this work, most of all research
from the field of psychology will be used to identify new strategies for recognizing
good CFs.
In order to generate counterfactuals, the algorithm DiCE (https://github.com/interpretml/DiCE) is used.


## Reproduce experiments
In order to reproduce the experiments from the bachelor thesis do the following:

- [ ] pip install -r requirements.txt
- [ ] run the tests.py file
- [ ] results of the tests are in 'interpretable-counterfactuals\unit_tests\unit_test_table.csv' and 'interpretable-counterfactuals\real_world_data\real_world_table.csv'

To reproduce plots of the concepts used in the thesis run 'Bachelor-Bench\code\plots_concepts.py'
