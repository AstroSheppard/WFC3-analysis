Retrieval codes, both for PLATON v3 (what I used for HAT-P-41b) and PLATON v5 (newest version).

retrieve/retrieve_v5.py set appropriate priors, read in data, and invoke PLATON retriever. Works with Mie scattering, partial clouds, offsets, etc. It saves the evidence, 3 sigma ranges, and median water abundance in bestfit directory.

%run retrieve.py 0 save_name mie_condensate

0 indicates transit, 1 indicates eclipse. If not doing Mie scattering, leave condensate blank (otherwise it reads in data file from optical_constants_mie)



