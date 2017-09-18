
#### Sep 11, 2017 

+ _steps_   
    + Performance 
        + of cyvcf2, read source to see why its slower potentially 
    + understand the access pattern that maybe slow
        + column based-query (but once extend to multiple samples, searches in matrix would be equivalent) 
    + understand existing VCFTable impl how it could be optmized 
        + `mmap` does the trick already ?
        + simply transpose matrix? since just `(CHROM+POS, (GT, DP, AD))` are relevant 
    + extends VCFTable to handle multiple samples 
        + data structure for sparse matrix 
        + interface with Python stack
            + use C++ interface for numpy..
    + provide cyvcf2 like interface,  [optional]
        + Cython wrapper around underlying impl
+ _todos_   
    + pre-computed matrix `n_variants X n_samples` for accessing `GT = {0, 1, 2, 3}`, `AF = AD_alt / AD_total`
