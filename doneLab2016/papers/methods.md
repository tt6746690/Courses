
[__A survey of best practices for RNA-seq data analysis__](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0881-8)

> A really useful paper detailing RNA-seq experimental preparation and data analysis (emphasis on informatic softwares)

![rnaseq pipeline](https://static-content.springer.com/image/art%3A10.1186%2Fs13059-016-0881-8/MediaObjects/13059_2016_881_Fig1_HTML.gif)

- important to remove rRNA, which comprise 90% of RNA in human. FFPE do not have enough starting material therefore requires rRNA depletion  
- short paired reads is sufficient for well-annotated organisms
- sequencing depth / library size : medium to highly expressed genes requires 10M, 100M is required to quantify precise genes/transcripts with low expression levels  
- although deep sequencing improves quantification and identification, it also introduces transcriptional noise and off-target transcripts
- replication to eliminate biological variability of system and technical variability of RNAseq  

QC
+ Raw reads: analysis of sequence quality, GC content, the presence of adaptors, overrepresented k-mers and duplicated reads in order to detect sequencing errors, PCR artifacts or contaminations. Use _fastQC_ or _NGSQC_. FASTX-Toolkit and Trimmomatic are used to discard low-quality reads  
+ Read alignments: expect 70 to 90% of RNAseq mapped to reference genome (this is a global indicator of overall sequencing accuracy)  If reads primarily accumulate at the 3’ end of transcripts in poly(A)-selected samples, this might indicate low RNA quality in the starting material. The GC content of mapped reads may reveal PCR biases. Tools for quality control in mapping include _Picard_, _RSeQC_ and _Qualimap_

Transcript identification/quantification
+ Mapping solely to the reference transcriptome of a known species precludes the discovery of new, unannotated transcripts and focuses the analysis on quantification alone.
+ Alignments may involve mapping to reference genome or reference transcriptome. Regardless, reads may be mapped uniquely, or could be multi-mapped reads. Genomic multireads are primarily due to repetitive sequences or shared domains of paralogous genes. They normally account for a significant fraction of the mapping output when mapped onto the genome and should not be discarded.  
+ Identification of novel transcript is challenging for short reads, since they rarely span several junctions.   
+ The most common application of RNA-seq is to estimate gene and transcript expression. This application is primarily based on the number of reads that map to each transcript sequence.
+ Aggregate raw counts of mapped reads using programs such as _HTSeq-count_ or _featureCounts_. At the genomic level, these tools often utilize a gene transfer format (GTF) file containing the genome coordinates of exons and genes, and often discard multireads.  
+ Raw read counts alone are not sufficient to compare expression levels among samples, as these values are affected by factors such as transcript length, total number of reads, and sequencing biases. The measure RPKM (reads per kilobase of exon model per million reads) is a within-sample normalization method that will remove the feature-length and library-size effects. This measure and its subsequent derivatives FPKM (fragments per kilobase of exon model per million mapped reads), a within-sample normalized transcript expression measure analogous to RPKs, and TPM (transcripts per million) are the most frequently reported RNA-seq gene expression values.

> Correcting for gene length is not necessary when comparing changes in gene expression within the same gene across samples, but it is necessary for correctly ranking gene expression levels within the sample to account for the fact that longer genes accumulate more reads.

+ Several sophisticated algorithms have been developed to estimate transcript-level expression by tackling the problem of related transcripts’ sharing most of their reads. _Cufflinks_ estimates transcript expression from a mapping to the genome obtained from mappers such as TopHat using an expectation-maximization approach that estimates transcript abundances.


![transcript alignments](https://static-content.springer.com/image/art%3A10.1186%2Fs13059-016-0881-8/MediaObjects/13059_2016_881_Fig2_HTML.gif)


Differential Gene Expression analysis

Alternative splicing analysis

Visualization
+ [_IGV_](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3603213/)

Gene fusion discovery

Integration with other data sets
+ With DNA sequencing for single nucleotide polymorphism (SNP) discovery, RNA-editing analyses, or expression quantitative trait loci (eQTL) mapping.
+ With DNA methylation
+ With miRNA to unravel the regulatory effects of miRNAs on transcript steady-state levels.

Integration and Visiaulization of multiplatform datasets

[__Differential expression in RNA-seq: A matter of depth__](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3227109/)  



---

#### Databases

[__BioXpress: an integrated RNA-seq-derived gene expression database for pan-cancer analysis__](http://database.oxfordjournals.org/content/2015/bav019.full)  
May want to explore public RNAseq datasets for pan-cancer analysis

[_What Databases Are Available For Rna-Seq Datasets?_](https://www.biostars.org/p/46059/)  
+ [GEO](http://www.ncbi.nlm.nih.gov/gds?term=rna-seq)  
+ [ENCODE](http://genome.crg.es/~jlagarde/encode_RNA_dashboard/) (for cell lines)  
+ [TCGA](http://cancergenome.nih.gov/)
+ [SRA](http://www.ncbi.nlm.nih.gov/sra)  
+ [1k GENOME](http://www.1000genomes.org/category/rnaseq)
