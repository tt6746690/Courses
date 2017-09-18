import genome_kit as gk
from genome_kit import Interval
from genome_kit import Genome

# gets sequence in DNA
interval = Interval("chr7", "+", 117120016, 117120201, "h37")
genome = Genome("h37")
genome.dna(interval)

# hierarchies
genome = Genome("gencode.v19")
gene = genome.genes["ENSG00000001626.10"]
tran = gene.transcripts[2]
exon = tran.exons[0]
genome.dna(exon)

# tracks
# multidimensional (multiple columns)
# stranded (unique data on each strand) or unstranded (same data for both strands)
# track data extracted in sense-strand (+ or 5->3 same sequence as mRNA) order
# instance of `GenomeTrack` but can build custom track with GenomeTrackBuilder
interval = Interval("chr17", "+", 41246881, 41246886, "h37")
genome = Genome("h37")
genome.phastcons_mammal_46way(interval)


# Annotations
# exons is an instance of ExonTable
genome = Genome("gencode.v19")
for gene in genome.genes:
    print(gene)
    for tran in gene.transcripts:
        print(" ", tran)
        for exon in tran.exons:
            print("    ", exon)
# <Gene ENSG00000223972.4 (DDX11L1) >
#    <Transcript ENST00000456328.2 of DDX11L1 >
#       <Exon 1 / 3 of ENST00000456328.2 >
#       <Exon 2 / 3 of ENST00000456328.2 >
#       <Exon 3 / 3 of ENST00000456328.2 >


# Interval
#   -- stranded +/-
#   -- 0 based internally
#   -- span of an interval excludes end position


#  0123456789
#  aaaaabbbbb
#     cccc
#     d
a = Interval("chr1", "+", 0, 5, "h38")
b = Interval("chr1", "+", 5, 10, "h38")
c = Interval("chr1", "+", 3,  7, "h38")
d = Interval("chr1", "+", 3,  4, "h38")

assert(len(a) == 5)         # OK
assert(a.contains(d))       # OK
assert(a.overlaps(c))       # OK    a does not overlap b
assert(a.upstream_of(b))    # OK


x = a.as_opposite_strand()
# Interval("chr1", "-", 0, 5, "h38")
y = b.as_opposite_strand()
# Interval("chr1", "-", 5, 10, "h38")
z = c.as_opposite_strand()
# Interval("chr1", "-", 3, 7, "h38")
w = d.as_opposite_strand()
# Interval("chr1", "-", 3, 4, "h38")


assert(not x.overlaps(d))
assert(not x.contains(d))
assert(x.contains(w))


# build arount 5 and 3 end
interval = Interval("chr1", "-", 4,  8, "h38")
interval.end5
# Interval("chr1", "-", 8, 8, "h38")         empty
interval.end3
# Interval("chr1", "-", 4, 4, "h38")         empty
interval.end3.expand(2, 3)
# Interval("chr1", "-", 1, 6, "h38")


# Feature extractions
a = Interval("chr7", "+", 117120016, 117120201, "h37")
b = a.as_opposite_strand()

# seems: position is absolute, and dna() always gives sequence in 5->3
# direction
genome.dna(a)
# 'AATTGGAAGCAAA...AACTTTTTTTCAG'
genome.dna(b)                       # reverse complement sequence
# 'CTGAAAAAAAGTT...TTTGCTTCCAATT'


# Variants(chrom, ref_pos, ref, alt)
# more general, as allow empty ref/alt alleles (i.e. chr7"117120150:A:- > CA:C)
# empty "" or  "-" or "."
# ref allele checked upon initialization
genome = Genome("h37")
variant = Variant("chr7", 117120149, "T", "G", genome)
# variant = Variant.from_string('chr7:117,120,150:A:G', genome)


# VariantGenome
# reference genome PLUS variants applied to it
# extract DNA sequence flanking 5' end of CFTR transcript
def extract_feature(genome):
    tran = genome.transcripts["ENST00000426809.1"]  # CFTR transcript
    span = tran.end5.expand(2, 5)                    # 7nt span at 5' end
    return genome.dna(span)


ref = Genome("gencode.v19")
variants = [Variant.from_string("chr7:117120149:A:G", ref),     # rs397508328
            Variant.from_string("chr7:117120151:G:T", ref)]     # rs397508657
var = VariantGenome(ref, variants)
print(extract_features(ref))
print(extract_features(var))


# Motif finding
genome = Genome('h37')
# Short sequence from CFTR
interval = Interval('chr7', '+', 117231957, 117232030, genome)
genome.dna(interval)
# 'TTGATATTTATATGTTTTTATATCTTAAAGCTGTGTCTGTAAACTGATGGCTAACAAAACTAGGATTTTGGTC'
motif = 'AACAA'
matches = genome.find_motif(interval, motif)
matches
# [Interval("chr7", "+", 117232009, 117232009, "h37", 117232009)]
