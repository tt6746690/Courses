
#### Questions 




```cpp 
void genome_dna::open_on_demand() const
{
	// TODO: acquire lock here and check _file for NULL again

	// Once we're inside the lock it's safe to use ncthis
	genome_dna* ncthis = const_cast<genome_dna*>(this);
	ncthis->open();

	// TODO: release lock here (implicitly, when falls out of scope)
}
```
+ how does `open_on_demane` works here, just a cast from `this`? 



#### Facts

+ _packing_ 
    + `packed` first dont have pointer type so serialization friendly
    + unpacked version used for accessing fields (especially C-string) 

+ why is `_aux` a `vector<char>`  
    + because used to store variable length C-string
    + indexed by `offset_t aux` in `packed_`


+  `genome.dna(interval)` always return DNA in 5->3 direction, regardless of strandedness

+ `interval_table`
    + `T`: usually a derived class of `interval_t`, some kind of `packed_xxx_t`




```cpp
#define UNPACK_INTERVAL(c) \
	pos_t pos5 = c.pos5; DG_MARK_USED(pos5); \
	pos_t pos3 = c.pos3; DG_MARK_USED(pos3); \
	sys_t sys = c.sys; DG_MARK_USED(sys); \
	chrom_t chrom = c.chrom; DG_MARK_USED(chrom); \
	refg_t refg = c.refg; DG_MARK_USED(refg); \
	strand_t strand = c.strand; DG_MARK_USED(strand) 
```
+ `DG_MARK_USED (x) x = x` 
    + eliminates variable unused compiler warning



```cpp
<template <typename> class P> 
void foo();
```
+ `P` is a template class that should be instantiated with exactly 1 template argument
+ omitting the name for nested `<typename>` because not relevant in this context





```cpp 
template <typename T>
struct interval_idx {
	pos_t max_interval_size[num_chrom][num_strand];
	array_view<index_t> by_pos5[num_chrom][num_strand];
	array_view<index_t> by_pos3[num_chrom][num_strand];
}
```
+ How does `interval_idx` work
    + indexing functionality for `table`, usually an `array_view` of indices to the memory mapped table, in other words an array of derived class of `interval_t`
    + `array_view<indices_t> by_pos5[i][j]`: 
        + a view into a vector of indices built for each chromosome/strand combination
        + each index `idx` is one such that `_elems[idx]` is the refering elements 
        + used in `find_3p_within`, `find_3p_aligned`, ...
    + why both `by_pos5` and `by_pos3`
        + for `find_5p_within`, `find_5p_aligned`, ...
    + what does `max_interval_size` do? 
        + holds maximum length of any interval on chromosome `i` and strand `j`. 
        + value used to implement overlap queries using just `by_pos5` and `by_pos3` indices, without going into the trouble of implementing an interval tree data structure in the format (which can be large) 



```cpp 
const packed_intr *get_prev_intr(const packed_exon *exon, const genome_anno &anno)
{
	const packed_tran &tran = anno.trans[exon->tran];
	if (exon->index == 0)
		return 0;
	return &anno.intrs[tran.intr0 + exon->index - 1];
}
```
+ as an example 
    + `A`, `C` intron, `B`, `D` exon
    + together form `tran`
    ```
    AAAAABBBBBBCCCCCCDDDDD
    0     1     2     3
    ``` 
    + is `A` and `C` stored contiguously in `anno.intrs`? 
        + Yes, consecutive `packed_intr` within a single transcript are stored consecutively in `intrs` table 
        + so `intrs[tran.intr0]` is `A` and `intrs[tran.intr0+1]` is `C`.





```cpp 
genome_anno::genome_anno()
: genes(*this)
, trans(*this)
, exons(*this)
, intrs(*this)
, cdss(*this)
// <--- INSERT TABLE CONSTRUCTORS HERE
{ 
}
```
+ how does passing `*this` to initializer expression works here?
    + `genome_anno_table<T>` requires a reference to `genome_anno` in its constructor (as parent object with the other `_table<T>`s)
        + so no default constructor
    + `*this` resolves to a reference to `genome_anno` instance being constructed, which is what `gene_table`, `trans_table`... needed 