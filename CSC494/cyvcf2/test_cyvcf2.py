from cyvcf2 import VCF

for variant in VCF('./vcftest.vcf.gz'):

    print("chrom\tstart\tend\tID\tREF\tALT\tQUAL\tFILTER")
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
        variant.CHROM, variant.start, variant.end, variant.ID, variant.REF, variant.ALT, variant.QUAL, variant.FILTER))

    print(variant.gt_types)
    print(variant.gt_bases)

    print(variant.INFO.get("DP"))

    print(str(variant))

    # sb = variant.format('AA')
    # print(sb.shape)
